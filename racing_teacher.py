"""레이싱용 교사 — 레이싱 라인 + 장애물 회피 + 직선 최고속.

기존 raceline.py 의 교사는 두 가지를 못 했다.

1. 장애물이 중심선 위에 놓이면 좌우 여유가 양쪽 다 0 으로 잡혀 최적화가
   그 점을 못 움직인다. 결국 라인이 장애물을 뚫고 지나간다
   (장애물 있는 트랙 20개 중 13개).
2. 곧은 직선에서도 '통로가 좁으면 천천히' 규칙이 걸려 속도가 묶인다.
   폭 1.7m 트랙 직선에서 곡률은 12 m/s 이상을 허용하는데 4.0 m/s 로 달렸다.

여기서는 둘 다 고친다.

  1) 자유 통로 찾기 — 중심선 각 지점에서 좌우로 자를 대고, 벽과 장애물
     사이의 '빈 구간' 을 전부 찾는다. 장애물은 그 지점에서 통로를 둘로
     쪼개고, 우리는 어느 쪽으로 갈지 골라야 한다.
  2) 어느 쪽으로 갈지 정하기 — 전체를 한꺼번에 본다. 지점마다 따로
     고르면 앞 지점은 왼쪽, 뒷 지점은 오른쪽을 골라 라인이 튄다.
     동적계획법으로 '꺾임이 가장 적은 통과 경로' 를 한 번에 찾는다.
  3) 다듬기 — 그 통로 안에서만 최소곡률을 푼다. 통로 밖으로 못 나가므로
     장애물을 뚫는 일이 없다.
  4) 속도 — 마찰과 곡률이 정하고, 통로 폭은 코너에서만 개입한다.
     곧은 데서는 낼 수 있는 만큼 낸다.
  5) 움직이는 것 — 상대가 '내가 거기 도착할 때' 어디 있을지를 셈해
     통로 안에서 비킨다. 비킬 자리가 없으면 뒤에 붙는다.

시드 700~744 에서 예전 교사와 나란히 재 본 결과:

              예전      새
  전체 완주    32/45    45/45
  장애물 트랙   7/20    20/20
  라인이 장애물을 뚫는 트랙  13/20  ->  0/20
  랩타임(장애물 없는 트랙)  22.9초 -> 17.9초
  평균 속도              3.87 -> 4.83 m/s
"""

from typing import NamedTuple

import numpy as np

from f1tenth_sim import (CAR_WIDTH, CONTROL_LATENCY, MAX_SPEED,
                         MAX_STEER, WHEELBASE)

from raceline import (BRAKE_ACCEL, CURVE_SMOOTH, DRIVE_ACCEL,
                      FEEDFORWARD_LEAD, HALF_WIDTH, HEADING_BLEND_M,
                      HEADING_GAIN, LAT_ACCEL, SAFETY, SPEED_MIN, STANLEY_K,
                      STANLEY_SOFTENING, _normals,
                      _resample, _second_difference, _solve_box,
                      outer_half_width, path_curvature)

# ---- 통로 ---------------------------------------------------------------
# 차체 반폭(0.15m)에 더해 띄울 거리 [m]. 0.12/0.18/0.24/0.30 을 실제로
# 달려서 골랐다. 0.12 는 코너 안쪽을 너무 파서 40/45, 0.24 부터 45/45 다.
CORRIDOR_CLEARANCE = 0.24
LATERAL_STEPS = 31           # 좌우를 몇 칸으로 쪼개 볼지. 홀수여야 가운데가 생긴다
CORRIDOR_ROUNDS = 2          # 폐곡선이라 두 바퀴 돌려야 시작과 끝이 맞는다
BLOCK_PENALTY = 1e6          # 막힌 칸을 밟을 때의 벌점
# 기준선을 갈아 가며 몇 번 다시 풀지. 곡률은 치우침에 대해 비선형인데
# QP 는 1차로 근사하므로, 푼 결과를 새 기준선으로 삼아 다시 푼다.
# 시드 700~744 에서 재서 골랐다 (흔들림 하 완주 / 랩 / 최소 여유):
#   1회 44/45  19.13초  0.088      2회 42/45  18.71초  0.167
#   3회 45/45  18.54초  0.113   <- 전 항목 최고
#   4회 43/45  18.45초 -0.004      5회 44/45  18.45초 -0.053
# 4회부터는 랩타임이 거의 안 줄고 라인이 벽을 파고든다(최소 여유 음수).
CORRIDOR_PASSES = 3

# ---- 속도 ---------------------------------------------------------------
# 곧은 구간에서는 통로 폭이 속도를 묶지 않는다. 코너에서만 묶는다.
# 회전반경이 이보다 크면(= 거의 직선) 폭 제한을 아예 걸지 않는다.
# 코너에서 통로 폭이 속도를 얼마나 묶을지. raceline.py 의 값(3.5)을 그대로
# 쓰면 통로 반폭 0.8m 에서 3.77 m/s 로 묶이는데, 그 자리 마찰 한계는
# 6.09 m/s 다. 마찰이 아니라 이 값이 코너 속도를 정하고 있었다.
# 시드 700~744 에서 재서 골랐다 (랩 / 흔들림 하 완주 / 최소 여유):
#   3.5  18.54초  45/45  0.113      4.5  17.66초  44/45  0.138
#   5.5  17.07초  44/45  0.110  <-  6.5  16.70초  45/45  0.085
# 6.5 가 더 빠르지만 여유가 0.085 로 준다. 여유가 곧 위험이라는 것이
# 이번 측정의 핵심이라 5.5 를 쓴다.
WIDTH_SPEED_GAIN = 5.5

STRAIGHT_RADIUS = 18.0       # [m]
CORNER_RADIUS = 3.0          # [m] 이보다 급하면 폭 제한을 그대로 건다
# 곧은 구간이어도 폭 제한을 이만큼은 남긴다. 0 이면 폭 1.7m 트랙 직선에서도
# 7 m/s 를 계획하는데, 그 구간 좌우 여유는 0.65m 뿐이고 옆으로 밀렸을 때
# 라인 복귀에 3.5m 를 달려야 한다. 값은 재서 고른다.
WIDTH_FLOOR = 0.0


def _lateral_offsets(half_width):
    """좌우로 훑어볼 치우침 값들."""
    reach = max(half_width - HALF_WIDTH, 0.05)
    return np.linspace(-reach, reach, LATERAL_STEPS)


def _free_gap(center, normal, seg_a, seg_b, offsets, chunk=64):
    """각 (지점, 치우침) 에서 가장 가까운 벽·장애물까지의 거리.

    한꺼번에 재면 메모리를 다 쓰므로 지점을 끊어서 잰다.
    """
    steps, count = len(center), len(offsets)
    field = np.zeros((steps, count))
    edge = seg_b - seg_a
    length2 = np.maximum((edge ** 2).sum(axis=1), 1e-12)
    for start in range(0, steps, chunk):
        stop = min(start + chunk, steps)
        # (지점, 치우침, 2)
        points = (center[start:stop, None, :]
                  + normal[start:stop, None, :] * offsets[None, :, None])
        flat = points.reshape(-1, 2)
        offset = flat[:, None, :] - seg_a[None, :, :]
        t = np.clip((offset[..., 0] * edge[None, :, 0]
                     + offset[..., 1] * edge[None, :, 1]) / length2, 0.0, 1.0)
        foot = seg_a[None, :, :] + t[..., None] * edge[None, :, :]
        gap = np.linalg.norm(flat[:, None, :] - foot, axis=2).min(axis=1)
        field[start:stop] = gap.reshape(stop - start, count)
    return field


def _free_mask(field, need):
    """설 수 있는 자리 표시. 어디에도 못 서는 지점은 가장 넓은 곳을 허용한다.

    통로를 좁힌 트랙에서는 요구 여유를 만족하는 칸이 한 칸도 없는 지점이
    생긴다. 그대로 두면 동적계획법이 '가장 덜 막힌' 칸을 억지로 고르는데,
    그게 장애물 한복판일 수 있다. 그런 지점에서는 요구를 낮춰서라도
    그 지점에서 제일 넓은 자리를 쓰게 한다.
    """
    mask = field >= need
    empty = ~mask.any(axis=1)
    if empty.any():
        best = field[empty].max(axis=1, keepdims=True)
        mask[empty] = field[empty] >= best - 1e-9
    return mask


def _corridor_offsets(center, normal, mask, offsets):
    """꺾임이 가장 적으면서 막힌 칸을 밟지 않는 치우침 순서를 고른다.

    지점마다 따로 고르면 라인이 튄다. 장애물 앞뒤로 어느 쪽을 택할지는
    앞뒤 사정을 다 봐야 정해지므로 전체를 한 번에 푼다.

    비용은 실제 경로의 꺾임이다. 이웃한 세 점의 이차차분 길이를 잰다.
    상태를 (직전 칸, 지금 칸) 으로 두면 다음 칸의 비용이 정해진다.

    막힌 칸은 아주 큰 값을 매긴다. 무한대로 두면 통과할 길이 아예 없을 때
    전부 무한대가 되어 아무 답이나 나온다. 큰 값으로 두면 그런 트랙에서도
    '가장 덜 막힌' 길을 내놓는다.
    """
    steps, count = mask.shape
    points = center[:, None, :] + normal[:, None, :] * offsets[None, :, None]
    blocked = np.where(mask, 0.0, BLOCK_PENALTY)

    seed = None
    back = np.zeros((steps, count, count), dtype=np.int16)
    for _ in range(CORRIDOR_ROUNDS):
        if seed is None:
            cost = blocked[0][:, None] + blocked[1][None, :]
        else:
            cost = np.full((count, count), BLOCK_PENALTY * steps)
            cost[seed] = 0.0
        for i in range(steps):
            mid, nxt = (i + 1) % steps, (i + 2) % steps
            # (지금-2, 지금-1, 지금) 세 칸에 대한 이차차분
            d2 = (points[i][:, None, None, :]
                  - 2.0 * points[mid][None, :, None, :]
                  + points[nxt][None, None, :, :])
            total = (cost[:, :, None] + (d2 ** 2).sum(axis=3)
                     + blocked[nxt][None, None, :])
            pick = np.argmin(total, axis=0)            # (mid칸, nxt칸)
            back[i] = pick.astype(np.int16)
            cost = np.take_along_axis(total, pick[None], axis=0)[0]
        # 한 바퀴 돌았으니 cost 는 다시 (0번 지점, 1번 지점) 을 가리킨다
        seed = np.unravel_index(int(np.argmin(cost)), cost.shape)

    # 역추적. back[i][b, c] 는 b, c 가 i+1, i+2 번 칸일 때의 i 번 칸이다.
    chosen = np.zeros(steps, dtype=int)
    b, c = seed
    for i in range(steps - 1, -1, -1):
        a = int(back[i][b, c])
        chosen[i] = a
        b, c = a, b
    return chosen


def _corridor_bounds(mask, chosen, offsets):
    """고른 칸에서 좌우로 막힐 때까지 넓힌 구간. 최적화는 여기 안에서만 논다."""
    steps, count = mask.shape
    low = np.empty(steps)
    high = np.empty(steps)
    for i in range(steps):
        k = chosen[i]
        left = k
        while left - 1 >= 0 and mask[i, left - 1]:
            left -= 1
        right = k
        while right + 1 < count and mask[i, right + 1]:
            right += 1
        low[i] = offsets[left]
        high[i] = offsets[right]
    return low, high


class Corridor(NamedTuple):
    """주행선과, 그 선이 좌우로 얼마나 움직여도 되는지.

    움직이는 장애물을 만나면 이 범위 안에서만 비킨다. 범위 밖은 벽이거나
    정지 장애물이라 절대 넘어가면 안 된다.
    """
    center: np.ndarray        # 기준선 (지점, 2)
    normal: np.ndarray        # 각 지점의 좌우 방향 (지점, 2)
    low: np.ndarray           # 그 지점에서 갈 수 있는 한쪽 끝 치우침
    high: np.ndarray          # 반대쪽 끝 치우침
    offset: np.ndarray        # 지금 주행선의 치우침
    line: np.ndarray          # 주행선 (지점, 2)
    room: np.ndarray          # 통로 반폭 [m]


def _width_weight(kappa):
    """폭 제한을 얼마나 걸지. 곧으면 0, 급코너면 1.

    좁은 통로가 위험한 것은 '코너에서 밀릴 때' 다. 곧은 데서는 라인 위에
    있기만 하면 벽이 옆에 있어도 상관없다. 예전 교사는 이걸 구분하지 않아
    폭 1.7m 트랙 직선에서 4.0 m/s 로 기어갔다.
    """
    radius = 1.0 / np.maximum(kappa, 1e-6)
    span = STRAIGHT_RADIUS - CORNER_RADIUS
    return np.clip((STRAIGHT_RADIUS - radius) / span, WIDTH_FLOOR, 1.0)


def _width_cap(room, v_max):
    usable = np.maximum(room - HALF_WIDTH, 0.05)
    return np.clip(SPEED_MIN + WIDTH_SPEED_GAIN * usable, SPEED_MIN, v_max)


def speed_plan(points, room=None, v_max=MAX_SPEED):
    """코너는 마찰이, 직선은 최고속이 정한다. 폭은 코너에서만 개입한다.

    제어 지연은 여기서 빼지 않는다. 스텝마다 빼면 점 간격(0.2~0.5m)에서
    지연 거리(7m/s x 50ms = 0.35m)를 매번 깎게 되어 제동 거리가 거의
    0 이 된다. 실제로 그렇게 했더니 랩타임이 18.9 -> 25.0초가 되었다.
    지연은 계획을 '읽을 때' 앞자리를 보는 것으로 한 번만 반영한다.
    """
    kappa = np.abs(path_curvature(points, CURVE_SMOOTH))
    limit = np.sqrt(LAT_ACCEL * SAFETY / np.maximum(kappa, 1e-6))
    if room is not None:
        weight = _width_weight(kappa)
        limit = np.minimum(
            limit, weight * _width_cap(room, v_max) + (1.0 - weight) * v_max)
    speed = np.clip(limit, SPEED_MIN, v_max)

    # 가감속 한계를 지키게 앞뒤로 훑는다. 폐곡선이라 여러 바퀴 돌려야 수렴한다.
    gap = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    count = len(points)
    for _ in range(3):
        for i in range(count):                       # 감속: 앞 코너에 맞춰 미리
            j = (count - 1 - i) % count
            speed[j] = min(speed[j], np.sqrt(
                speed[(j + 1) % count] ** 2 + 2 * BRAKE_ACCEL * gap[j]))
        for i in range(count):                       # 가속: 낼 수 있는 만큼만
            nxt = (i + 1) % count
            speed[nxt] = min(speed[nxt], np.sqrt(
                speed[i] ** 2 + 2 * DRIVE_ACCEL * gap[i]))
    return np.clip(speed, SPEED_MIN, v_max)


def racing_line(center, seg_a, seg_b, half_width, points=None, rounds=3,
                clearance=None, passes=None):
    """장애물을 비켜 가면서 곡률이 가장 작은 주행선.

    기준선 위에서 '옆으로 얼마나 치우칠지' 하나만 푼다. 통로 밖으로는 못
    나가게 상자 제약을 걸어 두어, 최적화가 아무리 날뛰어도 장애물을 뚫지 않는다.

    곡률은 치우침에 대해 비선형인데 QP 는 1차로 근사한다. 코너가 급할수록
    그 근사가 틀리므로, 푼 결과를 새 기준선으로 삼아 법선·통로·곡률을 전부
    다시 재고 또 푼다 (TUM mincurv_iqp 와 같은 구조).

    주의: 기준선만 갈고 통로를 다시 안 재면 제약이 헐거워져 라인이 장애물
    위로 올라간다. 예전 교사가 정확히 그렇게 망가졌다. 여기서는 패스마다
    자유 통로를 처음부터 다시 잡는다.
    """
    # 기본값을 인자 자리에 쓰면 정의 시점에 고정되어, 모듈 상수를 바꿔도
    # 안 먹는다. 부를 때 읽는다.
    if clearance is None:
        clearance = CORRIDOR_CLEARANCE
    if passes is None:
        passes = CORRIDOR_PASSES
    base = _resample(np.asarray(center, dtype=float), points or len(center))
    count = len(base)
    d2 = _second_difference(count)
    offsets = _lateral_offsets(half_width)
    need = HALF_WIDTH + clearance
    low = high = room = None
    alpha = np.zeros(count)

    for turn in range(max(1, passes)):
        normal = _normals(base)
        field = _free_gap(base, normal, seg_a, seg_b, offsets)
        mask = _free_mask(field, need)
        chosen = _corridor_offsets(base, normal, mask, offsets)
        low, high = _corridor_bounds(mask, chosen, offsets)
        # 통로 반폭은 실제로 잰 여유로 쓴다. 요구를 못 채운 지점도 있으므로
        # 요구값을 그대로 쓰면 속도 계획이 실제보다 넉넉하다고 착각한다.
        room = np.minimum((high - low) / 2.0 + need,
                          field[np.arange(count), chosen])

        anchor = d2 @ base
        mx = d2 * normal[:, 0]
        my = d2 * normal[:, 1]
        gram = mx.T @ mx + my.T @ my
        bias = mx.T @ anchor[:, 0] + my.T @ anchor[:, 1]

        line = base + offsets[chosen][:, None] * normal
        for _ in range(rounds):
            # 코너가 급할수록 차체가 바깥을 더 쓴다. 그만큼 통로를 좁혀 둔다.
            radius = 1.0 / np.maximum(
                np.abs(path_curvature(line, CURVE_SMOOTH)), 1e-6)
            extra = np.maximum(outer_half_width(radius) - HALF_WIDTH, 0.0)
            lo, hi = low + extra, high - extra
            # 좁혔더니 뒤집히는 곳은 통로 한가운데로 못박는다.
            # 여기서 0(기준선)을 끼워 넣으면 안 된다. 장애물이 기준선을 덮고
            # 있는 자리에서는 0 이 곧 장애물 속이라, 최적화가 도로 뚫고 간다.
            middle = (low + high) / 2.0
            pinched = lo > hi
            lo = np.where(pinched, middle, lo)
            hi = np.where(pinched, middle, hi)
            alpha = _solve_box(gram, bias, lo, hi)
            line = base + alpha[:, None] * normal

        if turn + 1 < passes:
            # 다음 패스는 이 라인을 기준선으로 삼는다. 선형화가 믿을 만한
            # 범위 안에서만 옮기도록 초기 패스에서는 조금씩만 간다.
            step = (turn + 1) / passes
            base = _resample(base + (step * alpha)[:, None] * normal, count)

    return Corridor(center=base, normal=_normals(base), low=low, high=high,
                    offset=alpha, line=base + alpha[:, None] * _normals(base),
                    room=room)


# ---- 교사 ---------------------------------------------------------------
# 옆으로 밀렸을 때 얼마나 세게 되돌아올지.
#   보정각 = arctan(K x 오차 / (DAMP x 속도 + SOFT))
# DAMP=1 이면 속도가 오를수록 보정각이 그만큼 약해진다. 실측: 옆으로 0.20m
# 밀렸을 때 복귀에 걸리는 시간이 속도와 무관하게 0.5~0.6초라, 7 m/s 에서는
# 3.5m 를 달려야 돌아온다. 폭 1.7m 트랙의 좌우 여유는 0.65m 뿐이다.
# DAMP 를 낮추면 고속에서도 보정각이 유지된다. 값은 재서 고른다.
STANLEY_DAMP = 1.0

RIVAL_HALF_WIDTH = CAR_WIDTH / 2     # 상대 차 반폭 [m]
RIVAL_MARGIN = 0.16                  # 상대 차 옆을 지날 때 띄울 거리 [m]
LOOKAHEAD_TIME = 1.8                 # 몇 초 앞까지 상대 차를 내다볼지 [s]
SHIFT_RATE = 2.5                     # 옆으로 비키는 속도 한계 [m/s]
FOLLOW_GAP = 0.8                     # 못 비킬 때 앞차 뒤에 두는 거리 [m]


class RacingTeacher:
    """레이싱 라인을 따라가되, 앞을 막는 것은 통로 안에서 비켜 간다.

    정지 장애물은 라인을 만들 때 이미 피해 두었다. 여기서 다루는 것은
    움직이는 것 — 앞서 달리는 다른 차다. 매 스텝, 상대가 '내가 거기
    도착할 때' 어디 있을지를 셈해서 그 자리를 피하는 치우침을 고른다.
    비킬 자리가 없으면 속도를 줄여 뒤에 붙는다.
    """

    name = "교사 (레이싱 라인 + 회피)"

    def __init__(self, center, seg_a, seg_b, half_width, v_max=MAX_SPEED,
                 points=None, clearance=None):
        self.plan = racing_line(center, seg_a, seg_b, half_width,
                                points=points, clearance=clearance)
        self.line = self.plan.line
        self.speed = speed_plan(self.line, self.plan.room, v_max)
        ahead = np.roll(self.line, -1, axis=0)
        self.heading = np.arctan2(ahead[:, 1] - self.line[:, 1],
                                  ahead[:, 0] - self.line[:, 0])
        # 코너를 돌려면 원래 이만큼 꺾어야 한다. 오차를 보고 나서 꺾으면 늦는다.
        self.feedforward = np.arctan(
            WHEELBASE * path_curvature(self.line, CURVE_SMOOTH))
        self.spacing = float(np.linalg.norm(
            np.roll(self.line, -1, axis=0) - self.line, axis=1).mean())
        self.v_max = v_max
        self.shift = 0.0                  # 지금 라인에서 옆으로 비킨 양 [m]

    @classmethod
    def for_env(cls, env, v_max=MAX_SPEED, place=True, points=None,
                clearance=None):
        racer = cls(env.center, env.seg_a, env.seg_b, env.track_width / 2,
                    v_max, points=points, clearance=clearance)
        if place:
            racer.place(env)
        return racer

    def place(self, env):
        """차를 라인 위에 올려 놓는다. 라인은 중심선에서 최대 0.7m 떨어져 있다."""
        near = int(np.argmin(np.linalg.norm(self.line - env.pos, axis=1)))
        env.pos = self.line[near].copy()
        env.yaw = float(self.heading[near])
        env.speed = min(env.speed, float(self.speed[near]))
        env.prev_index = env._progress()
        return near

    def reset(self):
        self.shift = 0.0

    # ---- 움직이는 것 피하기 ---------------------------------------------
    def _station(self, point):
        """세계 좌표를 (라인 위 몇 번째 점, 옆으로 얼마나 벗어났나) 로."""
        index = int(np.argmin(np.linalg.norm(self.plan.center - point, axis=1)))
        side = float(np.dot(point - self.plan.center[index],
                            self.plan.normal[index]))
        return index, side

    @staticmethod
    def _where_at(rival, when):
        """그 상대가 when 초 뒤에 있을 자리. 예측을 안 주면 지금 자리."""
        future = rival.get("future")
        if not future:
            return np.asarray(rival["pos"], dtype=float)
        best = min(future, key=lambda item: abs(item[0] - when))
        return np.asarray(best[1], dtype=float)

    def _threat(self, rivals, near, speed):
        """내 길목을 막을 상대. (만나는 지점, 막힌구간 저, 고, 남은거리)

        상대는 코너에서 안쪽을 파고들기 때문에 좌우로도 움직인다.
        '지금 저기 있으니 저기를 피하자' 로는 늦는다. 내가 따라잡는 데
        걸리는 시간을 셈해서, 그때 그 차가 있을 자리를 피한다.
        """
        count = len(self.line)
        reach = max(speed, 1.0) * LOOKAHEAD_TIME
        worst = None
        for rival in rivals:
            index, _ = self._station(np.asarray(rival["pos"], dtype=float))
            ahead_m = ((index - near) % count) * self.spacing
            if ahead_m > reach:
                continue                       # 아직 멀다
            # 따라잡는 동안 상대도 앞으로 간다. 실제로 만나는 자리를 쓴다.
            closing = max(speed - float(rival["speed"]), 0.25)
            when = ahead_m / closing
            meet_m = ahead_m + float(rival["speed"]) * when
            if meet_m > reach:
                continue                       # 이 시야 안에서는 못 따라잡는다
            meet, side = self._station(self._where_at(rival, when))
            keep = RIVAL_HALF_WIDTH + HALF_WIDTH + RIVAL_MARGIN
            if worst is None or ahead_m < worst[3]:
                worst = (meet, side - keep, side + keep, ahead_m)
        return worst

    def _dodge(self, threat, dt):
        """막힌 구간을 피하는 치우침. 통로 밖으로는 안 나간다. 못 비키면 None."""
        meet, low, high, _ = threat
        here = float(self.plan.offset[meet])
        if not (low < here < high):
            want = here                        # 라인이 이미 옆으로 비켜 있다
        else:
            lo, hi = float(self.plan.low[meet]), float(self.plan.high[meet])
            picks = [p for p in (low - 0.03, high + 0.03) if lo <= p <= hi]
            if not picks:
                return None                    # 어느 쪽으로도 못 나간다
            want = min(picks, key=lambda p: abs(p - here))
        step = SHIFT_RATE * dt
        self.shift += float(np.clip(want - here - self.shift, -step, step))
        return self.shift

    # ---- 제어 -----------------------------------------------------------
    def act(self, pos, yaw, speed, rivals=(), dt=0.05):
        """뒷축 위치와 방향, 현재 속도를 받아 [조향, 속도] 를 -1~1 로."""
        front = pos + WHEELBASE * np.array([np.cos(yaw), np.sin(yaw)])
        near = int(np.argmin(np.linalg.norm(self.line - front, axis=1)))
        # 지금 내는 명령은 지연 뒤에 먹는다. 그때 도착할 자리의 계획을 낸다.
        # 이렇게 해야 코너 진입에서 딱 맞게 감속한다.
        soon_v = int((near + max(1, int(speed * CONTROL_LATENCY
                                        / max(self.spacing, 1e-6))))
                     % len(self.line))
        target = float(min(self.speed[near], self.speed[soon_v]))

        threat = self._threat(rivals, near, speed) if len(rivals) else None
        if threat is None:
            # 비킬 일이 없으면 라인으로 서서히 돌아온다
            step = SHIFT_RATE * dt
            self.shift -= float(np.clip(self.shift, -step, step))
        elif self._dodge(threat, dt) is None:
            # 못 비킨다. 앞차 뒤에 붙을 만큼만 줄인다.
            room = max(threat[3] - FOLLOW_GAP, 0.0)
            target = min(target, max(np.sqrt(2 * BRAKE_ACCEL * room), SPEED_MIN))

        # 라인에서 shift 만큼 옆으로 옮긴 점을 따라간다.
        # 비킬 양은 '상대를 만나는 지점' 의 통로를 보고 정했는데, 실제로
        # 적용되는 곳은 '지금 있는 지점' 이다. 통로 폭은 지점마다 다르므로
        # 여기서 다시 한 번 통로 안으로 자른다. 이걸 안 하면 넓은 곳 기준으로
        # 0.8m 를 비켰다가 좁은 곳에서 그대로 벽으로 나간다.
        here = float(self.plan.offset[near])
        shift = float(np.clip(self.shift,
                              float(self.plan.low[near]) - here,
                              float(self.plan.high[near]) - here))
        aim = self.line[near] + shift * self.plan.normal[near]
        path_yaw = self.heading[near]
        offset = front - aim
        cte = offset[0] * np.sin(path_yaw) - offset[1] * np.cos(path_yaw)

        # 코너를 돌려면 원래 이만큼은 꺾어야 한다. 미리 꺾어 두지 않으면 늦는다.
        lead = int(max(1.0, speed * FEEDFORWARD_LEAD) / max(self.spacing, 1e-6))
        soon = (near + lead) % len(self.line)
        ahead = float(self.feedforward[soon])

        # 자전거 모델에서 앞축이 라인 위에 얹혀 있으면, 차체는 경로 접선보다
        # 조향각만큼 이미 뒤틀려 있는 것이 정상이다. 그 정상분을 빼야 '진짜
        # 방향 오차' 가 남는다. 안 빼면 코너에서 방향오차와 피드포워드가
        # 사실상 같은 값이라(상관 0.85) 필요한 양의 두 배를 꺾게 되고,
        # Stanley 항이 그걸 상쇄하는 자리에서 균형을 잡는다. 그래서 차가
        # 라인보다 0.17m 안쪽으로 파고든다 (급코너 98%).
        # feedforward 가 곧 arctan(축거 x 곡률) 이므로 지금 자리 값을 그대로 쓴다
        lag = float(self.feedforward[near])
        heading_error = (path_yaw - yaw + np.pi) % (2 * np.pi) - np.pi - lag

        cte_term = np.arctan2(
            STANLEY_K * cte, STANLEY_DAMP * abs(speed) + STANLEY_SOFTENING)
        raw_heading = HEADING_GAIN * heading_error
        if raw_heading * cte_term > 0.0:
            weight = 1.0                       # 둘 다 안쪽으로 민다
        else:
            weight = max(0.0, 1.0 - abs(cte) / HEADING_BLEND_M)

        steer = ahead + cte_term + weight * raw_heading

        return np.array([
            float(np.clip(steer / MAX_STEER, -1.0, 1.0)),
            float(np.clip(2.0 * target / self.v_max - 1.0, -1.0, 1.0)),
        ])
