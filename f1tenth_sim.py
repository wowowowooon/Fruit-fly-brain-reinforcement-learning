"""ROBORACER(F1TENTH) 규격 경량 주행 시뮬.

공식 f1tenth_gym이 Python 3.8과 구버전 gym을 요구해서, 같은 규격으로 직접 만들었다.
센서와 차량 값은 대회 규정 사양에 맞춰 둔다. 실차로 옮길 때 다시 학습하지 않으려면
여기 숫자가 실물과 같아야 한다.
  센서  Hokuyo UST-10LX. 270도, 10m, 0.25도 분해능, 40Hz
  차량  자전거 모델, 축거 0.33m, 조향 +-0.42 rad
  제어  조향각과 속도를 연속값으로 받음
  트랙  벽으로 둘러싸인 폐곡선. 시드마다 모양이 다름
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --- 차량 (Traxxas 1:10) ---
# 팀 저장소 Roboracer-AJOU/jetson_code 의 vehicle_geometry.py 실측값(2026-08-16).
# 추정값이 아니라 실제로 잰 치수다. 실차를 다시 재면 여기도 같이 고쳐야 한다.
WHEELBASE = 0.33        # 뒷축(base_link)에서 앞축까지 [m]
CAR_WIDTH = 0.30        # 전폭 [m] (반폭 0.15)
REAR_OVERHANG = 0.10    # base_link 뒤로 튀어나온 길이 [m]
FRONT_OVERHANG = 0.50   # base_link 앞으로 튀어나온 길이 [m]
CAR_LENGTH = FRONT_OVERHANG + REAR_OVERHANG    # 0.60 m
MAX_STEER = np.deg2rad(40.0)   # 최대 조향각. 좌우 40도씩 (실차 실측)
MAX_SPEED = 7.0         # 목표 최고 속도 [m/s]. 보수적으로 잡았다
MAX_ACCEL = 4.0         # 가속 한계 [m/s^2]
MAX_BRAKE = 6.0         # 감속 한계 [m/s^2]. 브레이크가 가속보다 세다

# --- 타이어가 버티는 한계 ---
# 조향각만으로는 얼마나 도는지가 정해지지 않는다. 바퀴를 아무리 꺾어도
# 타이어가 못 버티면 차는 그냥 바깥으로 밀려난다(언더스티어).
#
# 이걸 안 넣으면 시뮬에서만 가능한 주행을 배운다. 40도까지 꺾을 수 있으면
# 최소 회전반경이 0.39m 이고, 7m/s 로 돌면 횡가속이 125 m/s^2 (12.7G) 다.
# 실차는 그 자리에서 미끄러진다. 배운 대로 하면 실차에서 코너마다 나간다.
#
# f1tenth_gym 기본 노면 마찰 mu = 1.0489 (실차 동정으로 얻은 값).
TIRE_FRICTION = 1.0489
GRAVITY = 9.81
LAT_ACCEL_MAX = TIRE_FRICTION * GRAVITY        # 10.3 m/s^2
# 노면은 매번 다르다. 대회장 바닥이 미끄러울 수도 있다.
FRICTION_RANGE = (0.75, 1.15)

# 서보도 즉시 안 꺾인다. f1tenth_gym 의 조향 속도 한계.
STEER_RATE_MAX = 3.2    # [rad/s]. 40도를 꺾는 데 0.22초

# 자이로가 읽을 수 있는 최대 각속도. 타이어 한계에 걸리므로
# 조향각이 아니라 마찰로 정해진다.  yaw_rate = a_lat / v
YAW_RATE_MAX = LAT_ACCEL_MAX / 2.0

# 차체가 뒷축 기준으로 차지하는 앞뒤 범위. 충돌 판정에 쓴다.
BODY_BACK = -REAR_OVERHANG
BODY_FRONT = FRONT_OVERHANG
BODY_RADIUS = CAR_WIDTH / 2

# --- 라이다 (Hokuyo UST-10LX, 대회 규정 필수 센서) ---
LIDAR_FOV = np.deg2rad(270)
LIDAR_RANGE = 10.0      # 최대 측정 거리 [m]. 이 값이 바뀌면 재학습해야 한다
LIDAR_BEAMS = 1081      # 270도 / 0.25도 + 1
LIDAR_HZ = 40.0         # 25ms 주기
CONTROL_DT = 1.0 / LIDAR_HZ
LIDAR_AHEAD = 0.31      # base_link 에서 라이다까지 [m]. 팀 실측값
# 라이다에서 본 차체. 앞으로는 조금, 뒤로는 많이 튀어나와 있다.
#   앞 범퍼 +0.19m / 뒤 범퍼 -0.41m / 좌우 각 0.15m
# 코너에서 뒤가 바깥으로 쓸리는데 라이다는 그쪽을 잘 못 본다.

# --- 제어 지연 ---
# 라이다 측정 -> 계산 -> VESC -> 모터까지 실제로 시간이 걸린다.
# 넣지 않고 학습하면 실차에서 핸들이 떨린다(oscillation).
CONTROL_LATENCY = 0.05  # [s]


# --- 트랙 다양성 ---
# 실차는 처음 보는 코스에서 바로 달려야 한다. 한 가지 모양만 학습하면
# 그 모양만 외운다. 크기, 폭, 비율, 코너 날카로움을 모두 흔든다.
#
# 기준으로 삼은 실제 코스 (2026 대회 ifac_roboracer 맵):
#   전체 22.2 x 8.1 m, 통로 폭 약 2.0 m (가장 넓은 곳 3.15 m)
# 학습 범위는 이보다 좁은 쪽과 넓은 쪽을 모두 덮어야 한다.
RADIUS_RANGE = (6.0, 20.0)     # 트랙 크기 [m]
WIDTH_RANGE = (1.7, 4.2)       # 주행로 폭 [m]. 실제 대회 코스가 1.83m 라
                               # 반드시 그보다 좁은 쪽까지 덮어야 한다
# 폭을 균등하게 뽑으면 실제 코스만큼 좁은 트랙이 8% 밖에 안 나온다.
# 제곱해서 좁은 쪽으로 몰아준다. 2.0m 이하가 40% 정도 나온다.
WIDTH_SKEW = 2.0               # 클수록 좁은 트랙이 많아진다

# 실제 대회 코스에는 180도로 되돌아오는 유턴이 있다. 일반 코너와 달리
# 라이다가 출구를 못 보는 상태에서 최대 조향으로 돌아야 한다.
# 이런 구간이 학습 트랙에 없으면 그 자리에서만 계속 박는다.
HAIRPIN_SHARE = 0.35           # 이 비율만큼은 유턴이 있는 트랙으로 만든다
HAIRPIN_SPIKES = (2, 5)        # 트랙 하나에 넣을 유턴 개수
HAIRPIN_GAP = (1.15, 1.9)      # 유턴 두 다리 사이 간격 / 주행로 폭
                               # 1.15면 벽 두께가 폭의 15%. 아주 빡빡한 유턴
HAIRPIN_DEPTH = (0.35, 0.75)   # 유턴이 안쪽으로 파고드는 깊이 / 트랙 반경
OUTER_WAYPOINTS = 4            # 유턴 사이 바깥 둘레를 몇 점으로 이을지.
                               # 적으면 그 자리가 뾰족해져 주행 불가가 된다
ASPECT_RANGE = (1.0, 3.0)      # 가로:세로 비율. 실제 코스는 2.7:1 로 길쭉하다
HARMONICS = (2, 3, 4, 5, 6)    # 높을수록 코너가 날카로워진다
CURVE_MARGIN = 1.30            # 곡률 반경이 폭 절반의 이 배는 되어야 함 (주행 한계)

# 급코너를 '허용'만 하면 거의 안 나온다. 트랙마다 목표 급함을 정해 놓고
# 그만큼 급한 코너가 실제로 있는 트랙만 받아들인다.
#   실제 대회 코스는 최대 조향(포화)이 전체 주행의 5.9% 를 차지한다.
#   허용만 했을 때 학습 트랙은 0.0% 였다. 그래서 모델이 덜 꺾었다.
TIGHTNESS_RANGE = (CURVE_MARGIN, 3.2)   # 최소 곡률반경 / (폭 절반)


# 실차는 매끄러운 수학 곡선이 아니라 스캔으로 만든 지저분한 벽을 본다.
# 대회 맵은 2.5cm 격자에 저장되어 있어서 벽이 계단 모양이고, 스캔 자체의
# 불규칙함도 섞여 있다. 매끈한 벽만 보고 배우면 실제 라이다 값에 놀란다.
WALL_GRID = 0.025       # 맵 해상도. 벽이 이 간격으로 계단진다 [m]
WALL_JITTER = 0.012     # 스캔의 불규칙함 [m]


def _offset_walls(center, width, rng=None):
    """중심선 좌우로 폭의 절반만큼 떨어진 벽."""
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    walls = [center - normal * width / 2, center + normal * width / 2]

    if rng is not None:
        rough = []
        for wall in walls:
            # 법선 방향으로만 흔든다. 트랙을 따라가는 방향은 건드리지 않는다.
            offset = rng.normal(0.0, WALL_JITTER, (len(wall), 1))
            noisy = wall + normal * offset
            # 격자에 스냅해서 계단 모양을 만든다
            rough.append(np.round(noisy / WALL_GRID) * WALL_GRID)
        walls = rough
    return walls[0], walls[1]


def _min_curve_radius(center):
    """중심선에서 가장 급한 코너의 곡률 반경. 작을수록 급커브."""
    d1 = (np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)) / 2
    d2 = np.roll(center, -1, axis=0) - 2 * center + np.roll(center, 1, axis=0)
    cross = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    speed = np.linalg.norm(d1, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.where(cross > 1e-12, speed ** 3 / cross, np.inf)
    return float(np.min(radius))


def _self_overlaps(center, width):
    """트랙이 자기 자신과 겹치는가. 떨어진 두 지점이 폭보다 가까우면 겹친다."""
    gap = np.linalg.norm(center[:, None, :] - center[None, :, :], axis=2)
    steps = len(center)
    offset = np.abs(np.arange(steps)[:, None] - np.arange(steps)[None, :])
    far_apart = np.minimum(offset, steps - offset) > steps // 8
    # 유턴에서는 두 다리가 가까워지는 게 정상이다. 벽이 남을 만큼만 떨어지면 된다.
    return bool(gap[far_apart].min() < width + 0.22)


def _resample(curve, points):
    """폐곡선을 등간격 점 points개로 다시 찍는다."""
    closed = np.vstack([curve, curve[:1]])
    steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    want = np.linspace(0.0, arc[-1], points, endpoint=False)
    return np.stack([np.interp(want, arc, closed[:, i]) for i in (0, 1)], axis=1)


def _chaikin(corners, rounds):
    """꼭짓점을 깎아 코너를 둥글린다. 직선 구간은 직선으로 남는다."""
    for _ in range(rounds):
        head = corners
        tail = np.roll(corners, -1, axis=0)
        cut = np.empty((2 * len(corners), 2))
        cut[0::2] = 0.75 * head + 0.25 * tail
        cut[1::2] = 0.25 * head + 0.75 * tail
        corners = cut
    return corners


def _smooth_loop(rng, radius, points):
    """푸리에 급수로 만든 매끈한 폐곡선. 직선 구간이 없다."""
    theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
    r = np.full(points, radius)
    for harmonic in HARMONICS:
        # 높은 차수일수록 작게 넣어야 트랙이 찢어지지 않는다
        amplitude = rng.uniform(0.0, 0.30) / harmonic
        r += amplitude * radius * np.cos(
            harmonic * theta + rng.uniform(0, 2 * np.pi)
        )
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def _rounded_polygon(vertices, corner_radius, points):
    """다각형 꼭짓점을 정해진 반경의 원호로 깎는다.

    Chaikin 과 달리 코너 반경을 직접 지정할 수 있다. 반경을 작게 주면
    진짜 헤어핀이 나온다. 변 중간은 직선으로 남는다.
    """
    path = []
    count = len(vertices)
    for i in range(count):
        here = vertices[i]
        to_prev = vertices[i - 1] - here
        to_next = vertices[(i + 1) % count] - here
        len_prev = np.linalg.norm(to_prev)
        len_next = np.linalg.norm(to_next)
        if len_prev < 1e-6 or len_next < 1e-6:
            continue
        to_prev = to_prev / len_prev
        to_next = to_next / len_next

        inner = np.arccos(np.clip(np.dot(to_prev, to_next), -1.0, 1.0))
        if inner < 0.08 or inner > np.pi - 0.08:
            path.append(here)             # 거의 일직선이거나 접힌 꼭짓점
            continue

        half = inner / 2
        # 변 길이를 넘지 않는 선에서 원하는 반경을 쓴다
        reach = min(corner_radius / np.tan(half), len_prev * 0.45, len_next * 0.45)
        actual = reach * np.tan(half)
        start = here + to_prev * reach
        end = here + to_next * reach
        bisect = to_prev + to_next
        bisect = bisect / np.linalg.norm(bisect)
        center = here + bisect * (actual / np.sin(half))

        a0 = np.arctan2(*(start - center)[::-1])
        a1 = np.arctan2(*(end - center)[::-1])
        sweep = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi   # 짧은 쪽으로
        steps = max(4, int(abs(sweep) * actual / 0.04))
        for t in np.linspace(0.0, 1.0, steps):
            a = a0 + sweep * t
            path.append(center + actual * np.array([np.cos(a), np.sin(a)]))

    return _resample(np.array(path), points)


def _hairpin_loop(rng, radius, points, corner_radius, width, aspect=1.0):
    """180도 되돌아오는 유턴이 들어간 코스.

    두 가지를 조심해야 유턴 반경이 살아남는다.

    1. 예리한 꼭짓점 하나를 깎으면 실제 반경이 무너진다. 각이 좁을수록
       호가 짧아져서 1.2m 를 요청해도 0.4m 가 나온다.
       그래서 유턴을 직각 두 개로 쪼갠다. 들어가서 90도 꺾고, 폭만큼
       건너간 뒤, 다시 90도 꺾어 나온다.

           A ────────────╮ B
                         │   <- 2*turn 만큼 건너간다
           D ────────────╯ C

    2. 길쭉하게 누르는 것을 유턴에까지 적용하면 유턴이 찌그러진다.
       3:1 로 누르면 곡률이 9배가 된다. 그래서 유턴이 놓일 '자리'만
       타원 위에서 정하고, 유턴 모양 자체는 누르지 않는다.
    """
    spikes = int(rng.integers(*HAIRPIN_SPIKES))
    turn = corner_radius * 1.3          # 깎이면서 줄어드는 만큼 미리 키워 둔다
    # 유턴 하나가 들어가려면 다리 길이가 반경의 5배는 필요하다.
    # 그렇다고 트랙이 한없이 커지면 한 바퀴가 400m 가 되어 버린다.
    radius = float(np.clip(max(radius, turn * 5.5), 6.0, 26.0))

    base = np.linspace(0, 2 * np.pi, spikes, endpoint=False)
    base = base + rng.uniform(-0.15, 0.15, spikes)
    squash = np.array([1.0, aspect])
    step = 2 * np.pi / spikes

    vertices = []
    for angle in base:
        # 유턴이 놓일 자리는 타원 위에서 고른다 (트랙 전체가 길쭉해진다)
        anchor = radius * np.array([np.cos(angle), np.sin(angle)]) / squash
        far = float(np.linalg.norm(anchor)) * rng.uniform(0.92, 1.08)
        out = anchor / max(np.linalg.norm(anchor), 1e-9)
        side = np.array([-out[1], out[0]])

        near = far * (1.0 - rng.uniform(*HAIRPIN_DEPTH))
        # 다리가 짧으면 원호를 못 그리고, 너무 깊으면 반대쪽 유턴과 겹친다
        near = float(np.clip(near, turn * 2.4, max(far - turn * 2.6, turn * 2.4)))

        # 반시계로 돌므로 -side 쪽에서 들어와 +side 쪽으로 나간다.
        # 순서를 뒤집으면 경로가 자기 자신을 가로질러 반경이 0 이 된다.
        for reach, offset in ((far, -turn), (near, -turn),
                              (near, turn), (far, turn)):
            vertices.append(out * reach + side * offset)

        # 다음 유턴까지 바깥으로 돌아간다. 이 경유점이 없으면 트랙 한가운데를
        # 가로질러서 유턴끼리 서로 교차한다.
        # 한 점만 찍으면 그 자리가 뾰족해진다. 타원을 따라 여러 점으로 잇는다.
        for k in range(1, OUTER_WAYPOINTS + 1):
            bend = angle + step * k / (OUTER_WAYPOINTS + 1)
            vertices.append(radius * 1.15
                            * np.array([np.cos(bend), np.sin(bend)]) / squash)

    return _rounded_polygon(np.array(vertices), corner_radius, points)


def _circuit_loop(rng, radius, points, corner_radius):
    """직선과 코너로 이루어진 서킷형. 실제 대회 코스가 이 모양이다.

    꼭짓점까지의 거리를 크게 흔들면 코스가 안으로 깊게 파고들어
    180도 가까이 돌아서는 헤어핀이 생긴다.
    """
    corners = int(rng.integers(5, 11))
    angles = np.linspace(0, 2 * np.pi, corners, endpoint=False) + \
        rng.uniform(-0.35, 0.35, corners)
    reach = radius * rng.uniform(0.40, 1.20, corners)
    polygon = np.stack([reach * np.cos(angles), reach * np.sin(angles)], axis=1)
    return _rounded_polygon(polygon, corner_radius, points)


def make_track(seed=0, points=400):
    """폐곡선 트랙. 중심선과 좌우 벽을 돌려준다.

    시드마다 크기, 폭, 코너 모양이 다르고 두 가지 형태가 섞여 나온다.
      매끈한 곡선형  코너가 이어지는 오벌
      서킷형        직선 구간과 코너가 번갈아 나옴 (대회 코스에 가까움)
    차가 지나갈 수 없는 모양이 나오면 버리고 다시 뽑는다.
    같은 시드는 항상 같은 트랙이 나온다.
    """
    radius = width = None
    target = np.random.default_rng([seed, 99]).uniform(*TIGHTNESS_RANGE)
    for attempt in range(400):
        rng = np.random.default_rng([seed, attempt])
        radius = rng.uniform(*RADIUS_RANGE)
        low, high = WIDTH_RANGE
        width = low + (high - low) * rng.random() ** WIDTH_SKEW

        # 코너 반경을 폭에 맞춰 정한다. 작게 주면 헤어핀이 된다.
        # 차의 최소 회전 반경은 0.84m 이므로 그보다는 커야 돈다.
        corner_radius = max(width / 2 * target, 0.95)
        # 길쭉하게 눌러서 실제 코스 같은 비율을 만든다
        aspect = rng.uniform(*ASPECT_RANGE)
        pick = rng.random()
        if pick < HAIRPIN_SHARE:
            # 유턴은 주행 한계에 최대한 가깝게 빡빡하게 만든다.
            # target 을 그대로 쓰면 반경 4.5m 짜리 헐렁한 유턴이 나오는데,
            # 실제 대회 코스의 유턴은 2.06m 다. 그러면 학습이 안 된다.
            tight = max(width / 2 * CURVE_MARGIN * 1.08, 0.95)
            # 유턴 트랙은 누른 뒤에 깎아야 반경이 유지된다. 안에서 처리한다.
            center = _hairpin_loop(rng, radius, points, tight, width, aspect)
        else:
            if pick < HAIRPIN_SHARE + 0.40:
                center = _circuit_loop(rng, radius, points, corner_radius)
            else:
                center = _smooth_loop(rng, radius, points)
            center = center / np.array([1.0, aspect])

        # 좌우 뒤집기와 진행 방향 뒤집기.
        # 이걸 안 하면 모든 트랙이 반시계라서 좌회전만 배운다.
        # 실제 코스에서 급우회전이 필요한 순간에 좌회전을 내는 것을 확인했다.
        if rng.random() < 0.5:
            center = center * np.array([-1.0, 1.0])
        if rng.random() < 0.5:
            center = center[::-1]

        center = _resample(center, points)
        tightness = _min_curve_radius(center) / (width / 2)
        if tightness < CURVE_MARGIN:
            continue                      # 벽이 접히는 급커브. 주행 불가
        if tightness > target * 1.25:
            continue                      # 너무 완만하다. 급코너가 없는 트랙
        if _self_overlaps(center, width):
            continue                      # 트랙끼리 겹침

        inner, outer = _offset_walls(center, width, rng)
        return center, inner, outer

    # 여기까지 오면 운이 나쁜 시드다. 흔들림 없는 원으로 돌려준다.
    theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
    center = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    inner, outer = _offset_walls(center, width, rng)
    return center, inner, outer


# 실제 코스는 가운데 섬이 있는 복잡한 공간이라, 코너를 돌면 시야가 확 바뀐다.
# 단순한 고리 모양 트랙에서는 그런 일이 없어서 모델이 급변에 대비하지 못한다.
# 장애물을 넣으면 시야 급변과 급조향이 동시에 생기고, 회피 자체도 배운다.
OBSTACLE_COUNT = (0, 5)          # 트랙 하나에 놓을 개수
OBSTACLE_SIZE = (0.12, 0.30)     # 반지름 [m]
PASSAGE_MIN = 1.45               # 장애물 옆에 남겨 둘 통로 [m]. 차폭은 0.30m
CHICANE_CHANCE = 0.45            # 좌우 번갈아 놓아 지그재그로 만들 확률


def _blob(middle, size, rng, points=14):
    angle = np.linspace(0, 2 * np.pi, points, endpoint=False)
    wobble = size * rng.uniform(0.75, 1.25, points)
    return middle + np.stack(
        [wobble * np.cos(angle), wobble * np.sin(angle)], axis=1)


def make_obstacles(seed, center, width, passage=PASSAGE_MIN):
    """트랙 위에 장애물을 흩뿌린다. 지나갈 틈은 반드시 남긴다.

    일부는 좌우로 번갈아 놓아 지그재그(시케인)를 만든다. 실제 코스에서
    급조향이 요구되는 상황이 이렇게 생긴다.

    `passage` 를 줄이면 좁은 트랙에도 장애물이 놓이고 옆 통로도 좁아진다.
    기본값 1.45m 로는 폭 1.8~2.15m 미만 트랙에 장애물이 하나도 안 놓여서
    (학습 트랙 400개 중 132개) 좁은 곳에서 피하는 법을 배울 기회가 없다.
    """
    rng = np.random.default_rng([seed, 4242])
    low, high = OBSTACLE_COUNT
    count = int(rng.integers(low, high)) if high > low else low
    if count == 0:
        return []

    steps = len(center)
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    # 서로 너무 붙지 않게 트랙을 균등히 나눠 자리를 잡는다
    spacing = max(8, steps // 40)
    slots = rng.permutation(steps // spacing)[:count] * spacing
    chicane = rng.random() < CHICANE_CHANCE
    blobs = []
    for order, slot in enumerate(sorted(slots)):
        size = rng.uniform(*OBSTACLE_SIZE)
        room = width / 2 - size - passage / 2
        if room <= 0.05:
            continue                      # 이 트랙은 좁아서 못 놓는다
        # 시케인이면 좌우를 번갈아, 아니면 아무 쪽이나
        sign = (1.0 if order % 2 else -1.0) if chicane \
            else float(rng.choice([-1.0, 1.0]))
        side = sign * rng.uniform(0.45, 1.0) * room
        blobs.append(_blob(center[slot] + normal[slot] * side, size, rng))
    return blobs


def as_segments(loop):
    """폐곡선 점열을 선분 시작/끝 배열로."""
    return loop, np.roll(loop, -1, axis=0)


def _point_segment_distance(points, seg_a, seg_b):
    """여러 점에서 여러 선분까지의 거리. points (N,2), seg (M,2) -> (N,M)"""
    edge = seg_b - seg_a                                        # (M,2)
    length2 = np.maximum((edge ** 2).sum(axis=1), 1e-12)        # (M,)
    offset = points[:, None, :] - seg_a[None, :, :]             # (N,M,2)
    t = np.clip(
        (offset[..., 0] * edge[None, :, 0]
         + offset[..., 1] * edge[None, :, 1]) / length2, 0.0, 1.0
    )
    closest = seg_a[None, :, :] + t[..., None] * edge[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2)


def body_clearance(back, front, seg_a, seg_b, samples=7):
    """차체 중심선에서 벽까지의 최단 거리.

    차를 점이 아니라 길이 있는 물체로 본다. 중심선을 여러 점으로 잘라
    각 점에서 벽까지 거리를 재고, 그중 가장 가까운 값을 쓴다.
    이 값이 차폭 절반보다 작으면 어딘가 닿은 것이다.
    """
    if len(seg_a) == 0:
        return np.inf
    ratios = np.linspace(0.0, 1.0, samples)[:, None]
    points = back[None, :] + ratios * (front - back)[None, :]
    return float(_point_segment_distance(points, seg_a, seg_b).min())


def raycast(origin, angles, seg_a, seg_b, max_range):
    """한 지점에서 여러 방향으로 쏜 광선이 벽에 닿는 거리."""
    if len(seg_a) == 0:          # 사거리 안에 벽이 하나도 없다
        return np.full(len(angles), max_range)
    direction = np.stack([np.cos(angles), np.sin(angles)], axis=1)   # (N,2)
    edge = seg_b - seg_a                                             # (M,2)

    denom = (direction[:, None, 0] * edge[None, :, 1]
             - direction[:, None, 1] * edge[None, :, 0])             # (N,M)
    offset = seg_a[None, :, :] - origin[None, None, :]               # (1,M,2)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (offset[..., 0] * edge[None, :, 1]
             - offset[..., 1] * edge[None, :, 0]) / denom
        s = (offset[..., 0] * direction[:, None, 1]
             - offset[..., 1] * direction[:, None, 0]) / denom

    hit = (t > 0) & (s >= 0) & (s <= 1) & (np.abs(denom) > 1e-12)
    return np.minimum(np.where(hit, t, np.inf).min(axis=1), max_range)


class F1TenthEnv(gym.Env):
    """라이다로 벽을 보고 트랙을 도는 환경."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, n_beams=LIDAR_BEAMS, dt=CONTROL_DT, max_steps=4000,
                 track_seed=0, lidar_range=LIDAR_RANGE,
                 latency=CONTROL_LATENCY, laps=1, obstacles=True,
                 passage=PASSAGE_MIN):
        super().__init__()
        # 장애물을 끄면 같은 트랙을 장애물만 빼고 다시 달릴 수 있다.
        # 회피 실력과 순수 주행 실력을 갈라 보려면 이 짝 비교가 필요하다.
        self.obstacles = obstacles
        # 장애물 옆에 남길 통로. 좁히면 좁은 트랙에도 장애물이 생긴다.
        self.passage = passage
        self.n_beams = n_beams
        self.dt = dt
        self.max_steps = max_steps
        self.track_seed = track_seed
        self.lidar_range = lidar_range
        # 몇 바퀴를 돌고 끝낼지. 학습 데이터를 모을 때는 여러 바퀴가 이득이다.
        self.laps = laps
        # 명령이 실제로 먹히기까지 몇 스텝 걸리는지
        self.latency_steps = int(round(latency / dt))
        self.beam_angles = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, n_beams)

        # 관측: 라이다 거리(0~1) + 현재 속도(0~1)
        self.observation_space = spaces.Box(-1.0, 1.0, (n_beams + 3,), np.float32)
        # 행동: 조향(-1~1), 속도(-1~1)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

        self._load_track(track_seed)

    def _load_track(self, seed):
        self.center, inner, outer = make_track(seed)
        self.blobs = make_obstacles(
            seed, self.center,
            np.linalg.norm(outer - inner, axis=1).mean(),
            passage=getattr(self, "passage", PASSAGE_MIN),
        ) if getattr(self, "obstacles", True) else []
        starts, ends = [], []
        for loop in [inner, outer] + self.blobs:
            a, b = as_segments(loop)
            starts.append(a)
            ends.append(b)
        self.seg_a = np.concatenate(starts)
        self.seg_b = np.concatenate(ends)
        self.inner, self.outer = inner, outer

        # 보상을 칸 수가 아니라 미터로 주기 위해 중심선 길이를 재 둔다.
        # 트랙 크기가 시드마다 달라서 칸 하나의 거리도 달라진다.
        steps = np.linalg.norm(np.roll(self.center, -1, axis=0) - self.center, axis=1)
        self.step_length = float(steps.mean())
        self.lap_length = float(steps.sum())
        self.track_width = float(np.linalg.norm(outer - inner, axis=1).mean())
        # 사거리 밖 벽은 광선 계산에서 미리 뺀다. 선분 중점까지 거리가
        # (사거리 + 선분 절반)보다 멀면 절대 닿지 않는다.
        self.seg_mid = (self.seg_a + self.seg_b) / 2
        self.seg_reach = (
            np.linalg.norm(self.seg_b - self.seg_a, axis=1) / 2 + self.lidar_range
        )

    def _heading(self):
        return np.array([np.cos(self.yaw), np.sin(self.yaw)])

    @property
    def lidar_pos(self):
        """라이다가 실제로 있는 위치. 차량 앞쪽에 달려 있다."""
        return self.pos + LIDAR_AHEAD * self._heading()

    def _near_walls(self, origin, reach):
        near = np.linalg.norm(self.seg_mid - origin, axis=1) <= self.seg_reach + reach
        return self.seg_a[near], self.seg_b[near]

    def _scan(self):
        origin = self.lidar_pos
        angles = self.beam_angles + self.yaw
        seg_a, seg_b = self._near_walls(origin, 0.0)
        return raycast(origin, angles, seg_a, seg_b, self.lidar_range)

    def _clearance(self):
        """차체 어디든 벽에 가장 가까운 거리. 범퍼와 바퀴까지 포함한다."""
        heading = self._heading()
        back = self.pos + BODY_BACK * heading
        front = self.pos + BODY_FRONT * heading
        seg_a, seg_b = self._near_walls(self.pos, CAR_LENGTH)
        return body_clearance(back, front, seg_a, seg_b)

    def _observe(self, scan):
        # 라이다 + IMU. 실차도 /scan 과 /imu 두 가지를 받는다.
        # 각속도는 자이로가, 가속도는 가속도계가 직접 준다.
        return np.concatenate([
            scan / self.lidar_range,
            [self.speed / MAX_SPEED,
             self.yaw_rate / YAW_RATE_MAX,
             self.accel / MAX_ACCEL],
        ]).astype(np.float32)

    def _progress(self):
        """중심선 위 어디쯤 왔는지 (0~1)."""
        distances = np.linalg.norm(self.center - self.pos, axis=1)
        return int(np.argmin(distances))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None and seed != self.track_seed:
            self.track_seed = seed
            self._load_track(seed)

        start = self.center[0]
        heading = self.center[1] - self.center[0]

        # 상태는 뒷축 기준으로 굴린다(자전거 모델). 라이다는 거기서 앞으로 떨어져 있다.
        self.pos = start.copy()
        self.yaw = float(np.arctan2(heading[1], heading[0]))
        self.speed = 2.0
        self.steer = 0.0
        self.yaw_rate = 0.0
        self.accel = 0.0
        # 노면은 매번 다르다. 미끄러운 바닥에서도 달릴 수 있어야 한다.
        low, high = FRICTION_RANGE
        self.lat_accel_max = LAT_ACCEL_MAX * float(
            self.np_random.uniform(low, high))
        self.steps = 0
        self.lap_progress = 0
        self.prev_index = 0
        self.crashed = False
        # 아직 안 먹은 명령들. 지연을 흉내 낸다.
        # maxlen 을 주면 꽉 찬 상태에서 append 할 때 왼쪽이 먼저 버려진다.
        # 그러면 대기열에 명령이 하나만 남아 지연이 절반이 된다
        # (50ms 로 설정해 두고 실제로는 25ms 만 지연되고 있었다).
        # popleft 로 꺼내므로 길이는 저절로 유지된다.
        self.queued = deque([np.array([0.0, -1.0])] * self.latency_steps)

        scan = self._scan()
        return self._observe(scan), {}

    def step(self, action):
        # 지금 낸 명령은 줄 뒤에 서고, 앞에 있던 명령이 지금 먹는다.
        action = np.asarray(action, dtype=np.float64)
        if self.latency_steps > 0:
            self.queued.append(action)
            action = self.queued.popleft()

        want_steer = float(np.clip(action[0], -1, 1)) * MAX_STEER
        target = (float(np.clip(action[1], -1, 1)) + 1) / 2 * MAX_SPEED

        # 서보가 따라오는 속도에 한계가 있다. 한 번에 확 꺾이지 않는다.
        slew = STEER_RATE_MAX * self.dt
        steer = self.steer + np.clip(want_steer - self.steer, -slew, slew)
        self.steer = float(steer)

        # 속도도 즉시 바뀌지 않는다. 브레이크가 가속보다 세다.
        change = float(np.clip(target - self.speed,
                               -MAX_BRAKE * self.dt, MAX_ACCEL * self.dt))
        self.speed += change
        self.accel = change / self.dt          # IMU 가속도계가 재는 값

        # 자전거 모델 (뒷축 기준). 바퀴가 시키는 회전율.
        turn = self.speed / WHEELBASE * np.tan(self.steer)

        # 타이어가 버티는 만큼만 실제로 돈다. 넘치면 바깥으로 밀린다.
        if self.speed > 0.1:
            grip = self.lat_accel_max / self.speed
            turn = float(np.clip(turn, -grip, grip))
        self.yaw_rate = turn                   # 자이로가 재는 값

        self.pos = self.pos + self.speed * self.dt * self._heading()
        self.yaw += self.yaw_rate * self.dt
        self.steps += 1

        scan = self._scan()
        # 라이다 한 점이 아니라 차체 전체로 판정한다. 비싸므로 한 번만 잰다.
        clearance = self._clearance()
        self.crashed = bool(clearance < BODY_RADIUS)

        # 중심선을 따라 얼마나 나아갔는지로 보상
        index = self._progress()
        step_count = len(self.center)
        delta = (index - self.prev_index + step_count // 2) % step_count - step_count // 2
        self.prev_index = index
        self.lap_progress += delta

        # 트랙을 따라 나아간 거리(m). 스텝 시간이 고정이므로
        # 이걸 최대화하면 평균 속도가 최대화되고 곧 랩타임이 줄어든다.
        advance = delta * self.step_length
        reward = advance
        if self.crashed:
            reward -= 20.0

        done = self.crashed or self.lap_progress >= step_count * self.laps
        truncated = self.steps >= self.max_steps

        info = {
            "progress": self.lap_progress / step_count,
            "laps_done": self.lap_progress / step_count,
            "speed": self.speed,
            "crashed": self.crashed,
            "min_range": float(scan.min()),
            "clearance": clearance,
            "advance": advance,
            "lap_time": self.steps * self.dt,
        }
        return self._observe(scan), float(reward), done, truncated, info

    def render(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6), dpi=90)
        for wall in (self.inner, self.outer):
            loop = np.vstack([wall, wall[:1]])
            ax.plot(loop[:, 0], loop[:, 1], color="0.3", linewidth=1.2)

        scan = self._scan()
        angles = self.beam_angles + self.yaw
        origin = self.lidar_pos
        ends = origin + scan[:, None] * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )
        for end in ends[::6]:
            ax.plot([origin[0], end[0]], [origin[1], end[1]],
                    color="tab:orange", linewidth=0.4, alpha=0.5)

        heading = self._heading()
        body = np.stack([self.pos + BODY_BACK * heading,
                         self.pos + BODY_FRONT * heading])
        ax.plot(body[:, 0], body[:, 1], color="tab:green", linewidth=6, alpha=0.7)
        ax.plot(*origin, marker="o", color="tab:red", markersize=5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"speed {self.speed:.1f} m/s   progress {self.lap_progress / len(self.center):.1%}")

        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return image
