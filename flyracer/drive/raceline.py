"""레이싱 라인을 뽑고 Stanley 로 따라간다.

팀 저장소 Roboracer-AJOU/jetson_code 의 실차 방식을 옮긴 것이다.
  라인   scripts/generate_raceline_from_centerline.py  최소곡률 최적화
  속도   scripts/speed_profile.py
  추종   path_following/stanley_waypoint_follow_node.py

FGM 과 다른 점이 핵심이다. FGM 은 그때그때 뚫린 쪽으로 가므로 트랙 한가운데를
달린다. 레이싱 라인은 코너를 미리 깎아 들어가므로 훨씬 빠르고, 급조향도 실제로 쓴다.

이 교사는 자기 위치와 트랙 전체 모양을 안다(맵 기반). 학생인 초파리 회로는
라이다만 본다. 아는 게 많은 교사를 모르는 학생이 흉내 내게 하는 방식이다.
"""

import numpy as np

from flyracer.sim.f1tenth_sim import (
    CAR_WIDTH, FRONT_OVERHANG, LAT_ACCEL_MAX, MAX_ACCEL, MAX_BRAKE,
    MAX_SPEED, MAX_STEER, WHEELBASE, _resample,
)

# --- Stanley (팀 실차 파라미터) ---
STANLEY_K = 1.8            # 경로 이탈 복귀 세기
STANLEY_SOFTENING = 0.12   # 저속에서 발산하지 않게 분모에 더하는 값
HEADING_GAIN = 1.0         # 방향 오차 비중
HEADING_BLEND_M = 0.45     # 이탈이 이보다 크면 방향항을 줄인다
FEEDFORWARD_LEAD = 0.12    # 몇 초 앞의 곡률을 미리 반영할지 [s]
CURVE_SMOOTH = 11          # 곡률을 이 개수만큼 평균 내어 잔떨림을 없앤다

# --- 속도 프로파일 (팀 speed_profile.py) ---
# 시뮬의 타이어 한계와 같은 값을 쓴다. 교사가 시뮬보다 과감하게 계획하면
# 자기가 세운 계획을 자기가 못 따라가서 코너마다 밀려난다.
# 노면이 나쁜 경우(마찰 0.75배)에도 성립하도록 그쪽에 맞춘다.
LAT_ACCEL = LAT_ACCEL_MAX * 0.75   # 코너에서 버티는 횡가속도 [m/s^2]
BRAKE_ACCEL = MAX_BRAKE            # 감속 한계 [m/s^2]
DRIVE_ACCEL = MAX_ACCEL            # 가속 한계 [m/s^2]
SPEED_MIN = 1.5            # 하한 [m/s]
SAFETY = 0.65              # 실차 첫 주행 때는 0.6~0.7 로 낮춘다
WIDTH_SPEED_GAIN = 3.5     # 옆 공간 1m 당 허용 속도 [m/s]. 낮출수록 안전

# --- 라인 ---
WALL_CLEARANCE = 0.20      # 스윕 폭에 더해 벽에서 띄울 거리 [m]
HALF_WIDTH = CAR_WIDTH / 2


def wall_clearance_for(half_width):
    """트랙 폭에 맞춘 여유. 좁은 코스에서 고정값을 쓰면 라인이 안 나온다.

    실제 대회 코스는 통로가 1.83m(반폭 0.92m)라 차체 스윕 0.19m 를 빼면
    좌우로 쓸 수 있는 폭이 얼마 없다. 여유를 폭에 비례해 줄인다.
    """
    return float(np.clip(0.30 * half_width, 0.08, WALL_CLEARANCE))


def outer_half_width(radius):
    """반경 R 의 원호를 돌 때 차체가 경로 바깥으로 벗어나는 거리.

    차 앞이 0.50m 나 되어서 코너에서 앞 외측 모서리가 경로보다 바깥을 지난다.
    팀 vehicle_geometry.outer_half_width 와 같은 식이다.
        R=0.9 -> 0.263   R=1.5 -> 0.224   R=3.0 -> 0.186   직선 -> 0.15
    """
    r = np.abs(radius)
    wide = np.hypot(FRONT_OVERHANG, r + HALF_WIDTH) - r
    return np.where(r > 1e4, HALF_WIDTH, wide)


def _normals(points):
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)


def path_curvature(points, smooth=0):
    """각 점에서의 곡률 [1/m]. 부호는 좌회전이 양수.

    벽 여유를 강제하면서 라인에 잔 꺾임이 생긴다. 그대로 쓰면 곡률이
    들쭉날쭉해서 속도 계획과 피드포워드가 요동친다. smooth 를 주면
    그만큼의 이웃과 평균 내어 실제 코너 모양만 남긴다.
    """
    d1 = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / 2
    d2 = np.roll(points, -1, axis=0) - 2 * points + np.roll(points, 1, axis=0)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    kappa = cross / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-9)
    if smooth > 1:
        window = np.ones(smooth) / smooth
        pad = np.concatenate([kappa[-smooth:], kappa, kappa[:smooth]])
        kappa = np.convolve(pad, window, mode="same")[smooth:-smooth]
    return kappa


def _second_difference(n):
    """폐곡선용 2계 차분 행렬. 굽은 정도에 해당한다."""
    matrix = np.zeros((n, n))
    index = np.arange(n)
    matrix[index, index] = -2.0
    matrix[index, (index - 1) % n] = 1.0
    matrix[index, (index + 1) % n] = 1.0
    return matrix


def _largest_eigenvalue(matrix, rounds=30):
    vector = np.ones(len(matrix)) / np.sqrt(len(matrix))
    value = 1.0
    for _ in range(rounds):
        vector = matrix @ vector
        value = np.linalg.norm(vector)
        if value < 1e-12:
            return 1.0
        vector /= value
    return value


def _solve_box(gram, bias, low, high, iterations=600):
    """0.5 aGa + b·a 를 low<=a<=high 안에서 최소화. 가속 투영경사법."""
    step = 1.0 / (_largest_eigenvalue(gram) + 1e-9)
    alpha = np.zeros(len(bias))
    look = alpha.copy()
    momentum = 1.0
    for _ in range(iterations):
        moved = np.clip(look - step * (gram @ look + bias), low, high)
        nxt = (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        look = moved + ((momentum - 1.0) / nxt) * (moved - alpha)
        alpha, momentum = moved, nxt
    return alpha


def _rays_to_walls(origins, directions, seg_a, seg_b, reach):
    """점마다 각자의 방향으로 쏜 광선이 벽에 닿는 거리. (N,)

    점 하나씩 raycast 를 부르면 파이썬 루프가 400번 돌아 느리다.
    전부 한 번에 (점 x 선분) 행렬로 계산한다.
    """
    edge = seg_b - seg_a                                        # (M,2)
    denom = (directions[:, None, 0] * edge[None, :, 1]
             - directions[:, None, 1] * edge[None, :, 0])       # (N,M)
    offset = seg_a[None, :, :] - origins[:, None, :]            # (N,M,2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (offset[..., 0] * edge[None, :, 1]
             - offset[..., 1] * edge[None, :, 0]) / denom
        s = (offset[..., 0] * directions[:, None, 1]
             - offset[..., 1] * directions[:, None, 0]) / denom
    hit = (t > 1e-9) & (s >= 0) & (s <= 1) & (np.abs(denom) > 1e-12)
    return np.minimum(np.where(hit, t, np.inf).min(axis=1), reach)


def _lateral_room(line, normal, seg_a, seg_b, reach=6.0):
    """각 점에서 좌우 법선 방향으로 벽까지 몇 m 인지."""
    plus = _rays_to_walls(line, normal, seg_a, seg_b, reach)
    minus = _rays_to_walls(line, -normal, seg_a, seg_b, reach)
    return plus, minus


def _closest_on_walls(points, seg_a, seg_b):
    """각 점에서 가장 가까운 벽 위의 점과 그 거리."""
    edge = seg_b - seg_a
    length2 = np.maximum((edge ** 2).sum(axis=1), 1e-12)
    offset = points[:, None, :] - seg_a[None, :, :]
    t = np.clip((offset[..., 0] * edge[None, :, 0]
                 + offset[..., 1] * edge[None, :, 1]) / length2, 0.0, 1.0)
    foot = seg_a[None, :, :] + t[..., None] * edge[None, :, :]
    gap = np.linalg.norm(points[:, None, :] - foot, axis=2)
    nearest = np.argmin(gap, axis=1)
    rows = np.arange(len(points))
    return foot[rows, nearest], gap[rows, nearest]


def _enforce_clearance(line, seg_a, seg_b, keep, passes=40):
    """벽에 너무 붙은 점을 밀어낸다.

    최소곡률만 풀면 라인이 벽을 스칠 수 있다. 법선 방향으로만 여유를 쟀기
    때문에, 비스듬히 다가오는 벽이나 앞뒤로 놓인 장애물은 놓친다.
    실제 최단거리로 다시 재서 부족한 만큼 밀어낸다.

    평활을 매번 걸면 밀어낸 걸 도로 당겨서 수렴하지 않는다.
    앞부분에서만 다듬고 뒤쪽 패스는 밀어내기만 한다.
    """
    line = line.copy()
    for step in range(passes):
        foot, gap = _closest_on_walls(line, seg_a, seg_b)
        short = gap < keep
        if not short.any():
            break
        away = line[short] - foot[short]
        norm = np.linalg.norm(away, axis=1, keepdims=True)
        away = np.where(norm > 1e-9, away / np.maximum(norm, 1e-9), 0.0)
        line[short] = foot[short] + away * keep[short, None] * 1.05
        # 다듬기는 밀어낸 걸 도로 당긴다. 약하게, 그리고 앞쪽 패스에서만.
        if step < passes * 0.6:
            line = 0.84 * line + 0.08 * (np.roll(line, 1, axis=0)
                                         + np.roll(line, -1, axis=0))
    return line


def _sharpest(path):
    """이 경로에서 가장 급한 코너의 회전반경. 작을수록 급하다."""
    d1 = (np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)) / 2
    d2 = np.roll(path, -1, axis=0) - 2 * path + np.roll(path, 1, axis=0)
    cross = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    step = np.linalg.norm(d1, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.where(cross > 1e-12, step ** 3 / cross, np.inf)
    return float(np.min(radius))


def _worst_shortfall(path, seg_a, seg_b, clearance):
    """벽 여유가 가장 모자란 정도. 양수면 모든 점이 안전하다."""
    need = outer_half_width(1.0 / np.maximum(
        np.abs(path_curvature(path, CURVE_SMOOTH)), 1e-6)) + clearance
    _, gap = _closest_on_walls(path, seg_a, seg_b)
    return float(np.min(gap - need))


def _no_worse_than_center(line, center, seg_a, seg_b, clearance):
    """최적화 결과가 중심선보다 급해졌으면 중심선 쪽으로 되돌린다.

    벽 여유를 점마다 따로 밀어내다 보면 라인에 뾰족한 꺾임이 생긴다.
    그러면 '최소곡률' 최적화가 중심선보다 더 급한 라인을 내놓는다.
    중심선 1.2m 짜리 트랙에서 라인이 0.42m 까지 꺾이는 것을 확인했다.
    그 라인은 차체가 코너 바깥을 크게 쓸어서 그대로 벽에 닿는다.

    최적화가 실패한 경우에는 적어도 중심선만큼은 보장되어야 한다.

    단, '중심선은 안전하다' 는 전제가 항상 참은 아니다. 장애물은 트랙이
    좁으면 중심선에서 6~15cm 밖에 못 비켜서므로 중심선을 그대로 덮는다.
    그러면 되돌린 라인이 장애물을 뚫고, 교사가 장애물로 돌진하는 시범을
    보이게 된다 (장애물 있는 트랙 20개 중 13개에서 이렇게 되고 있었다).
    그래서 마지막에 반드시 여유를 다시 확보한다.
    """
    target = _sharpest(center)
    if _sharpest(line) >= target and _worst_shortfall(
            line, seg_a, seg_b, clearance) >= 0:
        return line

    best, best_gap = line, _worst_shortfall(line, seg_a, seg_b, clearance)
    for weight in (0.25, 0.5, 0.75, 1.0):
        blend = (1.0 - weight) * line + weight * center
        gap = _worst_shortfall(blend, seg_a, seg_b, clearance)
        if _sharpest(blend) >= target * 0.95 and gap >= 0:
            return blend
        # 안 되면 적어도 가장 덜 위험한 것을 남긴다
        if gap > best_gap:
            best, best_gap = blend, gap
    if best_gap >= 0:
        return best
    # 전부 부딪힌다. 뚫고 지나가는 시범을 보이느니 밀어내고 본다.
    return _enforce_clearance(best, seg_a, seg_b,
                              outer_half_width(1.0 / np.maximum(
                                  np.abs(path_curvature(best, CURVE_SMOOTH)),
                                  1e-6)) + clearance,
                              passes=60)


def minimum_curvature_line(center, seg_a, seg_b, rounds=6, points=None,
                           clearance=WALL_CLEARANCE):
    """곡률 제곱합이 최소가 되는 주행선.

    경로를 P = 기준선 + a*법선 으로 두면 굽은 정도가 a 의 1차식이라
    이차계획 문제가 된다. 한 번 풀고 끝내면 거의 안 움직인다. 푼 결과를
    새 기준선으로 삼아 법선과 벽 여유를 다시 재고 또 푸는 식으로 반복한다.
    """
    count = points or len(center)
    line = _resample(np.asarray(center, dtype=float), count)
    d2 = _second_difference(count)

    # 먼저 기준선부터 통과 가능하게 만든다.
    # 중심선 위에 장애물이 놓이면 좌우 여유가 0 으로 잡혀 최적화가 그 점을
    # 움직이지 못한다. 그러면 라인이 장애물을 뚫고 지나간다.
    line = _resample(_enforce_clearance(
        line, seg_a, seg_b,
        outer_half_width(1.0 / np.maximum(
            np.abs(path_curvature(line, CURVE_SMOOTH)), 1e-6)) + clearance,
        passes=90,
    ), count)

    for _ in range(rounds):
        normal = _normals(line)
        plus, minus = _lateral_room(line, normal, seg_a, seg_b)
        # 코너가 급할수록 차체가 더 바깥을 쓸므로 여유를 더 둔다
        swept = outer_half_width(1.0 / np.maximum(np.abs(path_curvature(line)), 1e-6))
        keep = swept + clearance
        high = np.maximum(plus - keep, 0.0)
        low = -np.maximum(minus - keep, 0.0)

        base = d2 @ line
        mx = d2 * normal[:, 0]
        my = d2 * normal[:, 1]
        gram = mx.T @ mx + my.T @ my
        bias = mx.T @ base[:, 0] + my.T @ base[:, 1]

        alpha = _solve_box(gram, bias, low, high)
        line = _resample(line + alpha[:, None] * normal, count)

        # 푼 결과가 벽을 스칠 수 있다. 실제 최단거리로 다시 재서 밀어낸다.
        need = outer_half_width(
            1.0 / np.maximum(np.abs(path_curvature(line)), 1e-6)
        ) + clearance
        line = _resample(_enforce_clearance(line, seg_a, seg_b, need), count)

    # 여유를 밀어내다 보면 라인에 뾰족한 꺾임이 남는다. 꺾임이 있으면
    # 곡률이 튀어서 속도 계획과 피드포워드가 망가진다. 펴고 다시 확인하기를
    # 번갈아 하면서, 펼수록 여유는 조금씩만 양보한다.
    for round_no in range(8):
        for _ in range(2):
            line = 0.5 * line + 0.25 * (np.roll(line, 1, axis=0)
                                        + np.roll(line, -1, axis=0))
        need = outer_half_width(
            1.0 / np.maximum(np.abs(path_curvature(line, CURVE_SMOOTH)), 1e-6)
        ) + clearance * (1.0 - 0.35 * round_no / 7)
        line = _enforce_clearance(line, seg_a, seg_b, need, passes=25)

    # 최적화가 중심선보다 나쁜 라인을 내놓는 경우가 있다. 그럴 때는
    # 중심선 쪽으로 되돌려서 최소한 중심선만큼은 되게 한다.
    guide = _resample(np.asarray(center, dtype=float), count)
    line = _no_worse_than_center(_resample(line, count), guide,
                                 seg_a, seg_b, clearance)
    return _resample(line, count)


def width_speed_cap(room, v_max=MAX_SPEED):
    """옆으로 남은 공간에 맞춘 속도 상한.

    좁은 통로에서는 조금만 벗어나도 벽이다. 추종 오차가 속도에 비례해
    커지므로, 회복할 여유가 없으면 애초에 천천히 가야 한다.
    실제 대회 코스처럼 통로가 1.8m 인 곳에서 7m/s 로 달리면 반드시 박는다.
    """
    usable = np.maximum(room - HALF_WIDTH, 0.05)
    return np.clip(SPEED_MIN + WIDTH_SPEED_GAIN * usable, SPEED_MIN, v_max)


def speed_profile(points, v_max=MAX_SPEED, room=None):
    """코너에서 미리 줄이고 탈출하며 올리는 속도 계획.

    1. 각 점에서 횡가속도 한계로 낼 수 있는 최고 속도
    2. 옆으로 남은 공간이 좁으면 더 줄인다
    3. 앞뒤로 훑으며 가감속 한계를 지키게 다듬는다
    """
    kappa = np.abs(path_curvature(points, CURVE_SMOOTH))
    limit = np.sqrt(LAT_ACCEL * SAFETY / np.maximum(kappa, 1e-6))
    if room is not None:
        limit = np.minimum(limit, width_speed_cap(room, v_max))
    speed = np.clip(limit, SPEED_MIN, v_max)

    gap = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    count = len(points)
    for _ in range(3):                        # 폐곡선이라 여러 바퀴 돌려야 수렴
        for i in range(count):                # 감속: 앞의 코너에 맞춰 미리 줄인다
            j = (count - 1 - i) % count
            nxt = (j + 1) % count
            speed[j] = min(speed[j], np.sqrt(
                speed[nxt] ** 2 + 2 * BRAKE_ACCEL * gap[j]))
        for i in range(count):                # 가속: 낼 수 있는 만큼만 올린다
            nxt = (i + 1) % count
            speed[nxt] = min(speed[nxt], np.sqrt(
                speed[i] ** 2 + 2 * DRIVE_ACCEL * gap[i]))
    return np.clip(speed, SPEED_MIN, v_max)


class StanleyRacer:
    """레이싱 라인을 Stanley 로 따라간다. 자기 위치를 아는 교사."""

    name = "교사 (레이싱 라인 + Stanley)"

    def __init__(self, center, seg_a, seg_b, v_max=MAX_SPEED, half_width=None):
        if half_width is None:
            normal = _normals(_resample(np.asarray(center, float), len(center)))
            plus, minus = _lateral_room(
                _resample(np.asarray(center, float), len(center)),
                normal, seg_a, seg_b)
            half_width = float(np.median(np.minimum(plus, minus)))
        self.line = minimum_curvature_line(
            center, seg_a, seg_b, clearance=wall_clearance_for(half_width))
        normal = _normals(self.line)
        plus, minus = _lateral_room(self.line, normal, seg_a, seg_b)
        self.room = np.minimum(plus, minus)
        self.speed = speed_profile(self.line, v_max, self.room)
        ahead = np.roll(self.line, -1, axis=0)
        self.heading = np.arctan2(ahead[:, 1] - self.line[:, 1],
                                  ahead[:, 0] - self.line[:, 0])
        # 코너를 돌려면 원래 이만큼은 꺾어야 한다. 오차를 보고 나서 꺾으면 늦는다.
        self.feedforward = np.arctan(
            WHEELBASE * path_curvature(self.line, CURVE_SMOOTH))
        self.spacing = float(np.linalg.norm(
            np.roll(self.line, -1, axis=0) - self.line, axis=1).mean())

    @classmethod
    def for_env(cls, env, v_max=MAX_SPEED, place=True):
        racer = cls(env.center, env.seg_a, env.seg_b, v_max)
        if place:
            racer.place(env)
        return racer

    def place(self, env):
        """차를 레이싱 라인 위에 올려 놓는다.

        환경은 중심선 시작점에 차를 놓는데, 레이싱 라인은 거기서 최대 0.7m
        떨어져 있다. 그대로 출발하면 첫 스텝부터 최대 조향으로 복귀하려다
        벽에 박는다. 실차에서도 출발 위치는 라인 위다.
        """
        near = int(np.argmin(np.linalg.norm(self.line - env.pos, axis=1)))
        env.pos = self.line[near].copy()
        env.yaw = float(self.heading[near])
        env.speed = min(env.speed, float(self.speed[near]))
        env.prev_index = env._progress()
        return near

    def reset(self):
        pass

    def act(self, pos, yaw, speed):
        """뒷축 위치와 방향, 현재 속도를 받아 [조향, 속도] 를 -1~1 로."""
        # Stanley 는 앞축을 기준점으로 쓴다
        front = pos + WHEELBASE * np.array([np.cos(yaw), np.sin(yaw)])
        near = int(np.argmin(np.linalg.norm(self.line - front, axis=1)))

        path_yaw = self.heading[near]
        heading_error = (path_yaw - yaw + np.pi) % (2 * np.pi) - np.pi

        # 경로 오른쪽에 있으면 cte > 0 -> 좌회전으로 복귀
        offset = front - self.line[near]
        cte = offset[0] * np.sin(path_yaw) - offset[1] * np.cos(path_yaw)

        cte_term = np.arctan2(STANLEY_K * cte, abs(speed) + STANLEY_SOFTENING)
        raw_heading = HEADING_GAIN * heading_error
        if raw_heading * cte_term > 0.0:
            weight = 1.0                      # 둘 다 안쪽으로 민다. 억제할 이유 없음
        else:
            weight = max(0.0, 1.0 - abs(cte) / HEADING_BLEND_M)

        # 곡률 피드포워드. 지금 있는 자리가 아니라 곧 도착할 자리의 코너를 본다.
        # 제어 지연과 서보 응답 때문에 미리 꺾어 두지 않으면 코너 밖으로 밀린다.
        lead = int(max(1.0, speed * FEEDFORWARD_LEAD) / max(self.spacing, 1e-6))
        soon = (near + lead) % len(self.line)
        steer = self.feedforward[soon] + cte_term + weight * raw_heading

        return np.array([
            float(np.clip(steer / MAX_STEER, -1.0, 1.0)),
            float(np.clip(2.0 * self.speed[near] / MAX_SPEED - 1.0, -1.0, 1.0)),
        ])
