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

# --- 차량 (Traxxas 1:10, F1TENTH 표준 섀시) ---
WHEELBASE = 0.33        # 뒷축에서 앞축까지 [m]
CAR_WIDTH = 0.31        # 바퀴 바깥까지의 폭 [m]
CAR_LENGTH = 0.58       # 뒤 범퍼에서 앞 범퍼까지 [m]
REAR_OVERHANG = 0.10    # 뒷축 뒤로 튀어나온 길이 [m]
MAX_STEER = 0.42        # 최대 조향각 [rad]
MAX_SPEED = 7.0         # 목표 최고 속도 [m/s]. 보수적으로 잡았다
MAX_ACCEL = 4.0         # 속도 변화율 한계 [m/s^2]

# 차체가 뒷축 기준으로 차지하는 앞뒤 범위. 충돌 판정에 쓴다.
BODY_BACK = -REAR_OVERHANG
BODY_FRONT = CAR_LENGTH - REAR_OVERHANG
BODY_RADIUS = CAR_WIDTH / 2

# --- 라이다 (Hokuyo UST-10LX, 대회 규정 필수 센서) ---
LIDAR_FOV = np.deg2rad(270)
LIDAR_RANGE = 10.0      # 최대 측정 거리 [m]. 이 값이 바뀌면 재학습해야 한다
LIDAR_BEAMS = 1081      # 270도 / 0.25도 + 1
LIDAR_HZ = 40.0         # 25ms 주기
CONTROL_DT = 1.0 / LIDAR_HZ
LIDAR_AHEAD = 0.38      # 뒷축에서 라이다까지 [m]. 차량 앞쪽에 장착
# 라이다에서 본 차체. 앞으로는 조금, 뒤로는 많이 튀어나와 있다.
#   앞 범퍼 +0.10m / 뒤 범퍼 -0.48m / 좌우 각 0.155m
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
WIDTH_RANGE = (1.6, 4.0)       # 주행로 폭 [m]. 실제 코스는 2m 안팎으로 좁다
ASPECT_RANGE = (1.0, 3.0)      # 가로:세로 비율. 실제 코스는 2.7:1 로 길쭉하다
HARMONICS = (2, 3, 4, 5, 6)    # 높을수록 코너가 날카로워진다
CURVE_MARGIN = 1.35            # 곡률 반경이 폭 절반의 이 배는 되어야 함


def _offset_walls(center, width):
    """중심선 좌우로 폭의 절반만큼 떨어진 벽."""
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return center - normal * width / 2, center + normal * width / 2


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
    return bool(gap[far_apart].min() < width * 1.15)


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


def _circuit_loop(rng, radius, points):
    """직선과 코너로 이루어진 서킷형. 실제 대회 코스가 이 모양이다."""
    corners = int(rng.integers(6, 12))
    angles = np.sort(rng.uniform(0, 2 * np.pi, corners))
    # 꼭짓점이 너무 몰리면 급코너가 되므로 최소 간격을 준다
    angles = np.linspace(0, 2 * np.pi, corners, endpoint=False) + \
        rng.uniform(-0.25, 0.25, corners)
    reach = radius * rng.uniform(0.55, 1.15, corners)
    polygon = np.stack([reach * np.cos(angles), reach * np.sin(angles)], axis=1)
    # 적게 깎으면 코너가 날카롭고, 많이 깎으면 둥글어진다
    return _resample(_chaikin(polygon, int(rng.integers(3, 5))), points)


def make_track(seed=0, points=400):
    """폐곡선 트랙. 중심선과 좌우 벽을 돌려준다.

    시드마다 크기, 폭, 코너 모양이 다르고 두 가지 형태가 섞여 나온다.
      매끈한 곡선형  코너가 이어지는 오벌
      서킷형        직선 구간과 코너가 번갈아 나옴 (대회 코스에 가까움)
    차가 지나갈 수 없는 모양이 나오면 버리고 다시 뽑는다.
    같은 시드는 항상 같은 트랙이 나온다.
    """
    radius = width = None
    for attempt in range(80):
        rng = np.random.default_rng([seed, attempt])
        radius = rng.uniform(*RADIUS_RANGE)
        width = rng.uniform(*WIDTH_RANGE)
        shape = _circuit_loop if rng.random() < 0.5 else _smooth_loop

        center = shape(rng, radius, points)
        # 길쭉하게 눌러서 실제 코스 같은 비율을 만든다
        aspect = rng.uniform(*ASPECT_RANGE)
        center = center / np.array([1.0, aspect])
        center = _resample(center, points)
        if _min_curve_radius(center) < width / 2 * CURVE_MARGIN:
            continue                      # 벽이 접히는 급커브
        if _self_overlaps(center, width):
            continue                      # 트랙끼리 겹침

        inner, outer = _offset_walls(center, width)
        return center, inner, outer

    # 여기까지 오면 운이 나쁜 시드다. 흔들림 없는 원으로 돌려준다.
    theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
    center = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    inner, outer = _offset_walls(center, width)
    return center, inner, outer


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
                 latency=CONTROL_LATENCY):
        super().__init__()
        self.n_beams = n_beams
        self.dt = dt
        self.max_steps = max_steps
        self.track_seed = track_seed
        self.lidar_range = lidar_range
        # 명령이 실제로 먹히기까지 몇 스텝 걸리는지
        self.latency_steps = int(round(latency / dt))
        self.beam_angles = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, n_beams)

        # 관측: 라이다 거리(0~1) + 현재 속도(0~1)
        self.observation_space = spaces.Box(0.0, 1.0, (n_beams + 1,), np.float32)
        # 행동: 조향(-1~1), 속도(-1~1)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

        self._load_track(track_seed)

    def _load_track(self, seed):
        self.center, inner, outer = make_track(seed)
        a1, b1 = as_segments(inner)
        a2, b2 = as_segments(outer)
        self.seg_a = np.concatenate([a1, a2])
        self.seg_b = np.concatenate([b1, b2])
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
        return np.concatenate([
            scan / self.lidar_range,
            [self.speed / MAX_SPEED],
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
        self.steps = 0
        self.lap_progress = 0
        self.prev_index = 0
        self.crashed = False
        # 아직 안 먹은 명령들. 지연을 흉내 낸다.
        self.queued = deque(
            [np.array([0.0, -1.0])] * self.latency_steps,
            maxlen=max(self.latency_steps, 1),
        )

        scan = self._scan()
        return self._observe(scan), {}

    def step(self, action):
        # 지금 낸 명령은 줄 뒤에 서고, 앞에 있던 명령이 지금 먹는다.
        action = np.asarray(action, dtype=np.float64)
        if self.latency_steps > 0:
            self.queued.append(action)
            action = self.queued.popleft()

        steer = float(np.clip(action[0], -1, 1)) * MAX_STEER
        target = (float(np.clip(action[1], -1, 1)) + 1) / 2 * MAX_SPEED

        # 속도는 즉시 바뀌지 않는다
        limit = MAX_ACCEL * self.dt
        self.speed += np.clip(target - self.speed, -limit, limit)

        # 자전거 모델 (뒷축 기준)
        self.pos = self.pos + self.speed * self.dt * self._heading()
        self.yaw += self.speed / WHEELBASE * np.tan(steer) * self.dt
        self.steps += 1

        scan = self._scan()
        # 라이다 한 점이 아니라 차체 전체로 판정한다.
        self.crashed = bool(self._clearance() < BODY_RADIUS)

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

        done = self.crashed or self.lap_progress >= step_count
        truncated = self.steps >= self.max_steps

        info = {
            "progress": self.lap_progress / step_count,
            "speed": self.speed,
            "crashed": self.crashed,
            "min_range": float(scan.min()),
            "clearance": self._clearance(),
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
