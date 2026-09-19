"""ROBORACER(F1TENTH) 규격 경량 주행 시뮬.

공식 f1tenth_gym이 Python 3.8과 구버전 gym을 요구해서, 같은 규격으로 직접 만들었다.
  센서  2D 라이다, 270도 시야
  차량  자전거 모델, 축거 0.33m, 조향 +-0.42 rad
  제어  조향각과 속도를 연속값으로 받음
  트랙  벽으로 둘러싸인 폐곡선. 시드마다 모양이 다름
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

WHEELBASE = 0.33        # 축거 [m]
CAR_WIDTH = 0.31        # 차폭 [m]
MAX_STEER = 0.42        # 최대 조향각 [rad]
MAX_SPEED = 8.0         # 최대 속도 [m/s]
LIDAR_FOV = np.deg2rad(270)
LIDAR_RANGE = 30.0      # 라이다 최대 거리 [m]


def make_track(seed=0, points=400, radius=18.0, width=3.2):
    """부드러운 폐곡선 트랙. 중심선과 좌우 벽을 돌려준다."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, points, endpoint=False)

    r = np.full(points, radius)
    for harmonic in (2, 3, 4):
        r += rng.uniform(-0.16, 0.16) * radius * np.cos(
            harmonic * theta + rng.uniform(0, 2 * np.pi)
        )

    center = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    return center, center - normal * width / 2, center + normal * width / 2


def as_segments(loop):
    """폐곡선 점열을 선분 시작/끝 배열로."""
    return loop, np.roll(loop, -1, axis=0)


def raycast(origin, angles, seg_a, seg_b, max_range):
    """한 지점에서 여러 방향으로 쏜 광선이 벽에 닿는 거리."""
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

    def __init__(self, n_beams=108, dt=0.05, max_steps=2000, track_seed=0):
        super().__init__()
        self.n_beams = n_beams
        self.dt = dt
        self.max_steps = max_steps
        self.track_seed = track_seed
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

    def _scan(self):
        angles = self.beam_angles + self.yaw
        return raycast(self.pos, angles, self.seg_a, self.seg_b, LIDAR_RANGE)

    def _observe(self, scan):
        return np.concatenate([
            scan / LIDAR_RANGE,
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

        self.pos = start.copy()
        self.yaw = float(np.arctan2(heading[1], heading[0]))
        self.speed = 2.0
        self.steps = 0
        self.lap_progress = 0
        self.prev_index = 0
        self.crashed = False

        scan = self._scan()
        return self._observe(scan), {}

    def step(self, action):
        steer = float(np.clip(action[0], -1, 1)) * MAX_STEER
        target = (float(np.clip(action[1], -1, 1)) + 1) / 2 * MAX_SPEED

        # 속도는 즉시 바뀌지 않는다
        self.speed += np.clip(target - self.speed, -4.0 * self.dt, 4.0 * self.dt)

        # 자전거 모델
        self.pos = self.pos + self.speed * self.dt * np.array(
            [np.cos(self.yaw), np.sin(self.yaw)]
        )
        self.yaw += self.speed / WHEELBASE * np.tan(steer) * self.dt
        self.steps += 1

        scan = self._scan()
        self.crashed = bool(scan.min() < CAR_WIDTH / 2)

        # 중심선을 따라 얼마나 나아갔는지로 보상
        index = self._progress()
        step_count = len(self.center)
        delta = (index - self.prev_index + step_count // 2) % step_count - step_count // 2
        self.prev_index = index
        self.lap_progress += delta

        reward = delta * 1.0 + self.speed * 0.01
        if self.crashed:
            reward -= 50.0

        done = self.crashed or self.lap_progress >= step_count
        truncated = self.steps >= self.max_steps

        info = {
            "progress": self.lap_progress / step_count,
            "speed": self.speed,
            "crashed": self.crashed,
            "min_range": float(scan.min()),
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
        ends = self.pos + scan[:, None] * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )
        for end in ends[::6]:
            ax.plot([self.pos[0], end[0]], [self.pos[1], end[1]],
                    color="tab:orange", linewidth=0.4, alpha=0.5)

        ax.plot(*self.pos, marker="o", color="tab:green", markersize=8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"speed {self.speed:.1f} m/s   progress {self.lap_progress / len(self.center):.1%}")

        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return image
