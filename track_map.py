"""ROS 점유격자 지도(.png + .yaml)를 시뮬 트랙으로 읽어들인다.

대회에서 배포하는 맵이 이 형식이다. 학습에는 쓰지 않고, 학습이 끝난 모델이
처음 보는 실제 코스에서 달릴 수 있는지 확인하는 데 쓴다.

  free_map = MapTrack("ifac_roboracer.yaml")
  env = MapEnv(free_map)
"""

from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

from f1tenth_sim import F1TenthEnv, as_segments


def _wall_segments(free, resolution, origin):
    """주행 가능 영역의 경계를 선분 목록으로 바꾼다.

    격자에서 '뚫린 칸'과 '막힌 칸'이 맞닿는 면이 곧 벽이다.
    그 면들을 그대로 짧은 선분으로 만든다.
    """
    rows, cols = free.shape
    ox, oy = origin[0], origin[1]

    def world(cx, cy):
        # 이미지 y축은 아래로, 월드 y축은 위로 간다
        return np.stack([ox + cx * resolution,
                         oy + (rows - cy) * resolution], axis=1)

    starts, ends = [], []
    # 세로 경계 (왼쪽/오른쪽 이웃과 다른 곳)
    edge = free[:, :-1] != free[:, 1:]
    y, x = np.nonzero(edge)
    starts.append(world(x + 1.0, y))
    ends.append(world(x + 1.0, y + 1.0))
    # 가로 경계 (위/아래 이웃과 다른 곳)
    edge = free[:-1, :] != free[1:, :]
    y, x = np.nonzero(edge)
    starts.append(world(x, y + 1.0))
    ends.append(world(x + 1.0, y + 1.0))

    return np.concatenate(starts), np.concatenate(ends)


class MapTrack:
    """맵 파일에서 벽과 주행 가능 영역을 읽어 둔다."""

    def __init__(self, yaml_path):
        # 윈도우에서 역슬래시 경로가 들어와도 되게 pathlib 으로 다룬다
        yaml_path = Path(yaml_path)
        with yaml_path.open(encoding="utf-8") as handle:
            meta = yaml.safe_load(handle)

        image = Image.open(yaml_path.parent / meta["image"]).convert("L")
        grid = np.asarray(image, dtype=np.float32) / 255.0
        if meta.get("negate", 0):
            grid = 1.0 - grid

        self.name = Path(meta["image"]).stem
        self.resolution = float(meta["resolution"])
        self.origin = meta["origin"]
        # free_thresh 보다 비어 있으면 주행 가능. 회색(미탐색)은 벽으로 본다.
        self.free = grid > (1.0 - float(meta.get("free_thresh", 0.196)))
        self.seg_a, self.seg_b = _wall_segments(
            self.free, self.resolution, self.origin
        )

        # 벽에서 가장 멀리 떨어진 곳이 코스 한가운데다
        self.clearance = ndimage.distance_transform_edt(self.free) * self.resolution

    def to_world(self, row, col):
        rows = self.free.shape[0]
        return np.array([self.origin[0] + col * self.resolution,
                         self.origin[1] + (rows - row) * self.resolution])

    def widest_point(self):
        row, col = np.unravel_index(np.argmax(self.clearance), self.clearance.shape)
        return self.to_world(row, col), float(self.clearance[row, col])

    def summary(self):
        rows, cols = self.free.shape
        drivable = self.free.sum() * self.resolution ** 2
        width = np.percentile(self.clearance[self.free], 90) * 2
        return (
            f"{self.name}  {cols * self.resolution:.1f} x "
            f"{rows * self.resolution:.1f} m\n"
            f"주행 가능 {drivable:.0f} m^2, 통로 폭 약 {width:.1f} m "
            f"(가장 넓은 곳 {self.clearance.max() * 2:.1f} m)\n"
            f"벽 선분 {len(self.seg_a):,}개"
        )


class MapEnv(F1TenthEnv):
    """실제 맵 위에서 달리는 환경. 나머지 물리와 센서는 그대로다."""

    def __init__(self, track, start_pose=None, **kwargs):
        self.track = track
        self.start_pose = start_pose
        super().__init__(track_seed=0, **kwargs)

    def _load_track(self, seed):
        self.seg_a, self.seg_b = self.track.seg_a, self.track.seg_b
        self.seg_mid = (self.seg_a + self.seg_b) / 2
        self.seg_reach = (
            np.linalg.norm(self.seg_b - self.seg_a, axis=1) / 2 + self.lidar_range
        )
        # 중심선은 나중에 한 바퀴 돌려서 채운다. 그 전에는 임시값.
        self.center = getattr(self, "center", np.zeros((2, 2)))
        self.inner = self.outer = np.zeros((0, 2))
        self.step_length = self.track.resolution
        self.lap_length = 0.0
        self.track_width = float(
            np.percentile(self.track.clearance[self.track.free], 90) * 2
        )

    def set_centerline(self, path):
        """한 바퀴 분량의 경로를 중심선으로 삼는다. 진행도 계산에 쓴다."""
        self.center = np.asarray(path, dtype=np.float64)
        steps = np.linalg.norm(
            np.roll(self.center, -1, axis=0) - self.center, axis=1
        )
        self.step_length = float(steps.mean())
        self.lap_length = float(steps.sum())

    def reset(self, seed=None, options=None):
        if self.start_pose is None:
            raise RuntimeError("출발 자세를 먼저 정해야 한다 (find_start 사용)")
        position, yaw = self.start_pose
        obs, info = super().reset(seed=seed, options=options)
        self.pos = np.asarray(position, dtype=np.float64).copy()
        self.yaw = float(yaw)
        self.speed = 1.5
        self.prev_index = self._progress()
        self.lap_progress = 0
        return self._observe(self._scan()), info
