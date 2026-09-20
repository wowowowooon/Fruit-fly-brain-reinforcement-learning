"""학습을 쓰지 않는 고전 컨트롤러.

비교 실험의 기준선이자 모방학습의 교사다. 한 곳에만 두고 모두가 이걸 불러 쓴다.
여러 파일에 복붙해 두면 조건이 어긋나서 비교가 무의미해진다.

교사는 팀 저장소 Roboracer-AJOU/jetson_code 의 FGM 노드를 그대로 옮긴 것이다.
  src/race_pkg/race_pkg/planning/fgm_node.py
실제 차에 올라가는 알고리즘을 흉내 내야 실차로 옮길 때 말이 된다.
여기서 임의로 바꾸면 안 된다. 팀 코드가 바뀌면 이 파일도 같이 고친다.
"""

import numpy as np

from fly_driver import N_MOTION
from f1tenth_sim import LIDAR_AHEAD, MAX_SPEED, MAX_STEER, WHEELBASE

# --- 팀 FGM 노드 파라미터 (fgm_node.py 와 같아야 함) ---
PREPROCESS_DIST = 5.0     # 이 거리 밖은 전부 5m 로 잘라서 본다
BUBBLE_RADIUS = 0.20      # 가장 가까운 점 주위를 지워 버리는 반경 [m]
GAP_THRESHOLD = 1.2       # 이보다 멀면 '뚫린 곳'. 아무 데도 없으면 0.5 로 낮춘다
GAP_FALLBACK = 0.5
TARGET_DIST = 3.0         # 갭 방향으로 이만큼 앞에 목표점을 둔다 [m]

# --- 목표점을 조향으로 바꾸는 부분 ---
# 팀 스택에서는 local_planner/control_node 가 맡는다. 그쪽은 맵과 경로를 함께
# 쓰는 큰 코드라 그대로 옮길 수 없어서, 같은 목표점을 쓰는 표준 pure pursuit 로 둔다.
LOOKAHEAD = TARGET_DIST

# --- 속도 (팀 코드에 대응물이 없어 여기서 정한다) ---
CRUISE_MIN = 1.8          # 아무리 막혀도 이 속도는 낸다 [m/s]
CRUISE_MAX = 6.5          # 앞이 완전히 뚫렸을 때 속도 [m/s]
LOOKAHEAD_FREE = 4.0      # 이만큼 앞이 비면 최고 속도로 본다 [m]


def _bubble(ranges, center, distance, angle_step, radius=BUBBLE_RADIUS):
    """가장 가까운 점 주위를 통째로 지운다. 팀 코드의 _create_bubble 과 같다."""
    safe_theta = np.arctan(radius / (distance + 0.001))
    span = int(safe_theta / angle_step)
    lo = max(0, center - span)
    hi = min(len(ranges), center + span)
    ranges[lo:hi] = 0.0


def fgm_target(scan, beam_angles):
    """팀 FGM 노드가 내놓는 목표 방향. 없으면 None."""
    angle_step = float(np.abs(np.diff(beam_angles)).mean())
    ranges = np.minimum(np.nan_to_num(scan, nan=0.0), PREPROCESS_DIST)

    closest = int(np.argmin(ranges))
    if ranges[closest] < PREPROCESS_DIST:
        _bubble(ranges, closest, float(ranges[closest]), angle_step)

    for threshold in (GAP_THRESHOLD, GAP_FALLBACK):
        open_idx = np.flatnonzero(ranges > threshold)
        if len(open_idx):
            break
    else:
        return None

    # 이어진 덩어리 중 가장 긴 것의 한가운데
    gaps = np.split(open_idx, np.flatnonzero(np.diff(open_idx) > 1) + 1)
    widest = max(gaps, key=len)
    return float(beam_angles[widest[len(widest) // 2]])


def follow_the_gap(scan, beam_angles, lidar_range):
    """팀 FGM + pure pursuit. 환경 규격인 [조향, 속도] 를 -1~1 로 돌려준다."""
    target = fgm_target(scan, beam_angles)
    if target is None:                      # 사방이 막혔다. 제일 먼 쪽으로
        target = float(beam_angles[int(np.argmax(scan))])

    # pure pursuit: 라이다 기준 목표점을 뒷축 기준으로 옮겨서 조향각을 낸다
    point = LOOKAHEAD * np.array([np.cos(target), np.sin(target)])
    point[0] += LIDAR_AHEAD
    reach = float(np.hypot(*point))
    curvature = 2.0 * point[1] / max(reach ** 2, 1e-6)
    steer = float(np.clip(np.arctan(WHEELBASE * curvature) / MAX_STEER, -1, 1))

    # 정면이 얼마나 뚫렸는지로 속도를 정한다
    ahead = float(scan[np.abs(beam_angles) <= np.deg2rad(20)].min())
    room = np.clip(ahead / LOOKAHEAD_FREE, 0.0, 1.0)
    speed = CRUISE_MIN + (CRUISE_MAX - CRUISE_MIN) * room * (1.0 - 0.7 * abs(steer))
    # 환경은 -1~1 을 0~MAX_SPEED 로 읽는다. m/s 를 그 규격으로 되돌린다.
    command = float(np.clip(2.0 * speed / MAX_SPEED - 1.0, -1.0, 1.0))
    return np.array([steer, command])


class GapFollower:
    """FlyDriver와 같은 방식으로 쓰기 위한 껍데기. 내부 상태는 없다."""

    name = "고전 (팀 FGM)"

    def __init__(self, env):
        self.beam_angles = env.beam_angles
        self.lidar_range = env.lidar_range

    def reset(self):
        pass

    def act(self, obs):
        return follow_the_gap(
            obs[:-N_MOTION] * self.lidar_range, self.beam_angles, self.lidar_range
        )
