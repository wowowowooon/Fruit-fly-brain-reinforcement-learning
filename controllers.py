"""학습을 쓰지 않는 고전 컨트롤러.

비교 실험의 기준선이자 모방학습의 교사다. 한 곳에만 두고 모두가 이걸 불러 쓴다.
여러 파일에 복붙해 두면 조건이 어긋나서 비교가 무의미해진다.
"""

import numpy as np

from f1tenth_sim import CAR_WIDTH, MAX_SPEED, MAX_STEER

FORWARD_FOV = np.deg2rad(100)  # 이 밖은 안 본다. 뒤가 뚫렸다고 후진할 수는 없다
SAFETY_MARGIN = 0.25           # 차폭에 더해 둘 여유 [m]
SMOOTH_DEG = 5.0               # 빔을 평균 낼 각도 폭. 빔 개수가 변해도 같은 시야 폭
GAP_MIN_RANGE = 2.0            # 이보다 가까우면 지나갈 구간으로 안 친다 [m]
CRUISE_MIN = 1.8               # 아무리 막혀도 이 속도는 낸다 [m/s]
CRUISE_MAX = 6.5               # 앞이 완전히 뚫렸을 때 속도 [m/s]
LOOKAHEAD = 6.0                # 이만큼 앞이 비면 최고 속도로 본다 [m]


def _longest_gap(open_mask):
    """True가 가장 길게 이어지는 구간의 (시작, 끝). 없으면 None."""
    edges = np.flatnonzero(np.diff(np.concatenate(
        ([0], open_mask.astype(np.int8), [0])
    )))
    if len(edges) == 0:
        return None
    starts, ends = edges[0::2], edges[1::2]
    widest = int(np.argmax(ends - starts))
    return int(starts[widest]), int(ends[widest])


def follow_the_gap(scan, beam_angles, lidar_range):
    """가장 넓게 뚫린 구간의 한가운데로 도는 고전 방식.

    1. 전방만 본다
    2. 가장 가까운 벽 주변을 차폭만큼 막는다 (거기로 가면 긁힌다)
    3. 지나갈 수 있는 구간 중 가장 넓은 곳을 찾아 그 한가운데로 조향
       "가장 먼 방향"이 아니라 "한가운데"인 이유: 사거리 끝에서는 값이 전부
       10.00m로 같아져서 가장 먼 방향이라는 게 의미를 잃는다
    4. 앞이 막혀 있거나 많이 꺾을수록 느리게
    """
    forward = np.abs(beam_angles) <= FORWARD_FOV
    angles = beam_angles[forward]
    view = np.minimum(scan[forward], lidar_range).astype(np.float64)

    # 빔 간격이 달라져도 같은 각도만큼 평균 낸다
    step = float(np.abs(np.diff(beam_angles)).mean())
    window = max(1, int(np.deg2rad(SMOOTH_DEG) / step))
    if window > 1:
        kernel = np.ones(window) / window
        smooth = np.convolve(view, kernel, mode="same")
        # mode="same"은 양 끝을 0으로 채워서 가장자리를 낮게 만든다. 되돌린다.
        norm = np.convolve(np.ones_like(view), kernel, mode="same")
        view = smooth / norm

    # 가장 가까운 벽 주위를 막는다. 차가 지나갈 수 없는 폭이다.
    nearest = int(np.argmin(view))
    blocked = np.arctan2(CAR_WIDTH / 2 + SAFETY_MARGIN, max(view[nearest], 0.05))
    free = view.copy()
    free[np.abs(angles - angles[nearest]) <= blocked] = 0.0

    span = _longest_gap(free >= GAP_MIN_RANGE)
    if span is None:                      # 사방이 막혔다. 제일 먼 쪽으로라도 간다
        target = angles[int(np.argmax(free))]
    else:
        start, end = span
        target = float(angles[start:end].mean())

    steer = float(np.clip(target / MAX_STEER, -1, 1))

    # 정면이 얼마나 뚫렸는지로 속도를 정한다
    ahead = float(view[np.abs(angles) <= np.deg2rad(20)].min())
    room = np.clip(ahead / LOOKAHEAD, 0.0, 1.0)
    speed = CRUISE_MIN + (CRUISE_MAX - CRUISE_MIN) * room * (1.0 - 0.7 * abs(steer))
    # 환경은 -1~1을 0~MAX_SPEED로 읽는다. m/s를 그 규격으로 되돌린다.
    command = float(np.clip(2.0 * speed / MAX_SPEED - 1.0, -1.0, 1.0))
    return np.array([steer, command])


class GapFollower:
    """FlyDriver와 같은 방식으로 쓰기 위한 껍데기. 내부 상태는 없다."""

    name = "고전 (Follow the Gap)"

    def __init__(self, env):
        self.beam_angles = env.beam_angles
        self.lidar_range = env.lidar_range

    def reset(self):
        pass

    def act(self, obs):
        return follow_the_gap(
            obs[:-1] * self.lidar_range, self.beam_angles, self.lidar_range
        )
