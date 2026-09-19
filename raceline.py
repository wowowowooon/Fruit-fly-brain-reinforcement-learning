"""레이싱 라인을 뽑고 Stanley 로 따라간다.

팀 저장소 Roboracer-AJOU/jetson_code 의 실차 방식을 그대로 옮긴 것이다.
  라인   scripts/generate_raceline_from_centerline.py  최소곡률 최적화
  추종   path_following/stanley_waypoint_follow_node.py  Stanley

FGM 과 다른 점이 핵심이다. FGM 은 그때그때 뚫린 쪽으로 가므로 트랙 한가운데를
달린다. 레이싱 라인은 코너를 미리 깎아 들어가므로 훨씬 빠르고, 급조향도 실제로 쓴다.

이 교사는 자기 위치와 트랙 전체 모양을 안다(맵 기반). 학생인 초파리 회로는
라이다만 본다. 아는 게 많은 교사를 모르는 학생이 흉내 내게 하는 방식이다.
"""

import numpy as np

from f1tenth_sim import MAX_STEER, MAX_SPEED, WHEELBASE

# --- Stanley (실차 파라미터) ---
STANLEY_K = 1.8            # 경로 이탈 복귀 세기
STANLEY_SOFTENING = 0.12   # 저속에서 발산하지 않게 분모에 더하는 값
HEADING_GAIN = 1.0         # 방향 오차 비중
HEADING_BLEND_M = 0.45     # 이탈이 이보다 크면 방향항을 줄인다

# --- 속도 프로파일 ---
LAT_ACCEL = 6.0            # 코너에서 버틸 수 있는 횡가속도 [m/s^2]
LONG_ACCEL = 4.5           # 가감속 한계 [m/s^2]
SPEED_MIN = 1.6            # 아무리 급해도 이 속도는 낸다 [m/s]

# --- 라인 ---
# 차는 점이 아니다. 앞이 0.50m 나 되어서 코너에서 앞 외측 모서리가 경로보다
# 더 바깥을 지난다. 팀 코드 vehicle_geometry.outer_half_width 기준으로
# 반경 1m 코너에서 0.25m 까지 벌어진다. 그만큼 벽에서 띄워야 한다.
LINE_CLEARANCE = 0.30      # 벽에서 띄울 거리 [m]


def _normals(points):
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)


def _second_difference(n):
    """폐곡선용 2계 차분 행렬. 곡률에 해당한다."""
    matrix = np.zeros((n, n))
    index = np.arange(n)
    matrix[index, index] = -2.0
    matrix[index, (index - 1) % n] = 1.0
    matrix[index, (index + 1) % n] = 1.0
    return matrix


def minimum_curvature_line(center, half_width, rounds=12):
    """중심선에서 좌우로 얼마나 비킬지 정해 곡률 총합을 최소화한다.

    경로를 P = 중심선 + a*법선 으로 두면 곡률 제곱합이 a 에 대한 이차식이 된다.
    벽 안쪽이라는 제약만 걸고 푼다.

    경사하강은 이 문제에서 거의 안 움직인다(4계 차분이라 조건수가 나쁘다).
    직접 풀고, 벽을 넘은 점은 고정한 뒤 나머지를 다시 푸는 식으로 반복한다.
    """
    count = len(center)
    normal = _normals(center)
    d2 = _second_difference(count)
    base = d2 @ center                        # 중심선 자체의 굽음
    mx = d2 * normal[:, 0]                    # a 가 x 성분에 주는 영향
    my = d2 * normal[:, 1]

    gram = mx.T @ mx + my.T @ my
    bias = mx.T @ base[:, 0] + my.T @ base[:, 1]
    gram += np.eye(count) * (1e-6 * np.trace(gram) / count + 1e-9)
    limit = float(np.maximum(half_width - LINE_CLEARANCE, 0.02))

    alpha = np.zeros(count)
    fixed = np.zeros(count, dtype=bool)
    for _ in range(rounds):
        free = ~fixed
        if not free.any():
            break
        rhs = -bias[free] - gram[np.ix_(free, fixed)] @ alpha[fixed]
        try:
            alpha[free] = np.linalg.solve(gram[np.ix_(free, free)], rhs)
        except np.linalg.LinAlgError:
            break
        over = np.abs(alpha) > limit
        if not over.any():
            break
        alpha = np.clip(alpha, -limit, limit)
        fixed |= over
    return center + np.clip(alpha, -limit, limit)[:, None] * normal


def path_curvature(points):
    """각 점에서의 곡률 [1/m]."""
    d1 = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / 2
    d2 = np.roll(points, -1, axis=0) - 2 * points + np.roll(points, 1, axis=0)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    speed = np.linalg.norm(d1, axis=1)
    return cross / np.maximum(speed ** 3, 1e-9)


def speed_profile(points, v_max=MAX_SPEED):
    """코너에서 미리 줄이고 탈출하며 올리는 속도 계획.

    1. 각 점에서 횡가속도 한계로 낼 수 있는 최고 속도
    2. 앞뒤로 훑으며 가감속 한계를 지키게 다듬는다
    """
    kappa = np.abs(path_curvature(points))
    limit = np.sqrt(LAT_ACCEL / np.maximum(kappa, 1e-6))
    speed = np.clip(limit, SPEED_MIN, v_max)

    gap = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    count = len(points)
    for _ in range(2):                        # 폐곡선이라 두 바퀴 돌려야 수렴
        for i in range(count):                # 감속: 앞의 코너에 맞춰 미리 줄인다
            j = (count - 1 - i) % count
            nxt = (j + 1) % count
            speed[j] = min(speed[j], np.sqrt(
                speed[nxt] ** 2 + 2 * LONG_ACCEL * gap[j]))
        for i in range(count):                # 가속: 낼 수 있는 만큼만 올린다
            nxt = (i + 1) % count
            speed[nxt] = min(speed[nxt], np.sqrt(
                speed[i] ** 2 + 2 * LONG_ACCEL * gap[i]))
    return speed


class StanleyRacer:
    """레이싱 라인을 Stanley 로 따라간다. 자기 위치를 아는 교사."""

    name = "교사 (레이싱 라인 + Stanley)"

    def __init__(self, center, half_width, v_max=MAX_SPEED):
        self.line = minimum_curvature_line(center, half_width)
        self.speed = speed_profile(self.line, v_max)
        self.heading = np.arctan2(
            np.roll(self.line, -1, axis=0)[:, 1] - self.line[:, 1],
            np.roll(self.line, -1, axis=0)[:, 0] - self.line[:, 0],
        )

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
        cte = (offset[0] * np.sin(path_yaw) - offset[1] * np.cos(path_yaw))

        cte_term = np.arctan2(STANLEY_K * cte, abs(speed) + STANLEY_SOFTENING)
        raw_heading = HEADING_GAIN * heading_error
        if raw_heading * cte_term > 0.0:
            weight = 1.0                      # 둘 다 안쪽으로 민다. 억제할 이유 없음
        else:
            weight = max(0.0, 1.0 - abs(cte) / HEADING_BLEND_M)
        steer = cte_term + weight * raw_heading

        target = self.speed[near]
        return np.array([
            float(np.clip(steer / MAX_STEER, -1.0, 1.0)),
            float(np.clip(2.0 * target / MAX_SPEED - 1.0, -1.0, 1.0)),
        ])
