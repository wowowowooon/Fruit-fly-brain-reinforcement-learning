"""맵 위에서 출발 자세를 정하고, 한 바퀴 궤적을 따 온다.

원래 test_on_map.py 안에 있었는데 평가 스크립트 여섯 개가 거기서
가져다 썼다. 보는 도구 안에 라이브러리가 들어 있으면, 그 도구를
옮기거나 고칠 때마다 평가가 같이 흔들린다.
"""

import numpy as np

from flyracer.drive.controllers import follow_the_gap
from flyracer.sim.f1tenth_sim import BODY_RADIUS


def _signed_area(path):
    """폐곡선의 부호 있는 면적. 양수면 반시계."""
    nxt = np.roll(path, -1, axis=0)
    return 0.5 * float(np.sum(path[:, 0] * nxt[:, 1] - nxt[:, 0] * path[:, 1]))


def find_start(track, env, counterclockwise=True):
    """중심선 위에 차를 놓고, 코스가 뻗은 방향으로 맞춘다.

    코스는 양쪽으로 다 뚫려 있어서 앞만 보고 고르면 방향이 반반이다.
    예전에는 한 바퀴 돌려 보고 정했는데, 그 주행이 실패하는 코스에서는
    방향을 못 정할 뿐 아니라 중심선까지 망가뜨렸다. 중심선 자체가
    폐곡선이므로 도는 방향은 거기서 바로 읽으면 된다.

    대회 주행 방향을 맞춰야 학습한 모델을 그대로 올릴 수 있다.
    """
    line = np.asarray(env.center, dtype=float)
    if len(line) < 8:
        line = track.centerline()
    # 원하는 방향으로 돌게 점 순서를 맞춘다 (양수 넓이 = 반시계)
    if (_signed_area(line) > 0) != counterclockwise:
        line = line[::-1]
        env.set_centerline(line)

    # 가장 넓은 곳에서 가장 가까운 중심선 점에서 출발한다
    widest, _ = track.widest_point()
    index = int(np.argmin(np.linalg.norm(line - widest, axis=1)))
    position = line[index].copy()
    ahead = line[(index + 3) % len(line)] - position
    heading = float(np.arctan2(ahead[1], ahead[0]))
    return position, heading


def trace_lap(env, max_steps=8000):
    """고전 컨트롤러로 한 바퀴 돌며 경로를 기록한다. 이게 중심선이 된다."""
    env.center = np.zeros((2, 2))       # 진행도 계산을 잠시 끈다
    obs, _ = env.reset()
    start = env.pos.copy()
    path, left_start = [], False

    for _ in range(max_steps):
        path.append(env.pos.copy())
        action = follow_the_gap(obs[:-env.motion] * env.lidar_range,
                                env.beam_angles, env.lidar_range)
        obs, _, _, _, info = env.step(action)
        if env._clearance() < BODY_RADIUS:
            return np.array(path), False
        gap = np.linalg.norm(env.pos - start)
        if gap > 4.0:
            left_start = True
        elif left_start and gap < 1.0 and len(path) > 200:
            return np.array(path), True
    return np.array(path), False
