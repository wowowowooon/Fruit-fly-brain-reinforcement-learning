"""실제 대회 맵에서 컨트롤러를 돌려 본다.

학습에는 쓰지 않는다. 처음 보는 코스에서 부딪히지 않고 얼마나 빨리 도는지만 본다.

  .\\.venv\\Scripts\\python.exe test_on_map.py "C:/.../ifac_roboracer.yaml"
"""

import argparse
from pathlib import Path

import numpy as np

from controllers import follow_the_gap
from paths import model as model_path
from fly_driver import N_MOTION
from f1tenth_sim import BODY_RADIUS
from track_map import MapEnv, MapTrack


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
        action = follow_the_gap(obs[:-N_MOTION] * env.lidar_range,
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


def run(env, controller, label, max_steps=8000):
    obs, _ = env.reset()
    controller_reset = getattr(controller, "reset", None)
    if controller_reset:
        controller_reset()
    worst = np.inf
    for _ in range(max_steps):
        action = controller(obs) if callable(controller) else controller.act(obs)
        obs, _, done, truncated, info = env.step(action)
        worst = min(worst, info["clearance"])
        if done or truncated:
            break
    print(f"  {label:<22} {info['lap_time']:>6.2f}초  진행 {info['progress']:>6.1%}  "
          f"최소 여유 {worst - BODY_RADIUS:>5.2f} m  "
          f"{'충돌' if info['crashed'] else '완주'}")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map_yaml")
    parser.add_argument("--model", default="C_fly871.pt")
    args = parser.parse_args()

    track = MapTrack(args.map_yaml)
    print(track.summary(), "\n")

    env = MapEnv(track)
    env.start_pose = find_start(track, env)
    print(f"출발 위치 {np.round(env.start_pose[0], 2)}  "
          f"방향 {np.rad2deg(env.start_pose[1]):.0f}도")

    # 중심선은 MapEnv 가 맵 모양에서 직접 뽑아 이미 갖고 있다.
    # 예전처럼 고전 컨트롤러로 한 바퀴 돌려 덮어쓰면, 그 주행이 실패했을 때
    # 잘린 경로가 중심선이 되어 진행도가 엉터리가 된다.
    print(f"중심선 {len(env.center)}점, 코스 길이 약 {env.lap_length:.1f} m\n")

    print("주행 결과")
    run(env, lambda obs: follow_the_gap(obs[:-N_MOTION] * env.lidar_range,
                                        env.beam_angles, env.lidar_range),
        "고전 (Follow the Gap)")

    if model_path(args.model).exists():
        from fly_driver import FlyDriver
        driver = FlyDriver(n_beams=env.n_beams, lidar_range=env.lidar_range,
                           dt=env.dt).load(args.model)
        run(env, driver, "초파리 배선")
    else:
        print(f"  ({args.model} 없음. 학습 먼저)")


if __name__ == "__main__":
    main()
