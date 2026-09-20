"""실제 대회 맵에서 컨트롤러를 돌려 본다.

학습에는 쓰지 않는다. 처음 보는 코스에서 부딪히지 않고 얼마나 빨리 도는지만 본다.

  .\\.venv\\Scripts\\python.exe test_on_map.py "C:/.../ifac_roboracer.yaml"
"""

import argparse
from pathlib import Path

import numpy as np

from controllers import follow_the_gap
from f1tenth_sim import BODY_RADIUS
from track_map import MapEnv, MapTrack


def _signed_area(path):
    """폐곡선의 부호 있는 면적. 양수면 반시계."""
    nxt = np.roll(path, -1, axis=0)
    return 0.5 * float(np.sum(path[:, 0] * nxt[:, 1] - nxt[:, 0] * path[:, 1]))


def find_start(track, env, counterclockwise=True):
    """가장 넓은 곳에 차를 놓고, 코스가 뻗은 방향을 찾는다.

    코스는 양쪽으로 다 뚫려 있어서 앞만 보고 고르면 방향이 반반이다.
    한 바퀴 돌아 보고 도는 방향이 원하는 쪽인지 확인한 뒤 필요하면 뒤집는다.
    대회 주행 방향을 맞춰야 학습한 모델을 그대로 올릴 수 있다.
    """
    position, _ = track.widest_point()
    best = None
    for yaw in np.linspace(-np.pi, np.pi, 72, endpoint=False):
        env.pos, env.yaw = position.copy(), float(yaw)
        scan = env._scan()
        ahead = scan[np.abs(env.beam_angles) <= np.deg2rad(10)].min()
        if best is None or ahead > best[0]:
            best = (ahead, yaw)
    heading = float(best[1])

    # 그 방향으로 한 바퀴 돌려 보고, 도는 쪽이 반대면 뒤집는다
    env.start_pose = (position, heading)
    path, closed = trace_lap(env)
    if closed and len(path) > 20:
        turning = _signed_area(path) > 0          # 양수 = 반시계
        if turning != counterclockwise:
            heading = float((heading + np.pi + np.pi) % (2 * np.pi) - np.pi)
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
    parser.add_argument("--model", default="fly_model.pt")
    args = parser.parse_args()

    track = MapTrack(args.map_yaml)
    print(track.summary(), "\n")

    env = MapEnv(track)
    env.start_pose = find_start(track, env)
    print(f"출발 위치 {np.round(env.start_pose[0], 2)}  "
          f"방향 {np.rad2deg(env.start_pose[1]):.0f}도")

    path, closed = trace_lap(env)
    print(f"중심선 추적: {len(path)}점, 한 바퀴 {'완성' if closed else '미완성'}")
    if not closed:
        print("  (고전 컨트롤러가 한 바퀴를 못 돌았다. 진행도는 참고용)")
    env.set_centerline(path)
    print(f"코스 길이 약 {env.lap_length:.1f} m\n")

    print("주행 결과")
    run(env, lambda obs: follow_the_gap(obs[:-N_MOTION] * env.lidar_range,
                                        env.beam_angles, env.lidar_range),
        "고전 (Follow the Gap)")

    if Path(args.model).exists():
        from fly_driver import N_MOTION, FlyDriver
        driver = FlyDriver(n_beams=env.n_beams, lidar_range=env.lidar_range,
                           dt=env.dt).load(args.model)
        run(env, driver, "초파리 배선")
    else:
        print(f"  ({args.model} 없음. 학습 먼저)")


if __name__ == "__main__":
    main()
