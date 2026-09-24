"""실제 대회 맵에서 컨트롤러를 돌려 본다.

학습에는 쓰지 않는다. 처음 보는 코스에서 부딪히지 않고 얼마나 빨리 도는지만 본다.

  .\\.venv\\Scripts\\python.exe test_on_map.py "C:/.../ifac_roboracer.yaml"
"""

import sys
from pathlib import Path

# 이 파일은 하위 폴더에 있다. flyracer 패키지를 찾으려면
# 저장소 뿌리를 경로에 넣어 줘야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import numpy as np

from flyracer.drive.controllers import follow_the_gap
from flyracer.paths import model as model_path
from flyracer.brain.fly_driver import N_MOTION
from flyracer.sim.f1tenth_sim import BODY_RADIUS
from flyracer.sim.track_map import MapEnv, MapTrack
from flyracer.sim.map_start import find_start, trace_lap


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
    parser.add_argument("--motion", type=int, default=3,
                        help="관측의 자기 움직임 채널 수. R안 3, T안 6")
    args = parser.parse_args()

    track = MapTrack(args.map_yaml)
    print(track.summary(), "\n")

    env = MapEnv(track, motion=args.motion)
    env.start_pose = find_start(track, env)
    print(f"출발 위치 {np.round(env.start_pose[0], 2)}  "
          f"방향 {np.rad2deg(env.start_pose[1]):.0f}도")

    # 중심선은 MapEnv 가 맵 모양에서 직접 뽑아 이미 갖고 있다.
    # 예전처럼 고전 컨트롤러로 한 바퀴 돌려 덮어쓰면, 그 주행이 실패했을 때
    # 잘린 경로가 중심선이 되어 진행도가 엉터리가 된다.
    print(f"중심선 {len(env.center)}점, 코스 길이 약 {env.lap_length:.1f} m\n")

    print("주행 결과")
    run(env, lambda obs: follow_the_gap(obs[:-env.motion] * env.lidar_range,
                                        env.beam_angles, env.lidar_range),
        "고전 (Follow the Gap)")

    if model_path(args.model).exists():
        from flyracer.brain.fly_driver import FlyDriver
        driver = FlyDriver(n_beams=env.n_beams, lidar_range=env.lidar_range,
                           dt=env.dt, motion=args.motion).load(args.model)
        run(env, driver, "초파리 배선")
    else:
        print(f"  ({args.model} 없음. 학습 먼저)")


if __name__ == "__main__":
    main()
