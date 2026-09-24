"""2026 대회 코스에서만, 노면을 바꿔 가며 여러 번 잰다.

grid_pick 의 대회 주행은 env.reset() 을 시드 없이 불렀다. 그러면 노면
마찰이 0.75~1.15 중 무작위로 뽑힌다. 그래서 똑같은 가중치를 두 번 재면
결과가 갈렸다(high_g99_s0 최고/마지막은 같은 모델인데 한 번은 9.0초 완주,
한 번은 충돌). 대회 코스 완주는 우리가 미리 정한 '필수 조건' 이라
한 번 굴린 숫자로 판정하면 안 된다.

노면을 0.75~1.15 에 고르게 7점 박아 고정하고 그 7번을 다 본다.
미끄러운 바닥(0.75)에서도 도는지가 실차에서 제일 중요하다.

  python experiments/grid_race.py --model grid_high_g99_s0.pt --motion 6
  python experiments/grid_race.py --model R_fly871_v1_keep.pt --motion 3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np

from flyracer.sim.f1tenth_sim import BODY_RADIUS, LIDAR_RANGE
from flyracer.brain.fly_driver import FlyDriver
from flyracer.paths import model as model_path

MAP_YAML = r"C:\Users\woong\Desktop\ifac_roboracer_maps\ifac_roboracer.yaml"
# 학습에 쓰는 마찰 범위(0.75~1.15)에 고르게 7점.
GRIPS = (0.75, 0.82, 0.88, 0.95, 1.02, 1.08, 1.15)
# 노면 말고 다른 무작위(센서 잡음)를 고정하는 시드.
SEED = 1234


def run_once(driver, env, grip):
    env.friction = grip
    # 시드를 고정한다. 안 그러면 센서 잡음까지 매번 달라져 노면 말고 다른
    # 것이 섞인다. track_seed 를 미리 같은 값으로 맞춰 둬야 한다(measure 참고).
    # 안 맞추면 reset 이 트랙을 다시 읽으면서 중심선을 맵에서 새로 뽑고,
    # find_start 가 맞춰 둔 주행 방향이 원래대로 뒤집혀 차가 거꾸로 달린다.
    obs, _ = env.reset(seed=SEED)
    driver.reset()
    worst, info = np.inf, {"progress": 0.0, "lap_time": 0.0, "crashed": True}
    for _ in range(8000):
        obs, _, done, truncated, info = env.step(driver.act(obs))
        worst = min(worst, info["clearance"])
        if done or truncated:
            break
    return {
        "grip": grip,
        "done": bool(not info["crashed"] and info["progress"] > 0.97),
        "lap": float(info["lap_time"]),
        "progress": float(info["progress"]),
        "gap": float(worst - BODY_RADIUS),
    }


def measure(name, beams=541, motion=3):
    from flyracer.sim.track_map import MapEnv, MapTrack
    from flyracer.sim.map_start import find_start

    driver = FlyDriver(n_beams=beams, lidar_range=LIDAR_RANGE,
                       fly_wiring=True, motion=motion)
    driver.load(name)

    track = MapTrack(MAP_YAML)
    env = MapEnv(track, n_beams=beams, motion=motion)
    env.start_pose = find_start(track, env)
    # reset(seed=SEED) 이 트랙을 다시 읽지 않게 미리 맞춰 둔다.
    # find_start 가 뒤집어 놓은 중심선을 지키려는 것이다.
    env.track_seed = SEED

    runs = [run_once(driver, env, g) for g in GRIPS]
    laps = [r["lap"] for r in runs if r["done"]]
    return {
        "model": name,
        "runs": runs,
        "done": sum(r["done"] for r in runs),
        "total": len(GRIPS),
        "lap": float(np.mean(laps)) if laps else 0.0,
        "worst_lap": float(max(laps)) if laps else 0.0,
        "gap": float(min(r["gap"] for r in runs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--beams", type=int, default=541)
    parser.add_argument("--motion", type=int, default=3)
    args = parser.parse_args()

    stem = Path(args.model).stem
    out = model_path(args.model).with_name(f"race_{stem}.json")
    if out.exists():
        print(f"건너뜀 {stem} (이미 잼)", flush=True)
        return
    row = measure(args.model, args.beams, args.motion)
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    bad = [f"{r['grip']:.2f}" for r in row["runs"] if not r["done"]]
    print(f"{stem:<28} 대회 {row['done']}/{row['total']}  "
          f"평균 {row['lap']:.2f}초  최악 {row['worst_lap']:.2f}초  "
          f"최소여유 {row['gap']:+.3f}m"
          + (f"  실패노면 {','.join(bad)}" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
