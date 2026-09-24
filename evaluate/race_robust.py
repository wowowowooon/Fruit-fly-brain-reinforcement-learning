"""2026 대회 코스를 노면 7종 x 섭동 3단계로 잰다. 총 21판.

대회 코스 완주는 우리가 미리 정한 필수 조건이다. 그런데 노면만 바꿔
7번 재는 것으로도 아직 부족하다는 게 드러났다. 본 학습에서 저장된
후보들을 보면

  it1400  7/7 완주, 최소여유 +0.226m
  it1950  0/7 완주, 최소여유 -0.053m
  it2000  4/7 완주, 최소여유 -0.046m
  it3000  7/7 완주, 최소여유 +0.228m

같은 학습 안에서 500회차 사이에 7/7 과 0/7 을 오갔다. 그러면 마지막에
7/7 이 나온 것도 '그 순간에 그랬다' 는 뜻일 수 있다. 명령에 아주 작은
섭동을 줘서, 여유가 진짜로 있는지 아니면 아슬아슬하게 붙어 있는지 본다.

  python experiments/race_robust.py --model T_cand_it3000.pt --motion 6
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
GRIPS = (0.75, 0.82, 0.88, 0.95, 1.02, 1.08, 1.15)
# 0 = 그대로, 1e-6 = 수치 오차 수준, 1e-3 = 실차 명령 분해능
NUDGES = (0.0, 1e-6, 1e-3)
SEED = 1234


def run_once(driver, env, grip, nudge, rng):
    env.friction = grip
    obs, _ = env.reset(seed=SEED)
    driver.reset()
    worst, info = np.inf, {"progress": 0.0, "lap_time": 0.0, "crashed": True}
    for _ in range(8000):
        action = driver.act(obs)
        if nudge:
            action = action + rng.normal(0.0, nudge, 2)
        obs, _, done, truncated, info = env.step(action)
        worst = min(worst, info["clearance"])
        if done or truncated:
            break
    return {
        "grip": grip, "nudge": nudge,
        "done": bool(not info["crashed"] and info["progress"] > 0.97),
        "lap": float(info["lap_time"]),
        "gap": float(worst - BODY_RADIUS),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--beams", type=int, default=541)
    parser.add_argument("--motion", type=int, default=6)
    args = parser.parse_args()

    from flyracer.sim.track_map import MapEnv, MapTrack
    from flyracer.sim.map_start import find_start

    stem = Path(args.model).stem
    out = model_path(args.model).with_name(f"racerobust_{stem}.json")
    if out.exists():
        print(f"건너뜀 {stem}", flush=True)
        return

    driver = FlyDriver(n_beams=args.beams, lidar_range=LIDAR_RANGE,
                       fly_wiring=True, motion=args.motion)
    driver.load(args.model)
    track = MapTrack(MAP_YAML)
    env = MapEnv(track, n_beams=args.beams, motion=args.motion)
    env.start_pose = find_start(track, env)
    # reset 이 트랙을 다시 읽어 중심선 방향을 되돌리지 못하게 막는다
    env.track_seed = SEED

    runs = []
    for nudge in NUDGES:
        rng = np.random.default_rng(12345)
        runs.extend(run_once(driver, env, g, nudge, rng) for g in GRIPS)

    laps = [r["lap"] for r in runs if r["done"]]
    row = {
        "model": args.model, "runs": runs,
        "done": sum(r["done"] for r in runs), "total": len(runs),
        "lap": float(np.mean(laps)) if laps else 0.0,
        "worst_lap": float(max(laps)) if laps else 0.0,
        "gap": float(min(r["gap"] for r in runs)),
    }
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    per = {}
    for r in runs:
        per.setdefault(r["nudge"], []).append(r["done"])
    txt = "  ".join(f"섭동{('없음' if n == 0 else f'{n:g}')} {sum(v)}/{len(v)}"
                    for n, v in per.items())
    print(f"{stem:<22} 대회 {row['done']}/{row['total']}  평균 {row['lap']:.2f}초  "
          f"최소여유 {row['gap']:+.3f}m   {txt}", flush=True)


if __name__ == "__main__":
    main()
