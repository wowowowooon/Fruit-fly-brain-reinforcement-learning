"""완주율을 정직하게 잰다 — 한 번 돌려서 나온 숫자는 못 믿는다.

R안을 한 번 돌리면 44/45 가 나온다. 그런데 행동에 부동소수점 1비트
(1e-15) 만큼 섭동을 주면 42/45 가 된다. 7회 평균은 42.6+-1.1 이다.

뒤집히는 트랙은 항상 최소 여유가 6cm 미만인 것들이다. 그 순간 횡가속은
0.06~0.8G 뿐이라 미끄러진 것이 아니라 스스로 벽으로 파고든 것이다.
여유가 곧 위험이고, 한 번 돌린 숫자는 그 위험을 안 보여준다.

  python experiments/experiment_robust_completion.py R_fly871.pt
  python experiments/experiment_robust_completion.py T_fly871.pt --motion 6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np

from flyracer.sim.f1tenth_sim import BODY_RADIUS, F1TenthEnv, LIDAR_RANGE
from flyracer.brain.fly_driver import FlyDriver
from flyracer.paths import model as model_path

EVAL_START = 700
EVAL_TRACKS = 45
EPISODE_STEPS = 3000
# 실차 명령 분해능(1e-3)까지 훑는다. 1e-15 는 부동소수점 한 칸이다.
NUDGES = (0.0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3)


def drive(driver, seed, beams, nudge, rng, motion):
    env = F1TenthEnv(n_beams=beams, lidar_range=LIDAR_RANGE, track_seed=seed,
                     motion=motion)
    obs, _ = env.reset(seed=seed)
    driver.reset()
    info = {"progress": 0.0}
    gaps, shakes, speeds = [], [], []
    previous = None
    for step in range(EPISODE_STEPS):
        action = driver.act(obs)
        if nudge:
            action = action + rng.normal(0.0, nudge, 2)
        if previous is not None:
            shakes.append(abs(float(action[0]) - previous))
        previous = float(action[0])
        obs, _, crashed, timeout, info = env.step(action)
        gaps.append(info["clearance"] - BODY_RADIUS)
        speeds.append(env.speed)
        if crashed or timeout:
            break
    return {
        "done": bool(info["progress"] > 0.97),
        "lap": float((step + 1) * env.dt),
        "gap": float(np.min(gaps)),
        "shake": float(np.mean(shakes)) if shakes else 0.0,
        "speed": float(np.mean(speeds)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--motion", type=int, default=3)
    parser.add_argument("--beams", type=int, default=541)
    parser.add_argument("--tracks", type=int, default=EVAL_TRACKS)
    args = parser.parse_args()

    driver = FlyDriver(n_beams=args.beams, lidar_range=LIDAR_RANGE,
                       fly_wiring=not args.dense, motion=args.motion)
    driver.load(args.model)
    seeds = [EVAL_START + i for i in range(args.tracks)]

    print("=" * 70)
    print(f"{Path(args.model).stem}  —  섭동을 줘 가며 {len(NUDGES)}회")
    print("=" * 70)

    counts, laps, shakes, flips = [], [], [], {}
    gap_by_seed = {}
    for nudge in NUDGES:
        rng = np.random.default_rng(12345)
        runs = [drive(driver, s, args.beams, nudge, rng, args.motion)
                for s in seeds]
        done = sum(r["done"] for r in runs)
        counts.append(done)
        laps.extend(r["lap"] for r in runs if r["done"])
        shakes.extend(r["shake"] for r in runs)
        for s, r in zip(seeds, runs):
            gap_by_seed.setdefault(s, []).append(r["gap"])
            if not r["done"]:
                flips[s] = flips.get(s, 0) + 1
        tag = "없음" if nudge == 0 else f"{nudge:g}"
        print(f"  섭동 {tag:>7s}   완주 {done:2d}/{len(seeds)}", flush=True)

    mean, std = float(np.mean(counts)), float(np.std(counts))
    print("-" * 70)
    print(f"완주 {mean:.1f} ± {std:.1f} / {len(seeds)}   "
          f"(한 번만 재면 {counts[0]})")
    print(f"평균 랩 {np.mean(laps):.2f}초   떨림 {np.mean(shakes):.4f}")
    if flips:
        print("\n뒤집히는 트랙 (섭동에 따라 결과가 바뀜)")
        for s, n in sorted(flips.items(), key=lambda kv: -kv[1]):
            print(f"  시드 {s}  {n}/{len(NUDGES)}회 실패   "
                  f"최소 여유 {min(gap_by_seed[s]):+.3f}m")

    out = model_path(args.model).with_name(
        f"robust_completion_{Path(args.model).stem}.json")
    out.write_text(json.dumps({
        "model": str(args.model), "counts": counts,
        "mean": mean, "std": std,
        "lap": float(np.mean(laps)), "shake": float(np.mean(shakes)),
        "flips": {str(k): v for k, v in flips.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out.name}")


if __name__ == "__main__":
    main()
