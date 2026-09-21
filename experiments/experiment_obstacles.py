"""같은 트랙을 정지 장애물만 빼고 두 번 달린다.

장애물은 트랙이 좁으면 아예 안 놓인다. 그래서 '장애물 있는 트랙'과
'없는 트랙'을 그냥 나눠서 세면 장애물 탓인지 트랙 폭 탓인지 알 수 없다.
같은 시드를 장애물만 빼고 다시 달리면 그 구분이 된다.

  python experiments/experiment_obstacles.py C_fly871.pt
  python experiments/experiment_obstacles.py B_dense871.pt --dense
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time

import numpy as np

from f1tenth_sim import F1TenthEnv, LIDAR_RANGE
from fly_driver import FlyDriver
from paths import model as model_path

EVAL_START = 700
EVAL_TRACKS = 45
EPISODE_STEPS = 3000


def drive(driver, env, seed):
    obs, _ = env.reset(seed=seed)
    driver.reset()
    info = {"progress": 0.0}
    speeds = []
    for step in range(EPISODE_STEPS):
        obs, _, crashed, timeout, info = env.step(driver.act(obs))
        speeds.append(env.speed)
        if crashed or timeout:
            break
    return {
        "done": bool(info["progress"] > 0.97),
        "progress": float(info["progress"]),
        "lap_time": float((step + 1) * env.dt),
        "speed": float(np.mean(speeds)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--beams", type=int, default=541)
    parser.add_argument("--tracks", type=int, default=EVAL_TRACKS)
    args = parser.parse_args()

    driver = FlyDriver(n_beams=args.beams, lidar_range=LIDAR_RANGE,
                       fly_wiring=not args.dense)
    driver.load(args.model)

    print("=" * 70)
    print(f"{Path(args.model).stem}  —  정지 장애물 짝 비교")
    print("=" * 70)
    print(f"시드 {EVAL_START}~{EVAL_START + args.tracks - 1}, "
          f"같은 트랙을 장애물 빼고/넣고 두 번씩\n")

    started = time.time()
    rows = []
    for i in range(args.tracks):
        seed = EVAL_START + i
        bare_env = F1TenthEnv(n_beams=args.beams, track_seed=seed,
                              obstacles=False)
        full_env = F1TenthEnv(n_beams=args.beams, track_seed=seed)
        bare = drive(driver, bare_env, seed)
        full = drive(driver, full_env, seed)
        row = {"seed": seed, "width": float(full_env.track_width),
               "obstacles": len(full_env.blobs), "bare": bare, "full": full}
        rows.append(row)
        if row["obstacles"] == 0:
            continue                      # 장애물을 못 놓는 트랙은 볼 것이 없다

        def mark(r):
            return "완주" if r["done"] else f"{r['progress']:.0%}"
        note = ""
        if bare["done"] and not full["done"]:
            note = "  <- 장애물 때문에 실패"
        elif not bare["done"]:
            note = "  <- 장애물과 무관하게 실패"
        print(f"  시드 {seed}  폭 {row['width']:.2f}m  장애물 {row['obstacles']}개   "
              f"빼고 {mark(bare):>5s} {bare['lap_time']:5.1f}초   "
              f"넣고 {mark(full):>5s} {full['lap_time']:5.1f}초{note}")

    tested = [r for r in rows if r["obstacles"] > 0]
    bare_done = sum(r["bare"]["done"] for r in tested)
    full_done = sum(r["full"]["done"] for r in tested)
    both = [r for r in tested if r["bare"]["done"] and r["full"]["done"]]
    cost = [r["full"]["lap_time"] - r["bare"]["lap_time"] for r in both]

    summary = {
        "model": str(args.model),
        "wiring": "dense" if args.dense else "fly",
        "tracks": len(rows),
        "tracks_with_obstacles": len(tested),
        "bare_completed": int(bare_done),
        "full_completed": int(full_done),
        "lost_to_obstacles": int(bare_done - full_done),
        "lap_cost": float(np.mean(cost)) if cost else None,
        "seconds": time.time() - started,
        "runs": rows,
    }

    print("-" * 70)
    print(f"장애물이 놓인 트랙 {len(tested)}/{len(rows)}")
    print(f"장애물 빼고  {bare_done}/{len(tested)} 완주")
    print(f"장애물 넣고  {full_done}/{len(tested)} 완주   "
          f"(장애물 때문에 잃은 트랙 {summary['lost_to_obstacles']}개)")
    if summary["lap_cost"] is not None:
        print(f"둘 다 완주한 트랙에서 랩타임 손해 {summary['lap_cost']:+.2f}초")

    out = model_path(args.model).with_name(
        f"obstacles_{Path(args.model).stem}.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n저장: {out.name}")


if __name__ == "__main__":
    main()
