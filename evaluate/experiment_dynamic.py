"""움직이는 상대 차를 넣고 같은 트랙을 두 번 달린다.

같은 시드, 같은 트랙, 같은 모델. 다른 것은 앞서 달리는 차가 있느냐뿐이다.
짝을 지어 재므로 트랙 폭이나 모양 때문에 생기는 차이가 끼어들지 않는다.

학습에는 상대 차가 한 번도 나온 적이 없다. 배운 적 없는 상황에서
어느 배선이 덜 무너지는지를 본다.

  python experiments/experiment_dynamic.py C_fly871.pt
  python experiments/experiment_dynamic.py B_dense871.pt --dense
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time

import numpy as np

from flyracer.sim.f1tenth_sim import (BODY_RADIUS, PASSAGE_MIN, F1TenthEnv,
                         LIDAR_RANGE)
from flyracer.brain.fly_driver import FlyDriver
from flyracer.sim.moving_obstacles import RIVAL_PACE, MovingObstacleEnv
from flyracer.paths import model as model_path

EVAL_START = 700
EVAL_TRACKS = 45
EPISODE_STEPS = 3000


def drive(driver, env, seed):
    obs, _ = env.reset(seed=seed)
    driver.reset()
    info = {"progress": 0.0}
    closest = np.inf
    for step in range(EPISODE_STEPS):
        obs, _, crashed, timeout, info = env.step(driver.act(obs))
        gap = getattr(env, "rival_gap", lambda: np.inf)()
        closest = min(closest, gap)
        if crashed or timeout:
            break
    return {
        "done": bool(info["progress"] > 0.97),
        "progress": float(info["progress"]),
        "lap_time": float((step + 1) * env.dt),
        # 충돌 판정과 같은 잣대. 상대 차까지의 여유가 먼저 바닥났으면
        # 벽이 아니라 상대 차를 받은 것이다.
        "hit_rival": bool(info["crashed"] and closest < BODY_RADIUS),
        "closest": float(closest) if closest < np.inf else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--beams", type=int, default=541)
    parser.add_argument("--tracks", type=int, default=EVAL_TRACKS)
    parser.add_argument("--rivals", type=int, default=2)
    parser.add_argument("--passage", type=float, default=PASSAGE_MIN)
    parser.add_argument("--pace", type=float, nargs=2, default=list(RIVAL_PACE),
                        help="상대 차 속도 / 그 자리 한계 속도. 두 값 사이에서 뽑는다")
    parser.add_argument("--motion", type=int, default=3,
                        help="관측의 자기 움직임 채널 수. R안 3, T안 6")
    args = parser.parse_args()

    driver = FlyDriver(n_beams=args.beams, lidar_range=LIDAR_RANGE,
                       fly_wiring=not args.dense, motion=args.motion)
    driver.load(args.model)

    print("=" * 70)
    print(f"{Path(args.model).stem}  —  움직이는 상대 차 {args.rivals}대")
    print("=" * 70)
    print(f"시드 {EVAL_START}~{EVAL_START + args.tracks - 1}, "
          f"같은 트랙을 상대 차 없이/있게 두 번씩\n")

    started = time.time()
    rows = []
    for i in range(args.tracks):
        seed = EVAL_START + i
        still = drive(driver, F1TenthEnv(n_beams=args.beams, track_seed=seed,
                                        passage=args.passage,
                                        motion=args.motion), seed)
        moving_env = MovingObstacleEnv(n_beams=args.beams, track_seed=seed,
                                       rivals=args.rivals,
                                       passage=args.passage,
                                       motion=args.motion,
                                       rival_pace=tuple(args.pace))
        moving = drive(driver, moving_env, seed)
        row = {"seed": seed, "width": float(moving_env.track_width),
               "rivals": len(moving_env.rivals),
               "still": still, "moving": moving}
        rows.append(row)

        def mark(r):
            return "완주" if r["done"] else f"{r['progress']:.0%}"
        note = ""
        if moving["hit_rival"]:
            note = "  <- 상대 차를 받음"
        elif still["done"] and not moving["done"]:
            note = "  <- 상대 차 때문에 실패"
        print(f"  시드 {seed}  폭 {row['width']:.2f}m  상대 {row['rivals']}대   "
              f"없이 {mark(still):>5s} {still['lap_time']:5.1f}초   "
              f"있게 {mark(moving):>5s} {moving['lap_time']:5.1f}초{note}")

    with_rivals = [r for r in rows if r["rivals"] > 0]
    still_done = sum(r["still"]["done"] for r in with_rivals)
    moving_done = sum(r["moving"]["done"] for r in with_rivals)
    both = [r for r in with_rivals if r["still"]["done"] and r["moving"]["done"]]
    cost = [r["moving"]["lap_time"] - r["still"]["lap_time"] for r in both]

    summary = {
        "model": str(args.model),
        "wiring": "dense" if args.dense else "fly",
        "tracks": len(rows),
        "tracks_with_rivals": len(with_rivals),
        "still_completed": int(still_done),
        "moving_completed": int(moving_done),
        "lost_to_rivals": int(still_done - moving_done),
        "hit_rival": int(sum(r["moving"]["hit_rival"] for r in rows)),
        "lap_cost": float(np.mean(cost)) if cost else None,
        "seconds": time.time() - started,
        "runs": rows,
    }

    print("-" * 70)
    print(f"상대 차를 놓을 수 있었던 트랙 {len(with_rivals)}/{len(rows)}")
    print(f"상대 차 없이  {still_done}/{len(with_rivals)} 완주")
    print(f"상대 차 있게  {moving_done}/{len(with_rivals)} 완주   "
          f"(상대 차 때문에 잃은 트랙 {summary['lost_to_rivals']}개)")
    # 원래도 실패하던 트랙에서 상대 차를 받는 경우가 있어서 위 숫자와 다르다
    print(f"상대 차를 직접 받은 트랙 {summary['hit_rival']}개")
    if summary["lap_cost"] is not None:
        print(f"둘 다 완주한 트랙에서 랩타임 손해 {summary['lap_cost']:+.2f}초")

    out = model_path(args.model).with_name(
        f"dynamic_{Path(args.model).stem}.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n저장: {out.name}")


if __name__ == "__main__":
    main()
