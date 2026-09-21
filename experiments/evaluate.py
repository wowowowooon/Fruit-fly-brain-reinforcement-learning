"""모델을 같은 조건에서 재고 결과를 표로 남긴다.

A/B/C 를 비교하려면 재는 방법이 완전히 같아야 한다. 시드가 하나라도
겹치거나 노이즈 설정이 다르면 숫자를 나란히 놓을 수 없다.

  python evaluate.py models/C_fly871.pt
  python evaluate.py models/C_fly871.pt --dense     # B안을 잴 때
  python evaluate.py models/C_fly871.pt --map "...\\ifac_roboracer.yaml"

결과는 모델 옆에 .json 으로 남는다.
"""

import sys
from pathlib import Path

# 상위 폴더에 있는 공통 모듈을 찾을 수 있게 한다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from pathlib import Path

import numpy as np

from f1tenth_sim import PASSAGE_MIN, F1TenthEnv, LIDAR_RANGE
from fly_driver import FlyDriver
from paths import model as model_path
from vec_env import TRACK_POOL

# 학습에 쓴 트랙(0~399)과 절대 겹치지 않게 한참 뒤에서 뽑는다.
# 처음 보는 코스에서 달리는 것이 목표이므로 여기서 겹치면 의미가 없다.
EVAL_START = 700
EVAL_TRACKS = 45
EPISODE_STEPS = 3000


def _live_links(driver):
    """순환 블록에서 실제로 살아 있는 연결 수."""
    cell = driver.model.rnn.rnn_cell.layer_0
    mask = cell.sparsity_mask.data
    units = driver.wiring.units
    return int((mask[:, mask.shape[1] - units:] != 0).sum())


def run_track(driver, seed, beams, lidar_range, passage=PASSAGE_MIN):
    env = F1TenthEnv(n_beams=beams, lidar_range=lidar_range, track_seed=seed,
                     passage=passage)
    obs, _ = env.reset(seed=seed)
    driver.reset()
    info = {"progress": 0.0, "crashed": False}
    speeds = []
    for step in range(EPISODE_STEPS):
        obs, reward, crashed, timeout, info = env.step(driver.act(obs))
        speeds.append(env.speed)
        if crashed or timeout:
            break
    return {
        "seed": seed,
        "done": bool(info["progress"] > 0.97),
        "progress": float(info["progress"]),
        "lap_time": float((step + 1) * env.dt),
        "speed": float(np.mean(speeds)),
        "width": float(env.track_width),
        "obstacles": int(len(env.blobs)),
    }


def on_map(driver, path, beams, lidar_range):
    from test_on_map import find_start, trace_lap
    from track_map import MapEnv, MapTrack

    track = MapTrack(path)
    env = MapEnv(track, n_beams=beams, lidar_range=lidar_range)
    env.start_pose = find_start(track, env)
    obs, _ = env.reset()
    driver.reset()
    info = {"progress": 0.0}
    for step in range(EPISODE_STEPS):
        obs, reward, crashed, timeout, info = env.step(driver.act(obs))
        if crashed or timeout:
            break
    return {
        "name": track.name,
        "done": bool(info["progress"] > 0.97),
        "progress": float(info["progress"]),
        "lap_time": float((step + 1) * env.dt),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--dense", action="store_true",
                        help="전부 연결된 비교군 모델(B안)을 잴 때")
    parser.add_argument("--beams", type=int, default=541)
    parser.add_argument("--lidar-range", type=float, default=LIDAR_RANGE)
    parser.add_argument("--tracks", type=int, default=EVAL_TRACKS)
    parser.add_argument("--map", default=None)
    parser.add_argument("--passage", type=float, default=PASSAGE_MIN,
                        help="장애물 옆 통로 [m]. 레이싱용 모델은 1.10 으로 "
                             "학습하므로 같은 조건에서 재야 한다")
    args = parser.parse_args()

    driver = FlyDriver(n_beams=args.beams, lidar_range=args.lidar_range,
                       fly_wiring=not args.dense)
    driver.load(args.model)

    print("=" * 66)
    print(f"{Path(args.model).stem}")
    print("=" * 66)
    print(driver.summary())
    print(f"\n처음 보는 트랙 {args.tracks}개 "
          f"(시드 {EVAL_START}~{EVAL_START + args.tracks - 1}, "
          f"학습에 쓴 0~{TRACK_POOL - 1} 과 안 겹침)")
    print("-" * 66)

    started = time.time()
    rows = []
    for i in range(args.tracks):
        row = run_track(driver, EVAL_START + i, args.beams, args.lidar_range,
                        args.passage)
        rows.append(row)
        mark = "완주" if row["done"] else f"{row['progress']:.0%}"
        print(f"  시드 {row['seed']}  폭 {row['width']:.2f}m  "
              f"장애물 {row['obstacles']}개  {mark:>5s}  "
              f"{row['lap_time']:5.1f}초  평균 {row['speed']:.1f} m/s")

    done = [r for r in rows if r["done"]]
    narrow = [r for r in rows if r["width"] <= 2.0]
    narrow_done = [r for r in narrow if r["done"]]
    # 장애물은 트랙 만들 때 시드로 정해진다. 회피 실력과 순수 주행 실력을
    # 갈라 보려면 있는 트랙과 없는 트랙을 따로 세야 한다.
    with_obs = [r for r in rows if r["obstacles"] > 0]
    without_obs = [r for r in rows if r["obstacles"] == 0]

    summary = {
        "model": str(args.model),
        "wiring": "dense" if args.dense else "fly",
        "passage": float(args.passage),
        "neurons": int(driver.wiring.units),
        # wiring.adjacency_matrix 는 초파리 배선이라 B안(밀집)에서는 틀린 값이
        # 나온다. 모델이 실제로 쓰는 마스크를 센다.
        "links": int(_live_links(driver)),
        "params_stored": sum(p.numel() for p in driver.model.parameters()),
        "params_used": driver.effective_params(),
        "tracks": len(rows),
        "completed": len(done),
        "completion": len(done) / max(len(rows), 1),
        "lap_time": float(np.mean([r["lap_time"] for r in done]))
        if done else None,
        "speed": float(np.mean([r["speed"] for r in rows])),
        "narrow_tracks": len(narrow),
        "narrow_completed": len(narrow_done),
        "obstacle_tracks": len(with_obs),
        "obstacle_completed": sum(r["done"] for r in with_obs),
        "clear_tracks": len(without_obs),
        "clear_completed": sum(r["done"] for r in without_obs),
        "seconds": time.time() - started,
        "runs": rows,
    }

    if args.map:
        summary["map"] = on_map(driver, args.map, args.beams, args.lidar_range)

    print("-" * 66)
    print(f"완주      {summary['completed']}/{summary['tracks']} "
          f"({summary['completion']:.0%})")
    if summary["lap_time"]:
        print(f"평균 랩   {summary['lap_time']:.1f}초")
    print(f"평균 속도 {summary['speed']:.1f} m/s")
    print(f"좁은 트랙 {summary['narrow_completed']}/{summary['narrow_tracks']} "
          f"(폭 2.0m 이하)")
    print(f"장애물 있음 {summary['obstacle_completed']}/{summary['obstacle_tracks']}"
          f"   없음 {summary['clear_completed']}/{summary['clear_tracks']}")
    print(f"실제 사용 파라미터 {summary['params_used']:,}개")
    if args.map:
        result = summary["map"]
        verdict = "완주" if result["done"] else f"{result['progress']:.0%} 에서 충돌"
        print(f"\n2026 대회 코스  {verdict}  {result['lap_time']:.1f}초")

    out = model_path(args.model).with_suffix(".json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n저장: {out.name}")


if __name__ == "__main__":
    main()
