"""격자 18판 중 '어느 설정으로 본 학습을 걸지' 를 넓은 세트로 골라낸다.

학습 중에 찍히는 '검증 최고' 로는 못 고른다. 검증셋이 12트랙이라 한 칸이
8.3%p 고, 여섯 설정이 94.7~100.0% 안에 다 들어온다. 차이가 0.6트랙이다.
게다가 '최고' 는 8번 측정 중 최대값이라 운 좋은 한 번이 박제된다
(E[max] >= max[E]). R안이 852회차에서 뽑힌 것과 같은 함정이다.

그래서 학습이 끝난 모델을 직접, 넓게 다시 잰다.

  일반 45트랙 x 섭동 2단계 = 90판   완주율과 랩타임
  지그재그 12트랙                    대회를 닮은 좁고 각진 코스
  2026 대회 코스                     학습에 한 번도 안 쓴 진짜 시험

  python experiments/grid_pick.py --model grid_high_g99_s0.pt
  python experiments/grid_pick.py --all          # 18판 순서대로
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
from flyracer.paths import RACE_MAP

# 일반 트랙. experiment_robust_completion 과 같은 구간을 써야 R안 숫자와 비교된다.
EVAL_START = 700
EVAL_TRACKS = 45
# 지그재그 트랙. 학습(narrow 20%)에 쓴 시드와 안 겹치게 멀리 잡는다.
NARROW_START = 800
NARROW_TRACKS = 12
EPISODE_STEPS = 3000
# 한 번만 재면 못 믿는다. 0 과 실차 명령 분해능(1e-3) 두 점만 본다.
# 7점 전부 돌리면 18판 x 7 = 126 세트라 너무 오래 걸린다.
NUDGES = (0.0, 1e-3)

MAP_YAML = str(RACE_MAP)


def drive(driver, seed, beams, nudge, rng, motion, narrow=False):
    env = F1TenthEnv(n_beams=beams, lidar_range=LIDAR_RANGE, track_seed=seed,
                     motion=motion)
    if narrow:
        env.narrow = True
        env._load_track(seed)
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


def drive_map(driver, beams, motion):
    """2026 대회 코스. 학습에 한 번도 안 쓴 코스라 이게 진짜 시험이다."""
    from flyracer.sim.track_map import MapEnv, MapTrack
    from flyracer.sim.map_start import find_start

    track = MapTrack(MAP_YAML)
    env = MapEnv(track, n_beams=beams, motion=motion)
    env.start_pose = find_start(track, env)
    obs, _ = env.reset()
    driver.reset()
    worst, info = np.inf, {"progress": 0.0, "lap_time": 0.0, "crashed": True}
    for _ in range(8000):
        obs, _, done, truncated, info = env.step(driver.act(obs))
        worst = min(worst, info["clearance"])
        if done or truncated:
            break
    return {
        "done": bool(not info["crashed"] and info["progress"] > 0.97),
        "lap": float(info["lap_time"]),
        "gap": float(worst - BODY_RADIUS),
        "progress": float(info["progress"]),
    }


def measure(name, beams=541, motion=6):
    driver = FlyDriver(n_beams=beams, lidar_range=LIDAR_RANGE,
                       fly_wiring=True, motion=motion)
    driver.load(name)

    # 1) 일반 트랙, 섭동 2단계
    seeds = [EVAL_START + i for i in range(EVAL_TRACKS)]
    counts, laps, shakes, gaps = [], [], [], []
    for nudge in NUDGES:
        rng = np.random.default_rng(12345)
        runs = [drive(driver, s, beams, nudge, rng, motion) for s in seeds]
        counts.append(sum(r["done"] for r in runs))
        laps.extend(r["lap"] for r in runs if r["done"])
        shakes.extend(r["shake"] for r in runs)
        gaps.extend(r["gap"] for r in runs)

    # 2) 지그재그 트랙
    rng = np.random.default_rng(999)
    narrow_seeds = [NARROW_START + i for i in range(NARROW_TRACKS)]
    narrow_runs = [drive(driver, s, beams, 0.0, rng, motion, narrow=True)
                   for s in narrow_seeds]
    narrow_done = sum(r["done"] for r in narrow_runs)

    # 3) 2026 대회 코스
    try:
        race = drive_map(driver, beams, motion)
    except Exception as err:                      # 맵이 없어도 나머지는 남긴다
        race = {"done": False, "lap": 0.0, "gap": 0.0, "error": str(err)}

    return {
        "model": name,
        "normal_counts": counts,
        "normal_mean": float(np.mean(counts)),
        "normal_std": float(np.std(counts)),
        "normal_lap": float(np.mean(laps)) if laps else 0.0,
        "shake": float(np.mean(shakes)),
        "gap_p10": float(np.percentile(gaps, 10)),
        "narrow_done": narrow_done,
        "narrow_total": NARROW_TRACKS,
        "race_done": race["done"],
        "race_lap": race["lap"],
        "race_gap": race["gap"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--last", action="store_true",
                        help="'최고' 대신 마지막 정책을 잰다")
    parser.add_argument("--beams", type=int, default=541)
    parser.add_argument("--motion", type=int, default=6)
    args = parser.parse_args()

    if args.all:
        names = [f"grid_{c}_{g}_s{s}.pt"
                 for s in (0, 1, 2)
                 for g in ("g99", "g998")
                 for c in ("none", "mid", "high")]
    else:
        names = [args.model]
    if args.last:
        names = [n.replace(".pt", "_last.pt") for n in names]

    for name in names:
        out = model_path(name).with_name(f"pick_{Path(name).stem}.json")
        if out.exists():
            print(f"건너뜀 {name} (이미 잼)", flush=True)
            continue
        row = measure(name, args.beams, args.motion)
        out.write_text(json.dumps(row, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"{Path(name).stem:<24} 일반 {row['normal_mean']:.1f}±"
              f"{row['normal_std']:.1f}/45  랩 {row['normal_lap']:.1f}초  "
              f"지그재그 {row['narrow_done']}/{row['narrow_total']}  "
              f"대회 {'완주' if row['race_done'] else '실패'} "
              f"{row['race_lap']:.1f}초  떨림 {row['shake']:.3f}", flush=True)


if __name__ == "__main__":
    main()
