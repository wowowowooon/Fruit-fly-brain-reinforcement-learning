"""T안과 Y안(참는 층 얹은 것)을 같은 잣대로 재서 합격/불합격을 가른다.

원인 가리기에서 잣대가 분명해졌다 (대회 코스, 150초 연속 주행).

    장애물0 상대0   17.8바퀴  박음  0회
    장애물2 상대0   17.8바퀴  박음  0회      <- 장애물은 손대면 안 되는 영역
    장애물0 상대1   15.6바퀴  박음  8회      <- 고쳐야 하는 영역
    장애물2 상대1   16.6바퀴  박음 11회

그래서 기준을 이렇게 박는다. 만들기 전에 정한다.

    필수  장애물만 있을 때 박음 0회 유지 (건드리면 안 되는 영역)
    필수  일반 45트랙 44 이상 (T안 45.0)
    목표  상대차 있을 때 박는 횟수 절반 이하
    참고  랩타임. 참는 층이라 느려지는 것은 예상된 대가다

  python evaluate/overtake_check.py
  python evaluate/overtake_check.py --only rival
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np

from flyracer.brain.fly_driver import FlyDriver
from flyracer.drive.overtake import Overtaker
from flyracer.sim.f1tenth_sim import BODY_RADIUS, LIDAR_RANGE, F1TenthEnv
from flyracer.sim.map_hazards import MapHazardEnv
from flyracer.sim.map_start import find_start
from flyracer.sim.track_map import MapTrack

MAP_YAML = r"C:\Users\woong\Desktop\ifac_roboracer_maps\ifac_roboracer.yaml"
MOTION, BEAMS = 6, 541
STEPS = 6000              # 150초 연속 주행
GRIPS = (0.75, 0.88, 1.02, 1.15)


def make(model):
    d = FlyDriver(n_beams=BEAMS, lidar_range=LIDAR_RANGE, fly_wiring=True,
                  motion=MOTION)
    d.load(model)
    return d


def endless(driver, env, over, steps=STEPS):
    """끌 때까지 달리며 박는 횟수를 센다. 박으면 처음으로 돌린다."""
    env.laps, env.max_steps = 10 ** 6, 10 ** 9
    obs, _ = env.reset(seed=0)
    driver.reset()
    if over:
        over.reset()
    laps, crash, last = 0.0, 0, 0.0
    held = 0
    for _ in range(steps):
        action = driver.act(obs)
        if over:
            before = float(action[1])
            action = over.apply(action, obs[:-MOTION] * LIDAR_RANGE,
                                env.beam_angles, env.speed, env.dt)
            held += float(action[1]) < before - 1e-9
        obs, _, done, tr, info = env.step(action)
        laps += max(0.0, info["progress"] - last)
        last = info["progress"]
        if done or tr:
            if env.crashed:
                crash += 1
            obs, _ = env.reset(seed=0)
            driver.reset()
            if over:
                over.reset()
            last = 0.0
    return {"laps": laps, "crash": crash, "held": held / max(steps, 1)}


def hazard_env(track, obstacles, rivals, pace, seed=1):
    env = MapHazardEnv(track, hazard_seed=seed, n_obstacles=obstacles,
                       rivals=rivals, rival_pace=pace, teacher_rivals=True,
                       spawn_opposite=True, n_beams=BEAMS,
                       lidar_range=LIDAR_RANGE, motion=MOTION)
    env.start_pose = find_start(track, env)
    env.rebuild_hazards()
    return env


def race(driver, track, over):
    """대회 코스 노면 4종, 한 바퀴씩. 여유가 지켜지는지 본다."""
    rows = []
    for grip in GRIPS:
        env = MapHazardEnv(track, hazard_seed=1, n_obstacles=0, rivals=0,
                           n_beams=BEAMS, lidar_range=LIDAR_RANGE,
                           motion=MOTION)
        env.start_pose = find_start(track, env)
        env.rebuild_hazards()
        # reset(seed=) 이 트랙을 다시 읽으면 find_start 가 맞춰 둔 주행
        # 방향이 되돌아가 차가 거꾸로 달린다. 미리 맞춰 둬야 한다.
        env.track_seed = 1234
        env.friction = grip
        obs, _ = env.reset(seed=1234)
        driver.reset()
        if over:
            over.reset()
        gaps = []
        info = {"progress": 0.0, "crashed": True}
        for step in range(8000):
            action = driver.act(obs)
            if over:
                action = over.apply(action, obs[:-MOTION] * LIDAR_RANGE,
                                    env.beam_angles, env.speed, env.dt)
            obs, _, done, tr, info = env.step(action)
            gaps.append(info["clearance"] - BODY_RADIUS)
            if done or tr:
                break
        rows.append({"done": not info["crashed"] and info["progress"] > 0.97,
                     "lap": (step + 1) * env.dt,
                     "gap": float(np.min(gaps))})
    laps = [r["lap"] for r in rows if r["done"]]
    return {"done": sum(r["done"] for r in rows), "total": len(GRIPS),
            "lap": float(np.mean(laps)) if laps else 0.0,
            "gap": min(r["gap"] for r in rows)}


def normal(driver, over, seeds=range(700, 745)):
    rows = []
    for s in seeds:
        env = F1TenthEnv(n_beams=BEAMS, lidar_range=LIDAR_RANGE,
                         track_seed=s, motion=MOTION)
        obs, _ = env.reset(seed=s)
        driver.reset()
        if over:
            over.reset()
        info = {"progress": 0.0, "crashed": True}
        for step in range(3000):
            action = driver.act(obs)
            if over:
                action = over.apply(action, obs[:-MOTION] * LIDAR_RANGE,
                                    env.beam_angles, env.speed, env.dt)
            obs, _, done, tr, info = env.step(action)
            if done or tr:
                break
        rows.append({"done": not info["crashed"] and info["progress"] > 0.97,
                     "lap": (step + 1) * env.dt})
    laps = [r["lap"] for r in rows if r["done"]]
    return {"done": sum(r["done"] for r in rows), "total": len(list(seeds)),
            "lap": float(np.mean(laps)) if laps else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="T_fly871_final.pt")
    parser.add_argument("--only", choices=("hazard", "race", "normal"),
                        default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    track = MapTrack(MAP_YAML)
    result = {}

    if args.only in (None, "hazard"):
        print("대회 코스 150초 연속 주행 — 박는 횟수")
        print(f"{'조건':<24}{'T안':>18}{'Y안':>18}{'참은 비율':>10}")
        print("-" * 72)
        cases = (("장애물2 상대0", 2, 0, (0.85, 1.0)),
                 ("상대1 pace0.85", 0, 1, (0.85, 1.0)),
                 ("상대1 pace0.40", 0, 1, (0.40, 0.55)),
                 ("장애물2 상대1", 2, 1, (0.85, 1.0)))
        for name, ob, rv, pace in cases:
            t = endless(make(args.model), hazard_env(track, ob, rv, pace), None)
            y = endless(make(args.model), hazard_env(track, ob, rv, pace),
                        Overtaker(lidar_range=LIDAR_RANGE))
            result[name] = {"T": t, "Y": y}
            print(f"{name:<24}"
                  f"{t['laps']:>6.1f}바퀴 {t['crash']:>3d}회"
                  f"{y['laps']:>8.1f}바퀴 {y['crash']:>3d}회"
                  f"{y['held']:>9.0%}", flush=True)

    if args.only in (None, "race"):
        print("\n대회 코스 한 바퀴 (노면 4종, 방해물 없음)")
        t = race(make(args.model), track, None)
        y = race(make(args.model), track, Overtaker(lidar_range=LIDAR_RANGE))
        result["race"] = {"T": t, "Y": y}
        for label, r in (("T안", t), ("Y안", y)):
            print(f"  {label}  {r['done']}/{r['total']}  {r['lap']:.2f}초  "
                  f"여유 {r['gap']:+.3f}m", flush=True)

    if args.only in (None, "normal"):
        print("\n일반 45트랙")
        t = normal(make(args.model), None)
        y = normal(make(args.model), Overtaker(lidar_range=LIDAR_RANGE))
        result["normal"] = {"T": t, "Y": y}
        for label, r in (("T안", t), ("Y안", y)):
            print(f"  {label}  {r['done']}/{r['total']}  {r['lap']:.2f}초",
                  flush=True)

    print("\n" + "=" * 60)
    ok = True
    if "장애물2 상대0" in result:
        r = result["장애물2 상대0"]["Y"]
        good = r["crash"] == 0
        ok &= good
        print(f"필수  장애물만: 박음 {r['crash']}회 "
              f"{'통과' if good else '탈락 — 멀쩡한 걸 망가뜨렸다'}")
    if "normal" in result:
        r = result["normal"]["Y"]
        good = r["done"] >= 44
        ok &= good
        print(f"필수  일반 {r['done']}/45 (44 이상) "
              f"{'통과' if good else '탈락'}")
    for key in ("상대1 pace0.85", "상대1 pace0.40", "장애물2 상대1"):
        if key in result:
            t, y = result[key]["T"], result[key]["Y"]
            cut = y["crash"] <= t["crash"] / 2
            print(f"목표  {key}: {t['crash']} -> {y['crash']}회 "
                  f"{'달성' if cut else '미달'}")
    print(f"\n=> 참는 층 {'채택' if ok else '보류 — 필수 조건 미달'}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False,
                                             indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
