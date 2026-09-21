"""센서를 망가뜨려 가며 얼마나 버티는지 잰다.

초파리 회로를 쓰는 이유 중 하나가 '망가진 센서에도 덜 무너진다' 였다.
주장만 하지 말고 재야 한다. 같은 손상을 같은 트랙에 걸어 여러 모델을
나란히 놓을 수 있게 만든다.

  python experiment_faults.py C_fly871.pt
  python experiment_faults.py B_dense871.pt --dense
  python experiment_faults.py C_fly871.pt --imu      # IMU 쪽만

결과는 outputs/fault_<모델>.json 에 남는다.
"""

import sys
from pathlib import Path

# 상위 폴더에 있는 공통 모듈을 찾을 수 있게 한다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time

import numpy as np

from f1tenth_sim import F1TenthEnv, LIDAR_RANGE
from fly_driver import FlyDriver
from paths import OUTPUTS
from sensor_noise import NoisyLidar, SensorFault, fault_sweep

# 평가와 같은 시드를 쓴다. 학습에 쓴 0~399 와 겹치지 않는다.
EVAL_START = 700
EPISODE_STEPS = 3000


def imu_sweep():
    """IMU 만 망가뜨리는 경우. 자이로 영점과 진동을 키운다."""
    return {
        "자이로 잡음": [
            SensorFault(f"{g:.2f} rad/s", gyro=g)
            for g in (0.02, 0.10, 0.30, 0.60, 1.00)
        ],
        "자이로 영점 틀어짐": [
            SensorFault(f"{b:.2f} rad/s", bias=b)
            for b in (0.03, 0.10, 0.30, 0.60, 1.00)
        ],
        "가속도계 진동": [
            SensorFault(f"{a:.1f} m/s^2", accel=a)
            for a in (0.3, 1.0, 3.0, 6.0, 10.0)
        ],
    }


def make_map_env(path, beams, lidar_range):
    """2026 대회 코스. 실험마다 다시 만들면 느려서 한 번만 만든다."""
    from test_on_map import find_start
    from track_map import MapEnv, MapTrack

    track = MapTrack(path)
    env = MapEnv(track, n_beams=beams, lidar_range=lidar_range)
    env.start_pose = find_start(track, env)
    return env


def drive_once(driver, env, fault, lidar_range, seed):
    obs, _ = env.reset()
    driver.reset()
    broken = NoisyLidar(env.beam_angles, lidar_range, fault,
                        np.random.default_rng(seed))
    broken.reset()
    info = {"progress": 0.0}
    for _ in range(EPISODE_STEPS):
        obs, reward, finished, timeout, info = env.step(
            driver.act(broken(obs)))
        if finished or timeout:
            break
    return float(info["progress"])


def run(driver, fault, seeds, beams, lidar_range, map_env=None):
    """같은 손상을 여러 트랙에 걸어 완주율을 낸다.

    map_env 를 주면 대회 코스 하나만 여러 번(손상 씨앗을 바꿔 가며) 돈다.
    코스가 하나뿐이라 한 번만 재면 운에 좌우된다.
    """
    progress = []
    for seed in seeds:
        if map_env is not None:
            progress.append(drive_once(driver, map_env, fault,
                                       lidar_range, seed))
        else:
            env = F1TenthEnv(n_beams=beams, lidar_range=lidar_range,
                             track_seed=seed)
            env.reset(seed=seed)
            progress.append(drive_once(driver, env, fault,
                                       lidar_range, seed))
    done = sum(p > 0.97 for p in progress)
    return done / len(progress), float(np.mean(progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--imu", action="store_true",
                        help="라이다 대신 IMU 손상만 본다")
    parser.add_argument("--tracks", type=int, default=12)
    parser.add_argument("--beams", type=int, default=541)
    parser.add_argument("--lidar-range", type=float, default=LIDAR_RANGE)
    parser.add_argument("--map", default=None,
                        help="이 코스 하나에서만 잰다 (2026 대회 코스)")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="건너뛸 손상 항목. 예: --skip 한쪽 시야 차단")
    args = parser.parse_args()

    driver = FlyDriver(n_beams=args.beams, lidar_range=args.lidar_range,
                       fly_wiring=not args.dense)
    driver.load(args.model)
    seeds = list(range(EVAL_START, EVAL_START + args.tracks))
    sweep = imu_sweep() if args.imu else fault_sweep(args.lidar_range)

    map_env = None
    if args.map:
        map_env = make_map_env(args.map, args.beams, args.lidar_range)
        where = f"2026 대회 코스 ({map_env.track.name}) x {args.tracks}회"
    else:
        where = f"트랙 {args.tracks}개 (시드 {seeds[0]}~{seeds[-1]})"

    print("=" * 64)
    print(f"{args.model}   {'IMU' if args.imu else '라이다'} 손상 견딤")
    print(where)
    print("=" * 64)

    clean, _ = run(driver, SensorFault("깨끗"), seeds, args.beams,
                   args.lidar_range, map_env)
    print(f"\n손상 없음: 완주 {clean:.0%}\n")

    started = time.time()
    results = {"model": args.model, "clean": clean,
               "kind": "imu" if args.imu else "lidar",
               "where": "map" if args.map else "seeds", "groups": {}}
    for group, faults in sweep.items():
        if group == "없음" or any(s in group for s in args.skip):
            continue
        print(f"{group}")
        rows = []
        for fault in faults:
            rate, mean = run(driver, fault, seeds, args.beams,
                             args.lidar_range, map_env)
            keep = rate / clean if clean else 0.0
            bar = "#" * int(round(keep * 20))
            print(f"  {fault.name:>12s}  완주 {rate:>4.0%}  "
                  f"유지 {keep:>4.0%}  {bar}")
            rows.append({"fault": fault.name, "completion": rate,
                         "kept": keep, "progress": mean})
        results["groups"][group] = rows
        print()

    tag = ("map" if args.map else "seeds") + ("_imu" if args.imu else "")
    out = OUTPUTS / f"fault_{args.model.replace('.pt', '')}_{tag}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"{time.time() - started:.0f}초 걸림.  저장: {out.name}")


if __name__ == "__main__":
    main()
