"""실차에서 달라질 것들에 얼마나 버티나.

시뮬에서 잘 달리는 것과 실제 대회장에서 잘 달리는 것은 다르다. 바닥이
미끄럽고, 젯슨이 느려 명령이 늦게 먹고, 출발 자세가 라인 위가 아니다.
이런 것들이 바뀌었을 때 어느 배선이 덜 무너지는지를 본다.

네 가지를 잰다.
  노면    타이어가 버티는 힘을 55%~115% 로 바꾼다. 학습은 75~115% 에서 했다.
  지연    명령이 먹기까지의 시간을 50ms(학습값)에서 150ms 까지 늘린다.
  자세    출발할 때 차를 옆으로 밀고 비스듬히 돌려 놓는다. 되돌아오는가.
  부드러움 조향을 얼마나 급하게 꺾는가, 벽에 얼마나 붙는가.

  python experiments/experiment_robustness.py C_fly871.pt
  python experiments/experiment_robustness.py B_dense871.pt --dense
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time

import numpy as np

from f1tenth_sim import BODY_RADIUS, F1TenthEnv, LIDAR_RANGE
from fly_driver import FlyDriver
from paths import model as model_path

EVAL_START = 700
EVAL_TRACKS = 20
EPISODE_STEPS = 3000

# 학습은 마찰 75~115% 에서 했다. 그 아래는 배운 적 없는 바닥이다.
FRICTIONS = (1.15, 1.00, 0.85, 0.70, 0.55)
# 학습은 50ms 에서 했다. 젯슨이 밀리거나 통신이 늦으면 더 걸린다.
LATENCIES = (0.05, 0.075, 0.100, 0.150)
# (옆으로 민 거리 m, 돌려 놓은 각도 도)
POSES = ((0.0, 0.0), (0.0, 25.0), (0.30, 0.0), (0.30, 25.0), (0.30, -25.0))


def drive(driver, env, seed, offset=0.0, yaw_error=0.0):
    """한 판 달리고 기록한다. 출발 자세를 흐트러뜨릴 수 있다."""
    obs, _ = env.reset(seed=seed)
    if offset or yaw_error:
        index = env._progress()
        nxt = (index + 1) % len(env.center)
        tangent = env.center[nxt] - env.center[index]
        tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
        normal = np.array([-tangent[1], tangent[0]])
        env.pos = env.pos + normal * offset
        env.yaw += np.deg2rad(yaw_error)
        if env._clearance() < BODY_RADIUS:
            return None              # 밀었더니 벽 속이다. 이 트랙은 못 쓴다.
        obs = env._observe(env._scan())

    driver.reset()
    info = {"progress": 0.0, "crashed": False}
    speeds, shakes, gaps = [], [], []
    previous = None
    for step in range(EPISODE_STEPS):
        action = driver.act(obs)
        if previous is not None:
            shakes.append(abs(float(action[0]) - previous))
        previous = float(action[0])
        obs, _, crashed, timeout, info = env.step(action)
        speeds.append(env.speed)
        gaps.append(info["clearance"] - BODY_RADIUS)
        if crashed or timeout:
            break
    return {
        "done": bool(info["progress"] > 0.97),
        "progress": float(info["progress"]),
        "lap_time": float((step + 1) * env.dt),
        "speed": float(np.mean(speeds)),
        "shake": float(np.mean(shakes)) if shakes else 0.0,
        "min_gap": float(np.min(gaps)),
        "low_gap": float(np.percentile(gaps, 5)),
    }


def sweep(driver, beams, label, rows, make_env, tracks):
    """한 설정으로 여러 트랙을 돌고 한 줄로 줄인다."""
    runs = [r for r in (drive(driver, make_env(EVAL_START + i), EVAL_START + i)
                        for i in range(tracks)) if r is not None]
    done = [r for r in runs if r["done"]]
    row = {
        "label": label,
        "tracks": len(runs),
        "completed": len(done),
        "lap_time": float(np.mean([r["lap_time"] for r in done])) if done else None,
        "speed": float(np.mean([r["speed"] for r in runs])),
        "shake": float(np.mean([r["shake"] for r in runs])),
        "min_gap": float(np.min([r["min_gap"] for r in runs])),
    }
    rows.append(row)
    lap = f"{row['lap_time']:5.1f}초" if row["lap_time"] else "   -  "
    print(f"  {label:<22s} {row['completed']:2d}/{row['tracks']:<2d}  {lap}  "
          f"{row['speed']:4.1f} m/s  떨림 {row['shake']:.3f}  "
          f"최소여유 {row['min_gap']:+.2f}m", flush=True)
    return row


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

    print("=" * 74)
    print(f"{Path(args.model).stem}  —  실차 조건 견디기")
    print("=" * 74)
    print(f"시드 {EVAL_START}~{EVAL_START + args.tracks - 1}\n")

    started = time.time()
    out = {"model": str(args.model),
           "wiring": "dense" if args.dense else "fly"}

    print("노면 (타이어가 버티는 힘. 학습은 75~115%)")
    out["friction"] = []
    for grip in FRICTIONS:
        sweep(driver, args.beams, f"마찰 {grip:.0%}", out["friction"],
              lambda s, g=grip: F1TenthEnv(n_beams=args.beams, track_seed=s,
                                           friction=g), args.tracks)

    print("\n제어 지연 (학습은 50ms)")
    out["latency"] = []
    for delay in LATENCIES:
        sweep(driver, args.beams, f"지연 {delay * 1000:.0f}ms", out["latency"],
              lambda s, d=delay: F1TenthEnv(n_beams=args.beams, track_seed=s,
                                            friction=1.0, latency=d),
              args.tracks)

    print("\n출발 자세 흐트러뜨리기 (옆으로 민 거리 / 돌려 놓은 각도)")
    out["pose"] = []
    for offset, angle in POSES:
        label = f"{offset:.2f}m / {angle:+.0f}도"
        runs = []
        for i in range(args.tracks):
            seed = EVAL_START + i
            env = F1TenthEnv(n_beams=args.beams, track_seed=seed, friction=1.0)
            r = drive(driver, env, seed, offset, angle)
            if r is not None:
                runs.append(r)
        done = [r for r in runs if r["done"]]
        row = {"label": label, "tracks": len(runs), "completed": len(done),
               "lap_time": float(np.mean([r["lap_time"] for r in done]))
               if done else None,
               "speed": float(np.mean([r["speed"] for r in runs])),
               "shake": float(np.mean([r["shake"] for r in runs])),
               "min_gap": float(np.min([r["min_gap"] for r in runs]))}
        out["pose"].append(row)
        lap = f"{row['lap_time']:5.1f}초" if row["lap_time"] else "   -  "
        print(f"  {label:<22s} {row['completed']:2d}/{row['tracks']:<2d}  {lap}  "
              f"{row['speed']:4.1f} m/s  떨림 {row['shake']:.3f}  "
              f"최소여유 {row['min_gap']:+.2f}m", flush=True)

    out["seconds"] = time.time() - started
    path = model_path(args.model).with_name(
        f"robust_{Path(args.model).stem}.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n저장: {path.name}")


if __name__ == "__main__":
    main()
