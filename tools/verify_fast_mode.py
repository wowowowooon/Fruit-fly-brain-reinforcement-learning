"""빔을 줄여서 학습해도 되는가.

강화학습은 수백만 스텝이 필요한데 1081빔은 한 스텝에 6.25ms 걸린다.
빔을 줄이면 훨씬 빠르지만, 모델이 보는 37채널이 달라지면 안 된다.

구역별 대표 거리를 '하위 10퍼센타일'로 뽑기 때문에 이론적으로는
빔 개수와 무관해야 한다. 실제 트랙과 거친 벽에서도 그런지 확인한다.
"""

import sys
from pathlib import Path

# 상위 폴더에 있는 공통 모듈을 찾을 수 있게 한다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import numpy as np

from controllers import follow_the_gap
from f1tenth_sim import LIDAR_BEAMS, F1TenthEnv
from fly_driver import LidarEncoder

CANDIDATES = (108, 271, 541)


def features_along(env_full, beams, seeds, steps=500):
    """같은 주행을 하면서 여러 해상도로 동시에 특징을 뽑는다."""
    encoders = {n: LidarEncoder(n, lidar_range=env_full.lidar_range)
                for n in (LIDAR_BEAMS,) + beams}
    # 해상도별 빔 각도. 전체 스캔에서 골라 쓴다.
    picks = {n: np.linspace(0, LIDAR_BEAMS - 1, n).round().astype(int)
             for n in (LIDAR_BEAMS,) + beams}

    collected = {n: [] for n in (LIDAR_BEAMS,) + beams}
    for seed in seeds:
        obs, _ = env_full.reset(seed=seed)
        for enc in encoders.values():
            enc.reset()
        for _ in range(steps):
            scan = obs[:-1] * env_full.lidar_range
            for n in collected:
                collected[n].append(
                    encoders[n].raw(scan[picks[n]], obs[-1], env_full.dt)
                )
            action = follow_the_gap(scan, env_full.beam_angles,
                                    env_full.lidar_range)
            obs, _, done, trunc, _ = env_full.step(action)
            if done or trunc:
                break
    return {n: np.array(v) for n, v in collected.items()}


def main():
    env = F1TenthEnv()
    print("=== 1. 같은 주행에서 해상도별 특징 차이 ===")
    print("   (학습 트랙 6개, 거친 벽 포함)")
    data = features_along(env, CANDIDATES, range(6))
    full = data[LIDAR_BEAMS]
    scale = full.std(axis=0)
    scale[scale < 1e-3] = 1e-3

    print(f"   {'빔':>6} {'평균오차':>10} {'99%오차':>10} {'최대':>8}  (전체 대비 시그마)")
    for n in CANDIDATES:
        err = np.abs(data[n] - full) / scale
        print(f"   {n:>6} {err.mean():>10.3f} "
              f"{np.percentile(err, 99):>10.3f} {err.max():>8.2f}")

    print("\n=== 2. 해상도별 시뮬 속도 ===")
    for n in (LIDAR_BEAMS,) + CANDIDATES:
        e = F1TenthEnv(n_beams=n)
        obs, _ = e.reset(seed=0)
        t0 = time.perf_counter()
        for _ in range(400):
            obs, _, d, t, _ = e.step(
                follow_the_gap(obs[:-1] * e.lidar_range, e.beam_angles,
                               e.lidar_range))
            if d or t:
                obs, _ = e.reset(seed=0)
        ms = (time.perf_counter() - t0) / 400 * 1000
        speedup = data and ""
        print(f"   {n:>6}빔  {ms:>6.2f} ms/step   "
              f"1M스텝에 {ms * 1e6 / 1000 / 3600:>5.1f}시간")

    print("\n=== 3. 해상도별 고전 컨트롤러 성능 ===")
    print("   (컨트롤러가 같은 판단을 하는가)")
    for n in (LIDAR_BEAMS,) + CANDIDATES:
        e = F1TenthEnv(n_beams=n)
        laps, ok = [], 0
        for seed in range(10):
            obs, _ = e.reset(seed=seed)
            for _ in range(4000):
                obs, _, d, t, info = e.step(
                    follow_the_gap(obs[:-1] * e.lidar_range, e.beam_angles,
                                   e.lidar_range))
                if d or t:
                    break
            if not info["crashed"]:
                ok += 1
                laps.append(info["lap_time"])
        print(f"   {n:>6}빔  완주 {ok}/10  "
              f"평균 랩타임 {np.mean(laps):>5.1f}초" if laps
              else f"   {n:>6}빔  전멸")


if __name__ == "__main__":
    main()

