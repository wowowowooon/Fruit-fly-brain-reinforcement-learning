"""병렬 환경 처리량 측정. 강화학습을 돌리기 전에 최적 설정을 찾는다.

  .\\.venv\\Scripts\\python.exe bench_vec.py
"""

import sys
from pathlib import Path

# 상위 폴더에 있는 공통 모듈을 찾을 수 있게 한다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import numpy as np
import torch

from flyracer.brain.fly_driver import FlyDriver
from flyracer.brain.racing_policy import RacingPolicy
from flyracer.sim.vec_env import ParallelRunner


def measure(n_envs, workers, threads, policy, stats, steps=40):
    torch.set_num_threads(threads)
    runner = ParallelRunner(n_envs, 541, 10.0, workers=workers)
    hidden = None
    try:
        for _ in range(5):                    # 워밍업
            x = torch.from_numpy(runner.features(stats)).unsqueeze(1)
            action, _, _, hidden = policy.sample(x, hidden)
            runner.step(np.clip(action[:, 0].numpy(), -1, 1))
        started = time.perf_counter()
        for _ in range(steps):
            x = torch.from_numpy(runner.features(stats)).unsqueeze(1)
            action, _, _, hidden = policy.sample(x, hidden)
            runner.step(np.clip(action[:, 0].numpy(), -1, 1))
        elapsed = time.perf_counter() - started
    finally:
        runner.close()
    return n_envs * steps / elapsed


def main():
    driver = FlyDriver(n_beams=541).load("fly_model.pt")
    stats = (driver.encoder.mean, driver.encoder.std)
    policy = RacingPolicy(driver.model, len(driver.channels))

    print(f"{'환경':>5} {'작업자':>7} {'토치':>6} {'처리량':>12} {'밤새(8h)':>12}")
    print(f"{'':>5} {'단일':>7} {'4':>6} {188:>9} 스텝/초 {188 * 28800 / 1e6:>9.1f}M")
    for n_envs, workers, threads in [
        (16, 4, 4), (16, 8, 2), (24, 6, 3), (32, 8, 2),
        (32, 8, 4), (48, 8, 2), (64, 8, 2),
    ]:
        rate = measure(n_envs, workers, threads, policy, stats)
        print(f"{n_envs:>5} {workers:>7} {threads:>6} {rate:>9.0f} 스텝/초 "
              f"{rate * 28800 / 1e6:>9.1f}M")


if __name__ == "__main__":
    main()

