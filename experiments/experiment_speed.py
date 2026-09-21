"""한 스텝 판단하는 데 얼마나 걸리나.

초파리 배선을 쓰는 이유 중 하나가 연산이 가볍다는 것이다. 실차에서는
이 시간이 곧 제어 지연이 되고, 지연은 랩타임과 안정성에 직접 영향을
준다. 40Hz 로 돌려야 하므로 한 스텝에 25ms 안에 끝나야 한다.

  python experiment_speed.py
"""

import sys
from pathlib import Path

# 상위 폴더에 있는 공통 모듈을 찾을 수 있게 한다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import numpy as np
import torch

from f1tenth_sim import CONTROL_DT, F1TenthEnv
from fly_driver import FlyDriver

ROUNDS = 600
WARMUP = 50


def measure(driver, obs, label):
    driver.reset()
    for _ in range(WARMUP):
        driver.act(obs)
    times = []
    for _ in range(ROUNDS):
        start = time.perf_counter()
        driver.act(obs)
        times.append(time.perf_counter() - start)
    times = np.array(times) * 1000.0            # ms
    budget = CONTROL_DT * 1000.0
    print(f"  {label:<22} 중앙값 {np.median(times):>6.2f} ms   "
          f"95% {np.percentile(times, 95):>6.2f} ms   "
          f"제어 주기의 {np.median(times) / budget:>5.1%}")
    return float(np.median(times))


def main():
    torch.set_num_threads(1)                    # 실차 조건에 가깝게
    env = F1TenthEnv(n_beams=541)
    obs, _ = env.reset(seed=700)
    for _ in range(20):
        obs, *_ = env.step(np.array([0.3, 0.5]))

    print("=" * 66)
    print(f"한 스텝 판단 시간  (스레드 1개, 라이다 {env.n_beams}빔)")
    print(f"제어 주기 {CONTROL_DT * 1000:.1f} ms 안에 끝나야 한다")
    print("=" * 66)

    fly = FlyDriver(n_beams=541, fly_wiring=True)
    dense = FlyDriver(n_beams=541, fly_wiring=False)
    fly_ms = measure(fly, obs, "C 초파리 배선")
    dense_ms = measure(dense, obs, "B 밀집 배선")

    print("\n" + "=" * 66)
    print(f"  유효 연결   초파리 {fly.effective_params():,} vs "
          f"밀집 {dense.effective_params():,} "
          f"({dense.effective_params() / fly.effective_params():.0f}배)")
    print(f"  실제 시간   {dense_ms / fly_ms:.2f}배 빠름")
    if dense_ms / fly_ms < 1.3:
        print("\n  -> 연결은 59배 적은데 시간은 별로 안 줄었다.")
        print("     마스크로 끄는 방식이라 곱셈은 그대로 하기 때문이다.")
        print("     진짜 연산 이득을 보려면 희소 연산으로 바꿔야 한다.")


if __name__ == "__main__":
    main()
