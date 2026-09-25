"""실차 조향·차체 반응을 재서 시뮬과 맞는지 본다.

왜 필요한가
----------
시뮬은 조향을 '기울기 제한' 으로 모델링한다 (3.2 rad/s, 풀락 0.22초).
실차 ESP32 는 '지수 지연' 이다 (alpha=0.20 @50Hz, 시정수 약 90ms).
모양이 다른 모델이라 크기가 비슷해도 과도 구간이 다르다. 그 차이가
크면 시뮬에서 배운 조향이 실차에서 안 맞는다.

기존 텔레메트리 로그로는 못 잰다. 24개 로그 전부 yaw_rate_radps 와
lateral_accel_mps2 가 비어 있다 (IMU 가 안 붙었거나 로깅이 꺼져 있었다).
그래서 차에서 5분짜리 측정을 한 번 해야 한다.

차에서 할 것
-----------
바퀴를 띄우지 말고 **넓은 데서 저속(1~2 m/s)** 으로, 조향을 계단처럼
좌우로 꺾었다 풀었다 한다. 20Hz 로 기록되면 충분하다.

  1. IMU 가 붙어 있는지 먼저 확인
       ros2 topic hz /imu/data
       ros2 topic echo /imu/data --field angular_velocity
  2. 텔레메트리 로깅을 켠 채로 주행
  3. logs/<시각>/telemetry.csv 를 이 PC 로 가져와서

       python deploy/measure_response.py <telemetry.csv>

무엇을 보나
----------
  조향 명령(final_steering) -> 요레이트(yaw_rate_radps) 의 지연과 시정수
  같은 입력을 시뮬에 넣었을 때의 응답과 겹쳐 본다

  python deploy/measure_response.py <csv> --plot outputs/response.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv

import numpy as np

from flyracer.sim.f1tenth_sim import (CONTROL_LATENCY, MAX_STEER,
                                      STEER_RATE_MAX, WHEELBASE)


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig",
                                    errors="ignore")))
    def col(key):
        out = []
        for r in rows:
            try:
                v = float(r.get(key, ""))
                out.append(v if v == v else np.nan)
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.array(out)
    return rows, col


def sim_steer(cmd, dt):
    """시뮬의 서보 모델: 한 스텝에 STEER_RATE_MAX*dt 만큼만 움직인다."""
    out = np.zeros_like(cmd)
    cur = 0.0
    slew = STEER_RATE_MAX * dt
    for i, want in enumerate(cmd):
        cur += np.clip(want - cur, -slew, slew)
        out[i] = cur
    return out


def esp_steer(cmd, dt, alpha=0.20, esp_dt=0.02):
    """실차 ESP 모델: 매 20ms 마다 목표로 alpha 만큼 다가간다."""
    out = np.zeros_like(cmd)
    cur = 0.0
    for i, want in enumerate(cmd):
        # 기록 한 칸(dt) 동안 ESP 루프가 몇 번 도는지
        for _ in range(max(1, int(round(dt / esp_dt)))):
            cur += (want - cur) * alpha
        out[i] = cur
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()

    rows, col = load(args.csv)
    t = col("time_sec")
    steer = col("final_steering")          # S in [-1,1] = 풀락 비율
    yaw = col("yaw_rate_radps")
    speed = col("speed_mps")

    ok = np.isfinite(t) & np.isfinite(steer)
    print(f"행 {len(rows)}   조향 {np.isfinite(steer).sum()}   "
          f"요레이트 {np.isfinite(yaw).sum()}   속도 {np.isfinite(speed).sum()}")
    if np.isfinite(yaw).sum() < 50:
        print("\n요레이트가 없다. IMU 를 붙이고 다시 재야 한다.")
        print("  ros2 topic hz /imu/data   로 먼저 확인할 것")
        return

    good = ok & np.isfinite(yaw) & np.isfinite(speed)
    t, steer, yaw, speed = t[good], steer[good], yaw[good], speed[good]
    dt = float(np.median(np.diff(t)))
    print(f"기록 주기 {dt*1000:.1f} ms")

    # 실제 조향각 [rad] 과 자전거 모델이 예측하는 요레이트
    angle = steer * MAX_STEER
    fast = speed > 0.5
    if fast.sum() < 50:
        print("저속 구간뿐이라 요레이트로는 못 잰다. 1~2 m/s 로 다시.")
        return

    # 지연: 조향과 요레이트의 상호상관이 최대가 되는 시차
    a = angle[fast] - angle[fast].mean()
    b = yaw[fast] - yaw[fast].mean()
    n = min(len(a), 400)
    lags = np.arange(0, n // 4)
    score = [np.corrcoef(a[:len(a) - k], b[k:])[0, 1] for k in lags]
    best = int(lags[int(np.nanargmax(score))])
    print(f"\n조향 -> 요레이트 지연  {best * dt * 1000:.0f} ms  "
          f"(상관 {np.nanmax(score):.2f})")
    print(f"  시뮬이 가정한 지연     {CONTROL_LATENCY * 1000:.0f} ms")

    # 자전거 모델과 비교 — 실제로 그만큼 도는가
    pred = speed[fast] / WHEELBASE * np.tan(angle[fast])
    scale = float(np.polyfit(pred, yaw[fast], 1)[0])
    print(f"\n실제 요레이트 / 자전거모델 예측 = {scale:.2f}")
    print("  1.0 이면 시뮬과 같다. 0.5 면 실제로는 절반만 돈다는 뜻이고,")
    print("  그러면 조향 스케일이나 타이어 그립을 그만큼 낮춰야 한다.")

    # 두 서보 모델을 같은 명령에 넣어 얼마나 갈리는지
    s_sim = sim_steer(steer, dt)
    s_esp = esp_steer(steer, dt)
    diff = np.abs(s_sim - s_esp)
    print(f"\n서보 모델 차이 (같은 명령을 넣었을 때)")
    print(f"  평균 {diff.mean() * MAX_STEER * 57.3:.2f}도   "
          f"최대 {diff.max() * MAX_STEER * 57.3:.2f}도")
    print("  최대가 5도를 넘으면 시뮬 서보 모델을 ESP 쪽으로 바꾸고")
    print("  재학습하는 편이 낫다.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        ax[0].plot(t, steer, label="명령 S", lw=1)
        ax[0].plot(t, s_sim, label="시뮬 서보 모델", lw=1)
        ax[0].plot(t, s_esp, label="ESP 서보 모델", lw=1)
        ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(t, yaw, label="실제 요레이트", lw=1)
        ax[1].plot(t, speed / WHEELBASE * np.tan(angle),
                   label="자전거 모델 예측", lw=1)
        ax[1].legend(); ax[1].grid(alpha=.3)
        ax[1].set_xlabel("시간 [s]")
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(args.plot)
        print(f"\n그림 저장: {args.plot}")


if __name__ == "__main__":
    main()
