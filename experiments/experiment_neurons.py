"""주행 중 어느 뉴런이 언제 켜지는지 들여다본다.

초파리 배선을 쓰는 값어치 중 확실한 하나가 '왜 그렇게 조향했는지 볼 수
있다'는 것이다. 일반 신경망은 뉴런에 이름이 없어서 못 하지만, 우리
뉴런에는 LC4(접근 감지), EPG(방향 감각), LAL(조향 전단계) 같은
실제 이름과 역할이 붙어 있다.

급코너에서 실제로 접근 감지 세포가 켜지는지, 직선에서는 무엇이 도는지
세어 본다.

  python experiment_neurons.py --map "...\\ifac_roboracer.yaml"
  python experiment_neurons.py --seed 700
"""

import sys
from pathlib import Path

# 상위 폴더에 있는 공통 모듈을 찾을 수 있게 한다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import re
from collections import defaultdict

import numpy as np
import torch

from f1tenth_sim import F1TenthEnv, LIDAR_RANGE, MAX_STEER
from fly_driver import FlyDriver
from paths import OUTPUTS

STEPS = 2000
SHARP_STEER = 0.55        # 이 이상 꺾으면 급코너로 본다 (정규화값)


def family(name):
    """LC9_L -> LC, LAL067 -> LAL 처럼 계열만 남긴다."""
    match = re.match(r"[A-Za-z]+", str(name))
    return match.group() if match else "기타"


def drive(driver, env, obs):
    """한 바퀴 돌며 뉴런 상태와 조향을 같이 기록한다."""
    states, steers, speeds = [], [], []
    hidden = None
    driver.reset()
    for _ in range(STEPS):
        features = driver.encoder.encode(
            obs[:-3] * driver.lidar_range, obs[-3:], driver.dt)
        x = torch.from_numpy(features).view(1, 1, -1)
        with torch.no_grad():
            raw, hidden = driver.model.rnn(x, hidden)
            action = driver.model.command(
                raw[..., : driver.model.n_output])[0, 0].numpy()
        states.append(raw[0, 0].numpy().copy())
        steers.append(float(action[0]))
        speeds.append(env.speed)
        obs, reward, finished, timeout, info = env.step(action)
        if finished or timeout:
            break
    return np.array(states), np.array(steers), np.array(speeds), info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="C_fly871.pt")
    parser.add_argument("--map", default=None)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--beams", type=int, default=541)
    args = parser.parse_args()

    driver = FlyDriver(n_beams=args.beams, lidar_range=LIDAR_RANGE)
    driver.load(args.model)

    if args.map:
        from test_on_map import find_start
        from track_map import MapEnv, MapTrack
        track = MapTrack(args.map)
        env = MapEnv(track, n_beams=args.beams)
        env.start_pose = find_start(track, env)
        where = track.name
    else:
        env = F1TenthEnv(n_beams=args.beams, track_seed=args.seed)
        where = f"시드 {args.seed}"
    obs, _ = env.reset()

    states, steers, speeds, info = drive(driver, env, obs)
    sharp = np.abs(steers) > SHARP_STEER
    straight = np.abs(steers) < 0.15

    print("=" * 70)
    print(f"{where}   {len(states)}스텝  진행 {info['progress']:.0%}")
    print(f"급코너 {sharp.sum()}스텝 ({sharp.mean():.0%})   "
          f"직선 {straight.sum()}스텝 ({straight.mean():.0%})")
    print(f"최대 조향 {np.abs(steers).max() * np.rad2deg(MAX_STEER):.0f}도")
    print("=" * 70)

    if sharp.sum() < 10 or straight.sum() < 10:
        print("급코너나 직선이 너무 적어 비교할 수 없다")
        return

    # 뉴런마다 급코너일 때와 직선일 때의 활동 차이
    busy = np.abs(states)
    gap = busy[sharp].mean(axis=0) - busy[straight].mean(axis=0)
    types = driver.ordered.type.astype(str).values

    print("\n급코너에서 더 켜지는 뉴런 계열 (평균 차이)")
    print("-" * 70)
    by_family = defaultdict(list)
    for i, name in enumerate(types):
        by_family[family(name)].append(gap[i])
    ranked = sorted(by_family.items(),
                    key=lambda kv: -float(np.mean(kv[1])))
    roles = {
        "LC": "물체·접근 감지 (눈)", "LPLC": "충돌 임박 감지 (눈)",
        "LPT": "광역 흐름 (눈)", "HS": "수평 흐름 (눈)",
        "H": "수평 흐름 (눈)", "LAL": "조향 전단계 (뇌)",
        "EPG": "방향 감각 - 나침반 (뇌)", "FC": "목표 방향 (뇌)",
        "ExR": "상태 조절 (뇌)", "DNa": "조향 출력", "DNb": "조향·다리 출력",
        "DNg": "조향·속도 출력", "DNp": "도피·정지 출력", "MDN": "후진·제동 출력",
    }
    for name, values in ranked[:10]:
        print(f"  {name:<8s} {len(values):>4d}개  "
              f"{np.mean(values):>+7.4f}   {roles.get(name, '')}")

    print("\n개별 뉴런 상위 12개")
    print("-" * 70)
    for i in np.argsort(-gap)[:12]:
        row = driver.ordered.iloc[i]
        print(f"  {str(row.type):<12s} {str(row.instance):<16s} "
              f"{gap[i]:>+7.4f}")

    # 접근 감지 세포가 실제로 위험할 때 켜지는가
    loom_idx = [i for i, t in enumerate(types)
                if re.match(r"^(LC4|LPLC\d*)$", t)]
    if loom_idx:
        loom = busy[:, loom_idx].mean(axis=1)
        print("\n" + "=" * 70)
        print("접근 감지 세포 (LC4 / LPLC) 가 위험할 때 켜지는가")
        print("=" * 70)
        print(f"  세포 {len(loom_idx)}개")
        print(f"  급코너일 때 {loom[sharp].mean():.4f}")
        print(f"  직선일 때   {loom[straight].mean():.4f}")
        ratio = loom[sharp].mean() / max(loom[straight].mean(), 1e-9)
        print(f"  비율        {ratio:.2f}배")
        order = np.corrcoef(loom, np.abs(steers))[0, 1]
        print(f"  조향 크기와의 상관 {order:+.3f}")
        print("\n  -> 급코너에서 더 켜진다" if ratio > 1.1 else
              "\n  -> 뚜렷한 차이 없음")

    out = OUTPUTS / f"neurons_{where.replace(' ', '')}.npz"
    np.savez_compressed(out, states=states.astype(np.float32),
                        steers=steers, speeds=speeds, types=types)
    print(f"\n저장: {out.name}")


if __name__ == "__main__":
    main()
