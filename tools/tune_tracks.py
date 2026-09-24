"""교사가 제대로 도는지 확인하고, 트랙 난이도를 맞춘다.

모방학습을 돌리기 전에 항상 이걸 먼저 통과시켜야 한다.
교사가 못 도는 트랙에서 모은 데이터는 학생에게 독이 된다.
"""

import sys
from pathlib import Path

# 상위 폴더에 있는 공통 모듈을 찾을 수 있게 한다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np

import flyracer.sim.f1tenth_sim as S
from flyracer.sim.f1tenth_sim import F1TenthEnv
from flyracer.drive.raceline import StanleyRacer, outer_half_width, path_curvature


def drive(env, seeds, steps=6000):
    ok, laps, steer, worst_cte, progress = 0, [], [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        racer = StanleyRacer.for_env(env)
        cte = 0.0
        for _ in range(steps):
            front = env.pos + S.WHEELBASE * np.array(
                [np.cos(env.yaw), np.sin(env.yaw)])
            near = int(np.argmin(np.linalg.norm(racer.line - front, axis=1)))
            cte = max(cte, float(np.linalg.norm(front - racer.line[near])))
            action = racer.act(env.pos, env.yaw, env.speed)
            steer.append(action[0])
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        progress.append(info["progress"])
        worst_cte.append(cte)
        if not info["crashed"]:
            ok += 1
            laps.append(info["lap_time"])
    steer = np.abs(np.array(steer))
    return {
        "ok": ok, "n": len(list(seeds)),
        "lap": float(np.mean(laps)) if laps else 0.0,
        "progress": float(np.mean(progress)),
        "cte": float(np.mean(worst_cte)),
        "hard": float(np.mean(steer > 0.5)),
        "max": float(np.mean(steer > 0.8)),
    }


def line_safety(env, seeds):
    """라인이 벽에서 실제로 얼마나 떨어져 있는가."""
    slack = []
    for seed in seeds:
        env.reset(seed=seed)
        racer = StanleyRacer.for_env(env)
        from flyracer.drive.raceline import _closest_on_walls
        _, gap = _closest_on_walls(racer.line, env.seg_a, env.seg_b)
        need = outer_half_width(
            1.0 / np.maximum(np.abs(path_curvature(racer.line)), 1e-6))
        slack.append((gap - need).min())
    return float(np.min(slack)), float(np.mean(slack))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--beams", type=int, default=271)
    args = parser.parse_args()
    seeds = range(args.seeds)

    print(f"차 최소 회전반경 {S.WHEELBASE / np.tan(S.MAX_STEER):.2f} m  "
          f"(최대조향 {np.rad2deg(S.MAX_STEER):.1f}도)")
    print(f"{'여유':>5} {'장애물':>7} {'완주':>8} {'랩타임':>8} {'진행':>7} "
          f"{'라인이탈':>9} {'>0.5':>7} {'>0.8':>7} {'라인여유':>9}")

    settings = [
        (1.3, (0, 9)), (1.6, (0, 8)), (2.0, (0, 7)),
        (2.4, (0, 6)), (2.8, (0, 5)),
    ]
    for margin, obstacles in settings:
        S.CURVE_MARGIN = margin
        S.TIGHTNESS_RANGE = (margin, margin + 2.0)
        S.OBSTACLE_COUNT = obstacles
        env = F1TenthEnv(n_beams=args.beams)
        result = drive(env, seeds)
        worst, mean = line_safety(env, range(min(8, args.seeds)))
        print(f"{margin:>5.1f} {obstacles[1]:>7} "
              f"{result['ok']:>4}/{result['n']:<3} {result['lap']:>7.1f}초 "
              f"{result['progress']:>6.0%} {result['cte']:>8.2f}m "
              f"{result['hard']:>6.1%} {result['max']:>6.1%} "
              f"{worst:>+8.2f}m")


if __name__ == "__main__":
    main()

