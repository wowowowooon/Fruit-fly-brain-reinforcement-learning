"""Follow the Gap을 교사로 삼아 초파리 회로에 운전을 가르친다.

배선은 초파리 것 그대로 두고, 신호 세기만 조정한다.
교사 주행 중 일부러 흔들어서, 교사가 잘 안 가는 상황도 겪게 한다.
"""

import argparse
import time

import numpy as np
import torch
from torch import nn

from f1tenth_sim import LIDAR_RANGE, MAX_STEER, F1TenthEnv
from fly_driver import FlyDriver, LidarEncoder

SEQ_LEN = 32          # RNN에 한 번에 넣는 시간 길이
NOISE = 0.25          # 교사 행동에 섞는 흔들림


def teacher(env, scan):
    """가장 멀리 뚫린 방향으로 도는 고전 방식."""
    smoothed = np.convolve(scan, np.ones(9) / 9, mode="same")
    best = int(np.argmax(smoothed))
    steer = env.beam_angles[best] / MAX_STEER
    return np.array([np.clip(steer, -1, 1), 1.0 - 0.7 * abs(steer)])


def collect(n_tracks, n_beams, rng):
    """교사 주행을 기록한다. 기록하는 정답은 흔들리기 전 값."""
    env = F1TenthEnv(n_beams=n_beams)
    encoder = LidarEncoder(n_beams)
    episodes = []

    for track in range(n_tracks):
        obs, _ = env.reset(seed=track)
        encoder.reset()
        features, labels = [], []

        for _ in range(1200):
            scan = obs[:-1] * LIDAR_RANGE
            action = teacher(env, scan)
            features.append(encoder.raw(scan, obs[-1]))
            labels.append(action.copy())

            noisy = np.clip(action + rng.normal(0, NOISE, 2), -1, 1)
            obs, _, done, truncated, _ = env.step(noisy)
            if done or truncated:
                break

        if len(features) >= SEQ_LEN:
            episodes.append((
                np.array(features, dtype=np.float32),
                np.array(labels, dtype=np.float32),
            ))

    return episodes


def to_batches(episodes, batch_size, rng):
    """에피소드를 고정 길이 토막으로 잘라 섞는다."""
    chunks = []
    for features, labels in episodes:
        for start in range(0, len(features) - SEQ_LEN, SEQ_LEN // 2):
            chunks.append((
                features[start:start + SEQ_LEN],
                labels[start:start + SEQ_LEN],
            ))

    rng.shuffle(chunks)
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        batch = chunks[i:i + batch_size]
        yield (
            torch.from_numpy(np.stack([c[0] for c in batch])),
            torch.from_numpy(np.stack([c[1] for c in batch])),
        )


def evaluate(driver, seeds, n_beams):
    """실제로 달리게 해서 얼마나 갔는지 본다."""
    env = F1TenthEnv(n_beams=n_beams)
    results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        driver.reset()
        for _ in range(1500):
            obs, _, done, truncated, info = env.step(driver.act(obs))
            if done or truncated:
                break
        results.append((info["progress"], not info["crashed"]))
    progress = float(np.mean([r[0] for r in results]))
    survived = sum(r[1] for r in results)
    return progress, survived


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--beams", type=int, default=108)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    print(f"교사 주행 수집 중 (트랙 {args.tracks}개)...")
    started = time.time()
    episodes = collect(args.tracks, args.beams, rng)
    samples = sum(len(f) for f, _ in episodes)
    print(f"에피소드 {len(episodes)}개, 총 {samples}스텝  ({time.time() - started:.0f}초)\n")

    # 입력 값의 폭을 맞춰준다. 안 하면 신호가 죽어서 학습이 안 된다.
    stacked = np.concatenate([f for f, _ in episodes])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0)
    episodes = [((f - mean) / np.maximum(std, 1e-3), y) for f, y in episodes]
    print(f"입력 정규화 후 표준편차 {np.concatenate([f for f, _ in episodes]).std():.3f}")

    baseline = np.concatenate([y for _, y in episodes]).var(axis=0).mean()
    print(f"상수로 찍었을 때 오차 {baseline:.4f} (이보다 낮아야 배운 것)\n")

    driver = FlyDriver(n_beams=args.beams)
    driver.encoder.set_stats(mean, std)
    print(driver.summary(), "\n")

    optimizer = torch.optim.Adam(driver.model.parameters(), lr=args.lr)
    # 뒤로 갈수록 보폭을 줄여서 좋은 지점을 지나치지 않게 한다.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fn = nn.MSELoss()
    eval_seeds = [100, 101, 102, 103, 104]
    best = -1.0

    print(f"{'에폭':>6} {'오차':>8} {'시험주행':>10} {'완주':>6}")
    for epoch in range(1, args.epochs + 1):
        driver.model.train()
        losses = []
        for x, y in to_batches(episodes, args.batch, rng):
            optimizer.zero_grad()
            pred, _ = driver.model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(driver.model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        if epoch % 2 == 0 or epoch == args.epochs:
            driver.model.eval()
            progress, survived = evaluate(driver, eval_seeds, args.beams)
            mark = ""
            if progress > best:
                best = progress
                driver.save()
                mark = "  <- 저장"
            print(f"{epoch:>4}/{args.epochs} {np.mean(losses):>8.4f} "
                  f"{progress:>9.1%} {survived:>4}/{len(eval_seeds)}{mark}")
        else:
            print(f"{epoch:>4}/{args.epochs} {np.mean(losses):>8.4f}")

    print(f"\n최고 진행도 {best:.1%}  -> fly_model.pt 저장됨")
    print("viewer.py 를 다시 실행하고 F 를 누르면 학습된 회로가 운전합니다.")


if __name__ == "__main__":
    main()
