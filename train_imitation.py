"""Follow the Gap을 교사로 삼아 초파리 회로에 운전을 가르친다.

배선은 초파리 것 그대로 두고, 신호 세기만 조정한다.
교사 주행 중 일부러 흔들어서, 교사가 잘 안 가는 상황도 겪게 한다.

센서와 차량 값은 f1tenth_sim의 대회 규정 사양을 그대로 쓴다.
특히 라이다 사거리가 바뀌면 입력 분포가 달라지므로 다시 학습해야 한다.
"""

import argparse

import numpy as np
import torch
from torch import nn

from controllers import follow_the_gap
from f1tenth_sim import CONTROL_DT, LIDAR_BEAMS, LIDAR_RANGE, F1TenthEnv
from fly_driver import FlyDriver, LidarEncoder
from progress import Logger, Progress

SEQ_LEN = 64          # RNN에 한 번에 넣는 시간 길이 (40Hz에서 1.6초)
NOISE = 0.25          # 교사 행동에 섞는 흔들림
EPISODE_STEPS = 1200  # 40Hz에서 30초. 트랙 한 바퀴보다 조금 길다


def make_env(args):
    return F1TenthEnv(n_beams=args.beams, dt=CONTROL_DT,
                      lidar_range=args.lidar_range)


def collect(args, rng, log=print):
    """교사 주행을 기록한다. 기록하는 정답은 흔들리기 전 값."""
    env = make_env(args)
    encoder = LidarEncoder(args.beams, lidar_range=args.lidar_range)
    episodes = []
    bar = Progress(args.tracks, "수집", log)

    for track in range(args.tracks):
        obs, _ = env.reset(seed=track)
        encoder.reset()
        features, labels = [], []

        for _ in range(EPISODE_STEPS):
            scan = obs[:-1] * env.lidar_range
            action = follow_the_gap(scan, env.beam_angles, env.lidar_range)
            features.append(encoder.raw(scan, obs[-1], env.dt))
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
        steps = sum(len(f) for f, _ in episodes)
        bar.step(track + 1, f"{steps:>6}스텝")

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


def evaluate(driver, seeds, args, steps=1600):
    """실제로 달리게 해서 얼마나 갔는지 본다."""
    env = make_env(args)
    results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        driver.reset()
        for _ in range(steps):
            obs, _, done, truncated, info = env.step(driver.act(obs))
            if done or truncated:
                break
        results.append((info["progress"], not info["crashed"]))
    progress = float(np.mean([r[0] for r in results]))
    survived = sum(r[1] for r in results)
    return progress, survived


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--beams", type=int, default=LIDAR_BEAMS)
    parser.add_argument("--lidar-range", type=float, default=LIDAR_RANGE)
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--out", type=str, default="fly_model.pt")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    log = Logger()

    log("=" * 68)
    log("1단계 모방학습 — Follow the Gap 따라하기 (강화학습의 출발점)")
    log("=" * 68)
    log(f"하드웨어  라이다 {args.beams}빔 / {args.lidar_range:.0f}m / "
        f"{1 / CONTROL_DT:.0f}Hz  (ROBORACER 규정 사양)")
    log(f"계획      트랙 {args.tracks}개 수집 -> {args.epochs}에폭 학습\n")

    episodes = collect(args, rng, log)
    samples = sum(len(f) for f, _ in episodes)
    log(f"에피소드 {len(episodes)}개, 총 {samples}스텝\n")

    # 입력 값의 폭을 맞춰준다. 안 하면 신호가 죽어서 학습이 안 된다.
    stacked = np.concatenate([f for f, _ in episodes])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0)
    episodes = [((f - mean) / np.maximum(std, 1e-3), y) for f, y in episodes]

    baseline = np.concatenate([y for _, y in episodes]).var(axis=0).mean()
    log(f"상수로 찍었을 때 오차 {baseline:.4f} (이보다 낮아야 배운 것)\n")

    driver = FlyDriver(n_beams=args.beams, lidar_range=args.lidar_range)
    driver.encoder.set_stats(mean, std)
    log(driver.summary())
    log("")

    optimizer = torch.optim.Adam(driver.model.parameters(), lr=args.lr)
    # 뒤로 갈수록 보폭을 줄여서 좋은 지점을 지나치지 않게 한다.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fn = nn.MSELoss()
    eval_seeds = [100, 101, 102, 103, 104]
    best = -1.0

    bar = Progress(args.epochs, "학습", log)
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

        note = f"오차 {np.mean(losses):.4f}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            driver.model.eval()
            progress, survived = evaluate(driver, eval_seeds, args)
            note += f"  시험주행 {progress:>6.1%} {survived}/{len(eval_seeds)}완주"
            if progress > best:
                best = progress
                driver.save(args.out)
                note += " 저장"
        bar.step(epoch, note)
    bar.done()

    log(f"\n최고 진행도 {best:.1%}  -> {args.out} 저장됨")
    log("다음: 강화학습으로 이 모델을 출발점 삼아 랩타임을 줄인다.")
    log.close()


if __name__ == "__main__":
    main()
