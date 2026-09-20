"""강화학습(PPO)으로 랩타임을 줄인다.

모방학습은 교사(레이싱 라인 + Stanley)를 흉내 내는 데서 끝난다.
교사보다 빨라질 수 없고, 교사가 맵으로 아는 것(블라인드 코너)도 배울 수 없다.
여기서부터는 정답 없이 점수만 보고 스스로 빨라진다.

  점수 = 트랙을 따라 나아간 거리(m).  충돌하면 크게 깎고 끝낸다.
  스텝 시간이 고정이므로 이걸 최대화하면 곧 랩타임이 줄어든다.
  코너를 깎는 레이싱 라인도 여기서 저절로 나온다.

학습 중 센서를 일부러 망가뜨린다(도메인 랜덤화). 깨끗한 값만 보고 배우면
실차의 지저분한 라이다에서 무너진다.
"""

import argparse
import copy
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from f1tenth_sim import CONTROL_DT, LIDAR_BEAMS, LIDAR_RANGE, F1TenthEnv
from fly_driver import N_MOTION, FEATURE_LIMIT, FlyDriver, LidarEncoder
from progress import Logger, Progress
from racing_policy import RacingPolicy
from sensor_noise import DomainRandomizer
from vec_env import ParallelRunner

GAMMA = 0.99          # 먼 미래를 얼마나 따질지
LAMBDA = 0.95         # 이득 추정의 부드러움
CLIP = 0.2            # 한 번에 정책을 얼마나 바꿀지 제한
VALUE_COEF = 0.5
MAX_GRAD = 0.5

# 모방학습으로 배운 것을 잊지 않게 붙드는 항목.
# 점수만 쫓다 보면 "일단 안 박고 도는 법"을 통째로 잊어버리는 구간이 생긴다.
# 출발 시점의 정책에서 너무 멀어지면 약하게 당긴다. 계산 비용은 거의 없다.
# 학습이 진행되면 계수를 줄여서, 나중에는 교사를 넘어설 수 있게 풀어 준다.
ANCHOR_COEF = 0.05    # 처음 붙드는 세기
ANCHOR_DECAY = 0.6    # 전체 학습의 이 지점까지 붙들고, 이후로는 0 으로


def anchor_weight(step, total):
    """지금 시점에 모방학습 정책을 얼마나 붙들지."""
    span = max(total * ANCHOR_DECAY, 1.0)
    return ANCHOR_COEF * max(0.0, 1.0 - step / span)


class Runner:
    """여러 트랙을 동시에 굴린다. 한 트랙만 보면 그 트랙만 외운다."""

    def __init__(self, n_envs, beams, lidar_range, seed=0, randomize=True):
        self.rng = np.random.default_rng(seed)
        self.envs, self.encoders, self.faults = [], [], []
        for i in range(n_envs):
            env = F1TenthEnv(n_beams=beams, lidar_range=lidar_range)
            self.envs.append(env)
            self.encoders.append(LidarEncoder(beams, lidar_range=lidar_range))
            self.faults.append(
                DomainRandomizer(env.beam_angles, lidar_range,
                                 np.random.default_rng(seed * 1000 + i))
                if randomize else None
            )
        self.n_envs = n_envs
        self.track_pool = 400        # 이만큼의 서로 다른 트랙에서 뽑아 쓴다
        self.obs = [None] * n_envs
        self.done_stats = []
        for i in range(n_envs):
            self._reset(i)

    def _reset(self, i):
        seed = int(self.rng.integers(self.track_pool))
        raw, _ = self.envs[i].reset(seed=seed)
        self.encoders[i].reset()
        if self.faults[i] is not None:
            self.faults[i].reset()
            raw = self.faults[i](raw)
        self.obs[i] = raw

    def features(self, stats):
        """현재 관측을 모델 입력으로. (N, C)"""
        mean, std = stats
        out = []
        for i, env in enumerate(self.envs):
            raw = self.obs[i]
            f = self.encoders[i].raw(raw[:-N_MOTION] * env.lidar_range,
                                       raw[-N_MOTION:], env.dt)
            out.append((f - mean) / std)
        return np.clip(np.stack(out), -FEATURE_LIMIT,
                       FEATURE_LIMIT).astype(np.float32)

    def step(self, actions):
        """행동을 넣고 보상과 종료 여부를 돌려준다."""
        rewards = np.zeros(len(self.envs), dtype=np.float32)
        dones = np.zeros(len(self.envs), dtype=np.float32)
        for i, env in enumerate(self.envs):
            raw, reward, done, truncated, info = env.step(actions[i])
            if self.faults[i] is not None:
                raw = self.faults[i](raw)
            self.obs[i] = raw
            rewards[i] = reward
            if done or truncated:
                dones[i] = 1.0
                self.done_stats.append({
                    "progress": info["progress"],
                    "crashed": info["crashed"],
                    "lap_time": info["lap_time"],
                    "length": env.lap_length,
                })
                self._reset(i)
        return rewards, dones

    def drain(self):
        stats, self.done_stats = self.done_stats, []
        return stats


def gae(rewards, values, dones, last_value):
    """지금 행동이 평균보다 얼마나 좋았는지 (이득)."""
    steps = len(rewards)
    advantages = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(steps)):
        nxt = last_value if t == steps - 1 else values[t + 1]
        alive = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * nxt * alive - values[t]
        running = delta + GAMMA * LAMBDA * alive * running
        advantages[t] = running
    return advantages, advantages + values


def rollout(runner, policy, hidden, stats, steps):
    """정책을 굴려서 학습에 쓸 조각을 모은다."""
    n = runner.n_envs
    obs = np.zeros((steps, n, policy.critic[0].in_features - policy.n_output),
                   dtype=np.float32)
    acts = np.zeros((steps, n, 2), dtype=np.float32)
    logps = np.zeros((steps, n), dtype=np.float32)
    values = np.zeros((steps, n), dtype=np.float32)
    rewards = np.zeros((steps, n), dtype=np.float32)
    dones = np.zeros((steps, n), dtype=np.float32)
    start_hidden = hidden.clone() if hidden is not None else None

    for t in range(steps):
        feats = runner.features(stats)
        x = torch.from_numpy(feats).unsqueeze(1)          # (N,1,C)
        action, logp, value, hidden = policy.sample(x, hidden)
        obs[t] = feats
        acts[t] = action[:, 0].numpy()
        logps[t] = logp[:, 0].numpy()
        values[t] = value[:, 0].numpy()
        rewards[t], dones[t] = runner.step(np.clip(acts[t], -1, 1))
        # 끝난 트랙은 기억을 지운다
        keep = torch.from_numpy(1.0 - dones[t]).float().unsqueeze(-1)
        hidden = hidden * keep

    with torch.no_grad():
        x = torch.from_numpy(runner.features(stats)).unsqueeze(1)
        _, last_value, _ = policy(x, hidden)
    return {
        "obs": obs, "acts": acts, "logps": logps, "values": values,
        "rewards": rewards, "dones": dones,
        "last_value": last_value[:, 0].numpy(),
        "start_hidden": start_hidden, "end_hidden": hidden,
    }


def update(policy, optimizer, batch, epochs, entropy_coef, anchor=0.0,
           teacher=None):
    """모은 조각으로 정책을 민다. 시간 순서를 지켜야 기억이 맞는다."""
    advantages, returns = gae(
        batch["rewards"], batch["values"], batch["dones"], batch["last_value"]
    )
    norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs = torch.from_numpy(batch["obs"])
    acts = torch.from_numpy(batch["acts"])
    old_logps = torch.from_numpy(batch["logps"])
    adv = torch.from_numpy(norm.astype(np.float32))
    ret = torch.from_numpy(returns.astype(np.float32))
    alive = torch.from_numpy(1.0 - batch["dones"]).unsqueeze(-1)
    steps = obs.shape[0]

    # 모방학습 정책이 같은 상황에서 뭐라고 했을지 미리 받아 둔다
    reference = None
    if anchor > 0.0 and teacher is not None:
        with torch.no_grad():
            hidden = batch["start_hidden"]
            hidden = hidden.detach() if hidden is not None else None
            picks = []
            for t in range(steps):
                mean, hidden = teacher(obs[t].unsqueeze(1), hidden)
                picks.append(mean[:, 0])
                hidden = hidden * alive[t]
            reference = torch.stack(picks)

    losses = []
    for _ in range(epochs):
        hidden = batch["start_hidden"]
        if hidden is not None:
            hidden = hidden.detach()
        policy_loss = value_loss = entropy_sum = anchor_loss = 0.0
        for t in range(steps):
            logp, entropy, value, mean, hidden = policy.evaluate(
                obs[t].unsqueeze(1), hidden, acts[t].unsqueeze(1)
            )
            ratio = (logp[:, 0] - old_logps[t]).exp()
            clipped = torch.clamp(ratio, 1 - CLIP, 1 + CLIP)
            policy_loss = policy_loss - torch.min(
                ratio * adv[t], clipped * adv[t]
            ).mean()
            value_loss = value_loss + (value[:, 0] - ret[t]).pow(2).mean()
            entropy_sum = entropy_sum + entropy.mean()
            if reference is not None:
                anchor_loss = anchor_loss + (
                    mean[:, 0] - reference[t]
                ).pow(2).mean()
            hidden = hidden * alive[t]

        loss = (policy_loss + VALUE_COEF * value_loss
                - entropy_coef * entropy_sum + anchor * anchor_loss) / steps
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD)
        optimizer.step()
        losses.append(float(loss.detach()))

    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--envs", type=int, default=48)
    parser.add_argument("--workers", type=int, default=0,
                        help="환경을 굴릴 프로세스 수. 0이면 자동")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--entropy", type=float, default=0.004)
    parser.add_argument("--anchor", type=float, default=ANCHOR_COEF,
                        help="모방학습 정책을 붙드는 세기. 0이면 끔")
    # 학습은 빔을 줄여서 빠르게, 실차는 1081빔 그대로.
    # 구역 대표 거리를 하위 10퍼센타일로 뽑아서 해상도에 거의 안 휘둘린다.
    # 검증 결과 541빔이면 특징 차이가 99퍼센타일에서 0.12시그마로 무시할 만하고
    # 시뮬이 3배 빠르다. (108빔은 7배 빠르지만 차이가 0.66시그마로 커진다)
    parser.add_argument("--beams", type=int, default=541)
    parser.add_argument("--lidar-range", type=float, default=LIDAR_RANGE)
    parser.add_argument("--init", type=str, default="fly_model.pt")
    parser.add_argument("--out", type=str, default="fly_racer.pt")
    parser.add_argument("--dense", action="store_true",
                        help="초파리 배선 대신 전부 연결 (B안 비교군)")
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    # 환경을 프로세스로 나눠 돌리므로 토치는 남은 코어만 쓴다
    # 환경이 프로세스 8개를 쓰므로 토치는 2개면 충분하다.
    # 측정: 48환경/8프로세스/토치2 에서 991 스텝/초 (단일 프로세스 대비 5.3배)
    torch.set_num_threads(2)
    log = Logger()

    log("=" * 68)
    log("2단계 강화학습 — 교사를 넘어 랩타임 줄이기")
    log("=" * 68)

    driver = FlyDriver(n_beams=args.beams, lidar_range=args.lidar_range,
                       fly_wiring=not args.dense)
    if Path(args.init).exists():
        driver.load(args.init)
        log(f"출발점: {args.init} (모방학습 결과)")
    else:
        log("출발점 없음. 무작위에서 시작한다 (오래 걸린다)")
    stats = (driver.encoder.mean, driver.encoder.std)
    if stats[0] is None:
        raise SystemExit("정규화 통계가 없다. 모방학습을 먼저 돌려라.")

    policy = RacingPolicy(driver.model, len(driver.channels))
    optimizer = torch.optim.Adam([
        {"params": policy.policy_parameters(), "lr": args.lr},
        {"params": policy.critic.parameters(), "lr": args.lr * 3},
    ])

    # 모방학습 시점의 정책을 얼려서 따로 들고 있는다.
    # 학습 중 여기서 너무 멀어지면 되돌리는 힘으로 쓴다.
    anchor_model = None
    if args.anchor > 0.0:
        anchor_model = copy.deepcopy(driver.model).eval()
        for param in anchor_model.parameters():
            param.requires_grad_(False)

        def anchor_teacher(x, hidden):
            return anchor_model(x, hidden)

        log(f"모방학습 붙들기 계수 {args.anchor} "
            f"(전체의 {ANCHOR_DECAY:.0%} 지점까지 서서히 풀림)")
    else:
        anchor_teacher = None

    workers = args.workers or min(8, max(1, os.cpu_count() // 2))
    runner = ParallelRunner(args.envs, args.beams, args.lidar_range,
                            workers=workers, randomize=not args.no_noise)
    log(f"동시에 굴리는 트랙 {args.envs}개 "
        f"(프로세스 {workers}개, 토치 스레드 {torch.get_num_threads()}개)")
    log(f"한 번에 {args.steps}스텝")
    log(f"센서 손상 {'끔' if args.no_noise else '켬 (도메인 랜덤화)'}")
    log(f"수집량 {args.iters * args.envs * args.steps:,}스텝\n")

    hidden = None
    best = -np.inf
    bar = Progress(args.iters, "강화", log)
    history = []

    for it in range(1, args.iters + 1):
        batch = rollout(runner, policy, hidden, stats, args.steps)
        hidden = batch["end_hidden"].detach()
        pull = 0.0
        if args.anchor > 0.0:
            pull = anchor_weight(it, args.iters) * (args.anchor / ANCHOR_COEF)
        loss = update(policy, optimizer, batch, args.epochs, args.entropy,
                      anchor=pull, teacher=anchor_teacher)

        finished = runner.drain()
        history.extend(finished)
        history = history[-60:]
        note = f"보상 {batch['rewards'].sum(0).mean():>6.1f}"
        if history:
            laps = [h for h in history if not h["crashed"]]
            rate = len(laps) / len(history)
            speed = np.mean([h["length"] / h["lap_time"] for h in laps]) if laps else 0
            note += f"  완주 {rate:>5.0%}  평균속도 {speed:>4.1f}m/s"
            score = rate * 100 + speed
            if score > best and rate > 0.5:
                best = score
                torch.save({
                    "model": driver.model.state_dict(),
                    "mean": driver.encoder.mean, "std": driver.encoder.std,
                    "lidar_range": args.lidar_range, "dt": CONTROL_DT,
                    "log_std": policy.log_std.detach(),
                    "trained_beams": args.beams,
                    "iteration": it,
                }, args.out)
                note += " 저장"
        note += f"  흔들림 {policy.log_std.exp().mean():.2f}"
        if pull > 0:
            note += f"  붙듦 {pull:.3f}"
        bar.step(it, note)
    bar.done()

    log(f"\n-> {args.out} 저장됨")
    log("test_on_map.py 로 실제 대회 코스에서 확인하세요.")
    log.close()


if __name__ == "__main__":
    main()
