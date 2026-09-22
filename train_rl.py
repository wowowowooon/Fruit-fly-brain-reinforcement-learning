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

from f1tenth_sim import (CONTROL_DT, CRASH_PENALTY, LIDAR_BEAMS,
                         LIDAR_RANGE, MARGIN_PENALTY, PASSAGE_MIN,
                         SHAKE_PENALTY, F1TenthEnv)
from paths import model as model_path
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


# CAPS 공간적 항에서 관측을 얼마나 흔들어 볼지.
# 정규화된 관측 기준이라 0.1 이면 1/10 시그마만큼 흔드는 것이다.
CAPS_SIGMA = 0.1


def gae(rewards, values, dones, last_value, gamma=GAMMA, lam=LAMBDA):
    """지금 행동이 평균보다 얼마나 좋았는지 (이득)."""
    steps = len(rewards)
    advantages = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(steps)):
        nxt = last_value if t == steps - 1 else values[t + 1]
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * nxt * alive - values[t]
        running = delta + gamma * lam * alive * running
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
           hold_polarity=None, teacher=None,
           caps_time=0.0, caps_space=0.0, caps_sigma=CAPS_SIGMA,
           clip=CLIP, gamma=GAMMA, lam=LAMBDA):
    """모은 조각으로 정책을 민다. 시간 순서를 지켜야 기억이 맞는다.

    CAPS (Mysore et al., ICRA 2021) 두 항을 같이 민다. 보상에 벌점으로
    넣으면 다른 항에 묻히지만(흔들림 벌점 3배를 올려도 떨림이 2% 밖에
    안 줄었다), 손실에 직접 걸면 정책이 바로 밀린다.

      시간적  연속한 두 스텝의 행동이 비슷하게
      공간적  비슷한 상태면 비슷한 행동을 내게

    공간적 항이 핵심이다. 우리는 학습 중 라이다에 잡음을 넣는데(도메인
    랜덤화) 정책이 그 잡음에 그대로 반응하면 조향 떨림이 된다. 관측을
    조금 흔들어도 행동이 안 바뀌게 만드는 것이 바로 그 처방이다.

    논문 보고: 같은 PPO 에서 떨림 지표 43.9 -> 5.92 (86% 감소).
    단, 1/18 스케일 레이싱카 연구(I-RAS)는 CAPS 를 세게 걸면 정책이
    지나치게 뻣뻣해져 코너를 못 돈다고 지적한다. 세기는 재서 고른다.
    """
    advantages, returns = gae(
        batch["rewards"], batch["values"], batch["dones"], batch["last_value"],
        gamma=gamma, lam=lam,
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
        time_loss = space_loss = 0.0
        previous = None
        for t in range(steps):
            before = hidden
            logp, entropy, value, mean, hidden = policy.evaluate(
                obs[t].unsqueeze(1), hidden, acts[t].unsqueeze(1)
            )
            if caps_time > 0.0 and previous is not None:
                # 에피소드가 끊긴 자리는 빼야 한다. 다른 트랙의 첫 행동과
                # 이전 트랙의 마지막 행동을 비교하면 뜻이 없다.
                keep = alive[t - 1]
                time_loss = time_loss + (
                    (mean[:, 0] - previous) * keep).pow(2).mean()
            if caps_space > 0.0:
                noisy = obs[t] + torch.randn_like(obs[t]) * caps_sigma
                near, _, _ = policy(noisy.unsqueeze(1), before)
                space_loss = space_loss + (
                    mean[:, 0] - near[:, 0]).pow(2).mean()
            previous = mean[:, 0]
            ratio = (logp[:, 0] - old_logps[t]).exp()
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
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
                - entropy_coef * entropy_sum + anchor * anchor_loss
                + caps_time * time_loss + caps_space * space_loss) / steps
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD)
        optimizer.step()
        # 초파리 극성은 학습으로 뒤집히면 안 된다
        if hold_polarity is not None:
            hold_polarity()
        losses.append(float(loss.detach()))

    return float(np.mean(losses))


# 모델 선택에만 쓰는 트랙. 학습(0~399), 모방학습 시험(600~604),
# 최종 평가(700~744) 어디와도 안 겹친다.
VALID_SEEDS = tuple(range(500, 512))


def validate(driver, args, seeds=VALID_SEEDS):
    """고정된 트랙에서 탐색 없이 달려 본다. 이 점수로만 모델을 고른다."""
    from f1tenth_sim import F1TenthEnv
    done, laps, lengths = 0, [], []
    for seed in seeds:
        env = F1TenthEnv(n_beams=args.beams, lidar_range=args.lidar_range,
                         track_seed=seed, motion=args.motion, friction=1.0)
        obs, _ = env.reset(seed=seed)
        driver.reset()
        info = {"progress": 0.0}
        for step in range(3000):
            obs, _, crashed, timeout, info = env.step(driver.act(obs))
            if crashed or timeout:
                break
        if info["progress"] > 0.97:
            done += 1
            laps.append((step + 1) * env.dt)
            lengths.append(env.lap_length)
    rate = done / len(seeds)
    speed = float(np.mean([l / t for l, t in zip(lengths, laps)])) if laps else 0.0
    return rate, speed


def _snapshot(driver, policy, args, iteration, name):
    """가중치를 저장한다.

    paths.model 을 거치지 않으면 실행한 폴더에 떨어진다. 모델이 models/
    안에 있는 줄 알고 찾다가 '학습 전' 이라는 말을 듣게 된다.
    """
    torch.save({
        "model": driver.model.state_dict(),
        "mean": driver.encoder.mean, "std": driver.encoder.std,
        "lidar_range": args.lidar_range, "dt": CONTROL_DT,
        "log_std": policy.log_std.detach(),
        "trained_beams": args.beams,
        "motion": driver.motion,
        "fly_wiring": driver.fly_wiring,
        "iteration": iteration,
        # 이어서 학습할 때 쓸 것들
        "critic": policy.critic.state_dict(),
    }, model_path(name))


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
    # 레이싱용 재학습에서 쓴다. B·C 비교는 둘 다 기본값(0, 1.45)으로 했다.
    parser.add_argument("--rivals", type=float, default=0.0,
                        help="앞서 달리는 상대 차를 넣을 확률 (0~1)")
    parser.add_argument("--passage", type=float, default=PASSAGE_MIN,
                        help="장애물 옆에 남길 통로 [m]. 좁히면 좁은 트랙에도 "
                             "장애물이 생기고 회피가 필수가 된다")
    parser.add_argument("--crash-penalty", type=float, default=CRASH_PENALTY,
                        help="충돌 벌점. 한 바퀴 전진 보상이 100~250 이라 "
                             "20 은 너무 싸다")
    parser.add_argument("--shake-penalty", type=float, default=SHAKE_PENALTY,
                        help="조향을 급하게 꺾을 때의 벌점 (0~1 정규화)")
    parser.add_argument("--margin-penalty", type=float, default=MARGIN_PENALTY,
                        help="벽·장애물에 바짝 붙었을 때의 벌점 (0~1 정규화)")
    parser.add_argument("--multiply-penalty", action="store_true",
                        help="벌점을 전진 보상에 곱한다. 더하면 묻힌다")
    parser.add_argument("--crash-speed", type=float, default=0.0,
                        help="충돌 벌점에 속도 제곱을 이만큼 곱해 더한다")
    parser.add_argument("--motion", type=int, default=3,
                        help="관측의 운동 채널 수. 6 이면 조향각과 직전 행동 포함")
    parser.add_argument("--caps-time", type=float, default=0.0,
                        help="CAPS 시간적 항 — 연속한 두 스텝의 행동 차이를 줄인다")
    parser.add_argument("--caps-space", type=float, default=0.0,
                        help="CAPS 공간적 항 — 비슷한 상태면 비슷한 행동을 내게")
    parser.add_argument("--caps-sigma", type=float, default=CAPS_SIGMA,
                        help="공간적 항에서 관측을 흔들 크기")
    parser.add_argument("--real-tracks", type=float, default=0.0,
                        help="실제 서킷을 쓸 확률 (0~1)")
    parser.add_argument("--margin-start", type=float, default=0.20,
                        help="벽에서 이만큼 안쪽부터 벌점 [m]")
    parser.add_argument("--margin-power", type=float, default=1.0,
                        help="2 면 제곱. 0 근처에서 훨씬 급해진다")
    parser.add_argument("--narrow", type=float, default=0.0,
                        help="대회를 닮은 좁은 지그재그 트랙을 쓸 확률 (0~1)")
    parser.add_argument("--valid-every", type=int, default=50,
                        help="몇 회마다 고정 검증 트랙에서 재고 모델을 고를지")
    parser.add_argument("--clip", type=float, default=CLIP)
    parser.add_argument("--gae", type=float, default=LAMBDA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
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
                       fly_wiring=not args.dense, motion=args.motion)
    if model_path(args.init).exists():
        driver.load(args.init)
        log(f"출발점: {args.init} (모방학습 결과)")
    else:
        log("출발점 없음. 무작위에서 시작한다 (오래 걸린다)")
    stats = (driver.encoder.mean, driver.encoder.std)
    if stats[0] is None:
        raise SystemExit("정규화 통계가 없다. 모방학습을 먼저 돌려라.")

    policy = RacingPolicy(driver.model, len(driver.channels))
    # 강화학습 결과에서 이어 가는 경우, 비평가와 탐색 폭도 이어받는다.
    # 비평가를 새로 만들면 처음 수백 회 동안 학습 방향이 엉뚱해진다.
    resumed = model_path(args.init)
    if resumed.exists():
        saved = torch.load(resumed, map_location="cpu", weights_only=False)
        if "critic" in saved:
            policy.critic.load_state_dict(saved["critic"])
            with torch.no_grad():
                policy.log_std.copy_(saved["log_std"])
            log(f"비평가와 탐색 폭도 이어받았다 "
                f"(저장된 반복 {saved.get('iteration')})")
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
                            workers=workers, randomize=not args.no_noise,
                            rivals=args.rivals, passage=args.passage,
                            motion=args.motion, real_tracks=args.real_tracks,
                            narrow=args.narrow,
                            reward={
                                "crash_penalty": args.crash_penalty,
                                "shake_penalty": args.shake_penalty,
                                "margin_penalty": args.margin_penalty,
                                "multiply_penalty": args.multiply_penalty,
                                "crash_speed": args.crash_speed,
                                "margin_start": args.margin_start,
                                "margin_power": args.margin_power,
                            })
    log(f"동시에 굴리는 트랙 {args.envs}개 "
        f"(프로세스 {workers}개, 토치 스레드 {torch.get_num_threads()}개)")
    log(f"한 번에 {args.steps}스텝")
    log(f"센서 손상 {'끔' if args.no_noise else '켬 (도메인 랜덤화)'}")
    log(f"장애물 옆 통로 {args.passage:.2f}m")
    if args.caps_time or args.caps_space:
        log(f"CAPS  시간적 {args.caps_time:g}  공간적 {args.caps_space:g}"
            f"  (관측 흔들기 {args.caps_sigma:g})")
    if args.real_tracks or args.narrow:
        log(f"트랙 섞기  실제 서킷 {args.real_tracks:.0%}  "
            f"지그재그 {args.narrow:.0%}  나머지는 생성 트랙")
    log(f"보상  {'곱셈' if args.multiply_penalty else '덧셈'} 합성. "
        f"전진거리 + 충돌 -{args.crash_penalty:g}"
        f"{f' + 흔들림 -{args.shake_penalty:g}' if args.shake_penalty else ''}"
        f"{f' + 여유 -{args.margin_penalty:g}' if args.margin_penalty else ''}")
    log(f"움직이는 상대 차 "
        f"{'없음' if args.rivals <= 0 else f'{args.rivals:.0%} 확률로 1~2대'}")
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
                      anchor=pull, teacher=anchor_teacher,
                      hold_polarity=driver.hold_polarity,
                      caps_time=args.caps_time, caps_space=args.caps_space,
                      caps_sigma=args.caps_sigma,
                      clip=args.clip, gamma=args.gamma, lam=args.gae)

        finished = runner.drain()
        history.extend(finished)
        history = history[-60:]
        note = f"보상 {batch['rewards'].sum(0).mean():>6.1f}"
        if history:
            laps = [h for h in history if not h["crashed"]]
            rate = len(laps) / len(history)
            speed = np.mean([h["length"] / h["lap_time"] for h in laps]) if laps else 0
            note += f"  완주 {rate:>5.0%}  평균속도 {speed:>4.1f}m/s"
        # 모델은 '고정 검증 트랙' 점수로만 고른다. 학습 중 이동평균은
        # 흔들려서 운 좋은 구간이 박제된다.
        if it % args.valid_every == 0 or it == args.iters:
            driver.model.eval()
            v_rate, v_speed = validate(driver, args)
            driver.model.train()
            score = v_rate * 100 + v_speed
            note += f"  검증 {v_rate:>4.0%} {v_speed:.1f}m/s"
            if score > best:
                best = score
                _snapshot(driver, policy, args, it, args.out)
                note += " 저장"
        note += f"  흔들림 {policy.log_std.exp().mean():.2f}"
        if pull > 0:
            note += f"  붙듦 {pull:.3f}"
        bar.step(it, note)
    bar.done()
    # 마지막 정책도 따로 남긴다. '최고' 는 최근 60판의 완주율로 고르는데
    # 그 창이 흔들려서, 운 좋은 한 구간이 뽑히고 마지막 정책은 사라진다.
    # 둘 다 있어야 재서 고를 수 있다.
    last = Path(args.out)
    _snapshot(driver, policy, args, args.iters,
              f"{last.stem}_last{last.suffix}")
    log(f"마지막 정책은 {last.stem}_last{last.suffix} 에 따로 저장했다")

    log(f"\n-> {args.out} 저장됨")
    log("test_on_map.py 로 실제 대회 코스에서 확인하세요.")
    log.close()


if __name__ == "__main__":
    main()
