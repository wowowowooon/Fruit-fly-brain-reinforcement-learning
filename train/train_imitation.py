"""레이싱 라인을 교사로 삼아 초파리 회로에 운전을 가르친다.

교사는 팀 실차 방식(최소곡률 레이싱 라인 + Stanley)이다. 자기 위치와 트랙
모양을 아는 상태에서 가장 빠른 선을 그린다. 학생인 초파리 회로는 라이다만
보고 그 결과를 흉내 낸다. 아는 게 많은 교사를 모르는 학생이 따라 하는 방식이다.

FGM(그때그때 뚫린 쪽으로 가기)을 교사로 쓰면 트랙 한가운데로만 달려서
급조향을 배울 일이 없다. 실제 코스는 최대 조향이 필요한 구간이 6%나 된다.

배선은 초파리 것 그대로 두고, 신호 세기만 조정한다.
교사 주행 중 일부러 흔들어서, 교사가 잘 안 가는 상황도 겪게 한다.
"""

import sys
from pathlib import Path

# 이 파일은 하위 폴더에 있다. flyracer 패키지를 찾으려면
# 저장소 뿌리를 경로에 넣어 줘야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flyracer.sim.f1tenth_sim import (CONTROL_DT, LIDAR_BEAMS, LIDAR_RANGE, PASSAGE_MIN,
                         F1TenthEnv)
from flyracer.brain.fly_driver import N_MOTION, FlyDriver, LidarEncoder
from flyracer.sim.moving_obstacles import TRAIN_RIVAL_PACE, MovingObstacleEnv
from flyracer.sim.real_tracks import track_names
from flyracer.progress import Logger, Progress
from flyracer.drive.raceline import StanleyRacer
from flyracer.drive.racing_teacher import RacingTeacher

SEQ_LEN = 64          # RNN에 한 번에 넣는 시간 길이 (40Hz에서 1.6초)
# 교사 행동에 섞는 흔들림. 학생이 교사가 안 가는 자리도 겪게 하려는 것이다.
# 0.25 는 조향 난수가 10도나 되어 교사 자신이 45개 중 6개에서 벽으로 나간다.
# 시드 700~744 에서 재 보고 골랐다 (쓸 수 있는 주행 시간 기준):
#   0.25 -> 교사 39/45, 747초   0.15 -> 44/45, 888초
#   0.10 -> 45/45, 880초        0.05 -> 45/45, 869초
NOISE = 0.15
EPISODE_STEPS = 2600  # 40Hz에서 65초. 트랙당 여러 바퀴를 돈다
COLLECT_LAPS = 2      # 한 트랙에서 도는 바퀴 수. 레이싱 라인 계산이 비싸서
                      # 같은 트랙을 더 활용하는 편이 싸게 먹힌다


def make_env(args, laps=1):
    """상대 차를 쓸 때만 움직이는 환경. 안 쓰면 예전과 똑같다."""
    common = dict(n_beams=args.beams, dt=CONTROL_DT,
                  lidar_range=args.lidar_range, laps=laps,
                  passage=getattr(args, "passage", PASSAGE_MIN),
                  motion=getattr(args, "motion", 3))
    rivals = getattr(args, "rivals", 0)
    if rivals > 0:
        return MovingObstacleEnv(rivals=rivals,
                                 rival_pace=TRAIN_RIVAL_PACE, **common)
    return F1TenthEnv(**common)


def make_teacher(args, env):
    """레이싱 교사와 예전 교사 중 하나.

    예전 교사(raceline.StanleyRacer)는 라인이 장애물을 뚫고 지나가고
    직선에서 속도를 못 낸다. B·C 비교는 둘 다 그걸로 배웠으므로 그대로
    두고, 레이싱용 재학습부터 새 교사를 쓴다.
    """
    if args.teacher == "legacy":
        return StanleyRacer.for_env(env)
    return RacingTeacher.for_env(env)


def collect(args, rng, log=print):
    """교사 주행을 기록한다. 기록하는 정답은 흔들리기 전 값.

    교사가 충돌한 에피소드는 버린다. 잘못된 시범을 배우면 독이 된다.
    """
    env = make_env(args, laps=COLLECT_LAPS)
    encoder = LidarEncoder(args.beams, lidar_range=args.lidar_range)
    episodes = []
    circuits = track_names()[:getattr(args, "circuits", 0)]
    narrow = int(getattr(args, "narrow", 0))
    bar = Progress(args.tracks + len(circuits) + narrow, "수집", log)
    dropped = 0

    for track in range(args.tracks + len(circuits) + narrow):
        # 생성 트랙 -> 실제 서킷 -> 대회를 닮은 지그재그, 순서대로 돈다.
        in_circuit = args.tracks <= track < args.tracks + len(circuits)
        env.circuit = circuits[track - args.tracks] if in_circuit else None
        env.narrow = track >= args.tracks + len(circuits)
        seed = track if track < args.tracks else (
            900 + track - args.tracks - len(circuits) if env.narrow else 0)
        obs, _ = env.reset(seed=seed)
        encoder.reset()
        # 교사는 차를 레이싱 라인 위로 옮긴다. 옮긴 뒤의 관측을 다시 읽어야 한다.
        racer = make_teacher(args, env)
        obs = env._observe(env._scan())
        features, labels = [], []
        info = {"crashed": False}

        for _ in range(EPISODE_STEPS):
            scan = obs[:-env.motion] * env.lidar_range
            action = (racer.act(env.pos, env.yaw, env.speed,
                                env.rival_states(), env.dt)
                      if hasattr(env, "rival_states")
                      else racer.act(env.pos, env.yaw, env.speed))
            features.append(encoder.raw(scan, obs[-env.motion:], env.dt))
            labels.append(action.copy())

            noisy = np.clip(action + rng.normal(0, NOISE, 2), -1, 1)
            obs, _, done, truncated, info = env.step(noisy)
            if done or truncated:
                break

        # 충돌했으면 마지막 2초는 버린다. 그 구간이 잘못된 시범이다.
        if info["crashed"]:
            cut = max(0, len(features) - 80)
            features, labels = features[:cut], labels[:cut]
        if len(features) < SEQ_LEN:
            dropped += 1
        else:
            episodes.append((
                np.array(features, dtype=np.float32),
                np.array(labels, dtype=np.float32),
            ))
        steps = sum(len(f) for f, _ in episodes)
        bar.step(track + 1, f"{steps:>6}스텝  버림 {dropped}")

    return episodes


# ---- 모은 자료를 파일로 두기 -------------------------------------------
# 레이싱 라인을 푸는 데 트랙당 1.6초가 걸린다. 트랙 260개면 그것만 7분이고
# 시범 주행까지 하면 훨씬 길다. 한 번 모아 두면 계수를 바꿔 가며 다시
# 학습할 때마다 그 시간을 통째로 아낀다.


def _teacher_fingerprint():
    """교사 코드의 지문.

    교사를 고쳐 놓고 예전에 모은 자료를 그대로 쓰면, 고친 줄 알면서
    낡은 시범으로 학습하게 된다. 조용히 틀리는 종류의 사고라 코드 내용을
    해시에 넣어 자동으로 걸러지게 한다.
    """
    # 교사 코드가 흩어져 있다. 옮길 때마다 여기도 같이 짚어 줘야 한다.
    root = Path(__file__).resolve().parent.parent
    blob = b""
    for name in ("flyracer/drive/racing_teacher.py",
                 "flyracer/drive/raceline.py",
                 "flyracer/sim/moving_obstacles.py"):
        blob += (root / name).read_bytes()
    return hashlib.sha256(blob).hexdigest()[:16]


def _recipe(args):
    """이 자료가 어떤 조건에서 모은 것인지. 하나라도 다르면 다시 모아야 한다."""
    return {
        "teacher_code": _teacher_fingerprint(),
        "tracks": int(args.tracks),
        "beams": int(args.beams),
        "lidar_range": float(args.lidar_range),
        "teacher": str(args.teacher),
        "rivals": int(getattr(args, "rivals", 0)),
        "passage": float(getattr(args, "passage", PASSAGE_MIN)),
        "laps": int(COLLECT_LAPS),
        "pace": list(TRAIN_RIVAL_PACE),
        "motion": int(getattr(args, "motion", 3)),
        "circuits": int(getattr(args, "circuits", 0)),
        "narrow": int(getattr(args, "narrow", 0)),
        "noise": float(NOISE),
    }


def save_episodes(path, episodes, recipe):
    """길이가 제각각인 에피소드를 이어 붙이고 경계만 따로 적는다."""
    lengths = np.array([len(f) for f, _ in episodes], dtype=np.int64)
    np.savez_compressed(
        path,
        features=np.concatenate([f for f, _ in episodes]),
        labels=np.concatenate([l for _, l in episodes]),
        lengths=lengths,
        recipe=np.array(json.dumps(recipe, sort_keys=True)),
    )


def load_episodes(path, recipe):
    """조건이 같을 때만 돌려준다. 다르면 None."""
    if not Path(path).exists():
        return None
    data = np.load(path, allow_pickle=False)
    stored = json.loads(str(data["recipe"]))
    if stored != recipe:
        return None
    edges = np.concatenate([[0], np.cumsum(data["lengths"])])
    return [(data["features"][a:b], data["labels"][a:b])
            for a, b in zip(edges[:-1], edges[1:])]


def dagger(driver, args, rng, tracks, start, log=print):
    """학생을 직접 달리게 하고, 그 자리에서 교사가 뭐라 할지를 정답으로 적는다.

    흔들림을 섞는 방식은 '교사가 안 가는 자리' 를 무작위로 훑는 것이라,
    학생이 실제로 가는 자리와는 다르다. 학생이 조금 어긋난 채로 코너에
    들어가면 교사는 그런 자리에 가 본 적이 없어서 시범이 없다. 그래서
    학생을 운전대에 앉히고 교사는 옆에서 '나라면 이렇게 한다' 만 적는다.

    Ross et al., DAgger (2011). 강화학습이 어차피 학생 분포에서 도는데도
    이걸 먼저 하는 이유는, 출발점이 좋을수록 강화학습이 덜 헤매기 때문이다.
    """
    env = make_env(args, laps=COLLECT_LAPS)
    encoder = LidarEncoder(args.beams, lidar_range=args.lidar_range)
    episodes = []
    bar = Progress(tracks, "DAgger", log)
    crashed_n = 0

    for index in range(tracks):
        track = start + index
        obs, _ = env.reset(seed=track)
        racer = make_teacher(args, env)      # 교사가 차를 라인 위에 올린다
        obs = env._observe(env._scan())
        encoder.reset()
        driver.reset()
        features, labels = [], []
        info = {"crashed": False}

        for _ in range(EPISODE_STEPS):
            scan = obs[:-env.motion] * env.lidar_range
            features.append(encoder.raw(scan, obs[-env.motion:], env.dt))
            labels.append((racer.act(env.pos, env.yaw, env.speed,
                                     env.rival_states(), env.dt)
                           if hasattr(env, "rival_states")
                           else racer.act(env.pos, env.yaw, env.speed)).copy())
            obs, _, done, truncated, info = env.step(driver.act(obs))
            if done or truncated:
                break

        if info["crashed"]:
            crashed_n += 1
            cut = max(0, len(features) - 80)   # 충돌 직전 2초는 버린다
            features, labels = features[:cut], labels[:cut]
        if len(features) >= SEQ_LEN:
            episodes.append((np.array(features, dtype=np.float32),
                             np.array(labels, dtype=np.float32)))
        bar.step(index + 1,
                 f"{sum(len(f) for f, _ in episodes):>6}스텝  학생충돌 {crashed_n}")
    bar.done()
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
    parser.add_argument("--tracks", type=int, default=130)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--beams", type=int, default=541)
    parser.add_argument("--lidar-range", type=float, default=LIDAR_RANGE)
    parser.add_argument("--eval-every", type=int, default=3)
    # 레이싱용 재학습에서 쓴다. B·C 비교는 둘 다 legacy 로 배웠다.
    parser.add_argument("--teacher", choices=("racing", "legacy"),
                        default="legacy",
                        help="racing = 장애물 비켜 가고 직선에서 최고속을 내는 "
                             "새 교사. legacy = 예전 교사")
    parser.add_argument("--rivals", type=int, default=0,
                        help="앞서 달리는 상대 차 수. 0이면 정지 장애물만")
    parser.add_argument("--passage", type=float, default=PASSAGE_MIN,
                        help="장애물 옆에 남길 통로 [m]")
    parser.add_argument("--dagger", type=int, default=0,
                        help="학생을 달리게 해서 교사 정답을 붙이는 라운드 수")
    parser.add_argument("--dagger-tracks", type=int, default=60,
                        help="DAgger 한 라운드에 쓸 트랙 수")
    parser.add_argument("--motion", type=int, default=3,
                        help="관측의 운동 채널 수. 6 이면 조향각과 직전 행동 포함")
    parser.add_argument("--circuits", type=int, default=0,
                        help="실제 서킷을 몇 개나 수집에 넣을지 (0~23)")
    parser.add_argument("--narrow", type=int, default=0,
                        help="대회를 닮은 지그재그 트랙을 몇 개나 넣을지")
    parser.add_argument("--cache", type=str, default=None,
                        help="교사 시범을 모아 둘 파일(.npz). 조건이 같으면 "
                             "다음부터 수집을 건너뛴다")
    parser.add_argument("--out", type=str, default="fly_model.pt")
    parser.add_argument("--dense", action="store_true",
                        help="초파리 배선 대신 전부 연결 (B안 비교군)")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    log = Logger()

    log("=" * 68)
    # 교사는 최소곡률 라인을 풀지만, 꺾임이 생기면 중심선으로 되돌린다.
    # 실제로 트랙 75% 에서 중심선을 탄다. '레이싱 라인' 이라고 적어 두면
    # 안 하고 있는 것을 하고 있다고 읽게 된다.
    log("1단계 모방학습 — 교사 따라하기 (강화학습의 출발점)")
    log("=" * 68)
    log(f"하드웨어  라이다 {args.beams}빔 / {args.lidar_range:.0f}m / "
        f"{1 / CONTROL_DT:.0f}Hz  (ROBORACER 규정 사양)")
    log(f"계획      트랙 {args.tracks}개 수집 -> {args.epochs}에폭 학습\n")

    recipe = _recipe(args)
    episodes = load_episodes(args.cache, recipe) if args.cache else None
    if episodes is None:
        episodes = collect(args, rng, log)
        if args.cache:
            save_episodes(args.cache, episodes, recipe)
            log(f"모은 자료를 {args.cache} 에 저장했다")
    else:
        log(f"{args.cache} 에서 모아 둔 자료를 불러왔다 (수집 건너뜀)")
    samples = sum(len(f) for f, _ in episodes)
    log(f"에피소드 {len(episodes)}개, 총 {samples}스텝\n")

    # 입력 값의 폭을 맞춰준다. 안 하면 신호가 죽어서 학습이 안 된다.
    stacked = np.concatenate([f for f, _ in episodes])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0)
    episodes = [((f - mean) / np.maximum(std, 1e-3), y) for f, y in episodes]

    baseline = np.concatenate([y for _, y in episodes]).var(axis=0).mean()
    log(f"상수로 찍었을 때 오차 {baseline:.4f} (이보다 낮아야 배운 것)\n")

    driver = FlyDriver(n_beams=args.beams, lidar_range=args.lidar_range,
                       fly_wiring=not args.dense, motion=args.motion)
    driver.encoder.set_stats(mean, std)
    log(driver.summary())
    log("")

    best = run_training(driver, episodes, args, rng, log, -1.0, "학습")

    # DAgger: 학생을 달리게 하고 교사가 정답을 적어 자료를 보탠다.
    for turn in range(1, args.dagger + 1):
        log(f"\nDAgger {turn}회차 — 학생이 운전, 교사가 정답을 적는다")
        extra = dagger(driver, args, rng, args.dagger_tracks,
                       args.tracks + (turn - 1) * args.dagger_tracks, log)
        # 정규화는 처음 것을 그대로 쓴다. 바꾸면 이미 배운 가중치가 어긋난다.
        extra = [((f - mean) / np.maximum(std, 1e-3), y) for f, y in extra]
        episodes = episodes + extra
        log(f"자료 {len(episodes)}편, 총 "
            f"{sum(len(f) for f, _ in episodes):,}스텝\n")
        best = run_training(driver, episodes, args, rng, log, best,
                            f"학습{turn + 1}")

    log(f"\n최고 진행도 {best:.1%}  -> {args.out} 저장됨")
    log("다음: 강화학습으로 이 모델을 출발점 삼아 랩타임을 줄인다.")
    log.close()


def run_training(driver, episodes, args, rng, log, best, tag):
    """한 묶음의 자료로 학습한다. 더 좋아질 때만 저장한다."""
    optimizer = torch.optim.Adam(driver.model.parameters(), lr=args.lr)
    # 뒤로 갈수록 보폭을 줄여서 좋은 지점을 지나치지 않게 한다.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fn = nn.MSELoss()
    # 학습에 쓴 트랙(0~399)과 겹치면 안 된다. 100~104 는 수집 범위 안이라,
    # 본 트랙에서 잘한 모델을 고르고 있었다. 최종 평가(700~744)와도 겹치면
    # 안 되므로 그 사이를 쓴다.
    eval_seeds = [600, 601, 602, 603, 604]

    bar = Progress(args.epochs, tag, log)
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
            # 초파리 극성은 학습으로 뒤집히면 안 된다
            driver.hold_polarity()
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
    return best


if __name__ == "__main__":
    main()
