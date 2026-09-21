"""여러 트랙을 여러 CPU 코어에서 동시에 굴린다.

강화학습 시간의 90%가 시뮬레이션이고, 그게 파이썬 루프라 코어 하나만 쓴다.
환경을 늘려도 처리량이 그대로인 이유다. 프로세스를 나눠 코어를 다 쓰게 한다.

주고받는 건 라이다 원본(1081개)이 아니라 이미 37채널로 줄인 값이다.
프로세스 사이로 옮기는 양이 30배 적어서 통신 비용이 거의 안 든다.
"""

import multiprocessing as mp

import numpy as np

from f1tenth_sim import PASSAGE_MIN, F1TenthEnv
from fly_driver import N_MOTION, FEATURE_LIMIT, LidarEncoder
from moving_obstacles import MovingObstacleEnv
from sensor_noise import DomainRandomizer

TRACK_POOL = 400          # 이만큼의 서로 다른 트랙에서 뽑아 쓴다
RIVAL_COUNT = (1, 3)      # 상대 차를 넣을 때 몇 대까지


class _Slot:
    """작업자 하나가 맡는 환경 하나. 환경 + 인코더 + 센서 손상."""

    def __init__(self, beams, lidar_range, seed, randomize, rivals=0.0,
                 passage=PASSAGE_MIN, reward=None):
        # 상대 차를 쓸 때만 움직이는 환경을 쓴다. 안 쓰면 예전과 똑같다.
        maker = MovingObstacleEnv if rivals > 0.0 else F1TenthEnv
        self.env = maker(n_beams=beams, lidar_range=lidar_range,
                         passage=passage, **(reward or {}))
        self.encoder = LidarEncoder(beams, lidar_range=lidar_range)
        self.rng = np.random.default_rng(seed)
        self.rivals = float(rivals)
        if rivals > 0.0:
            self.env.rival_rng = self.rng
        self.fault = DomainRandomizer(
            self.env.beam_angles, lidar_range, np.random.default_rng(seed + 7)
        ) if randomize else None
        self.reset()

    def reset(self):
        if self.rivals > 0.0:
            # 매 판 상대 차 수를 새로 뽑는다. 없는 판도 섞여야 한다.
            self.env.rival_count = (
                int(self.rng.integers(*RIVAL_COUNT))
                if self.rng.random() < self.rivals else 0)
        raw, _ = self.env.reset(seed=int(self.rng.integers(TRACK_POOL)))
        self.encoder.reset()
        if self.fault is not None:
            self.fault.reset()
            raw = self.fault(raw)
        return self._features(raw)

    def _features(self, raw):
        return self.encoder.raw(
            raw[:-N_MOTION] * self.env.lidar_range,
            raw[-N_MOTION:], self.env.dt)

    def step(self, action):
        raw, reward, done, truncated, info = self.env.step(action)
        if self.fault is not None:
            raw = self.fault(raw)
        features = self._features(raw)
        finished = None
        if done or truncated:
            finished = {
                "progress": info["progress"],
                "crashed": info["crashed"],
                "lap_time": info["lap_time"],
                "length": self.env.lap_length,
            }
            features = self.reset()
        return features, reward, float(done or truncated), finished


def _worker(pipe, beams, lidar_range, seeds, randomize, rivals=0.0,
            passage=PASSAGE_MIN, reward=None):
    """작업자 프로세스. 맡은 환경들을 굴리고 39채널 값만 돌려준다."""
    slots = [_Slot(beams, lidar_range, s, randomize, rivals, passage, reward)
             for s in seeds]
    pipe.send(np.stack([slot.reset() for slot in slots]))
    try:
        while True:
            try:
                command, payload = pipe.recv()
            except (EOFError, OSError):
                break            # 부모가 먼저 닫았다. 조용히 끝낸다.
            if command == "close":
                break
            features, rewards, dones, finished = [], [], [], []
            for slot, action in zip(slots, payload):
                f, r, d, done_info = slot.step(action)
                features.append(f)
                rewards.append(r)
                dones.append(d)
                if done_info is not None:
                    finished.append(done_info)
            pipe.send((np.stack(features), np.array(rewards, dtype=np.float32),
                       np.array(dones, dtype=np.float32), finished))
    finally:
        pipe.close()


class ParallelRunner:
    """여러 프로세스에 환경을 나눠 담고 한꺼번에 굴린다.

    바깥에서 보기에는 환경 하나짜리 묶음처럼 동작한다.
    """

    def __init__(self, n_envs, beams, lidar_range, workers=None,
                 seed=0, randomize=True, rivals=0.0, passage=PASSAGE_MIN,
                 reward=None):
        workers = workers or min(8, max(1, n_envs // 2))
        groups = np.array_split(np.arange(n_envs), workers)
        groups = [g for g in groups if len(g)]

        context = mp.get_context("spawn")
        self.pipes, self.procs, self.sizes = [], [], []
        for index, group in enumerate(groups):
            parent, child = context.Pipe()
            seeds = [int(seed * 1000 + int(i)) for i in group]
            proc = context.Process(
                target=_worker,
                args=(child, beams, lidar_range, seeds, randomize,
                      rivals, passage, reward),
                daemon=True,
            )
            proc.start()
            child.close()
            self.pipes.append(parent)
            self.procs.append(proc)
            self.sizes.append(len(group))

        self.n_envs = sum(self.sizes)
        self.obs = np.concatenate([pipe.recv() for pipe in self.pipes])
        self.done_stats = []

    def features(self, stats):
        """정규화한 입력. (N, C)

        실제 주행(FlyDriver.act)과 똑같이 +-FEATURE_LIMIT 으로 자른다.
        학습 때와 주행 때 입력 처리가 다르면 실차에서 어긋난다.
        """
        mean, std = stats
        z = (self.obs - mean) / std
        return np.clip(z, -FEATURE_LIMIT, FEATURE_LIMIT).astype(np.float32)

    def step(self, actions):
        """행동을 나눠 보내고 결과를 모은다."""
        start = 0
        for pipe, size in zip(self.pipes, self.sizes):
            pipe.send(("step", actions[start:start + size]))
            start += size

        chunks, rewards, dones = [], [], []
        for pipe in self.pipes:
            features, reward, done, finished = pipe.recv()
            chunks.append(features)
            rewards.append(reward)
            dones.append(done)
            self.done_stats.extend(finished)

        self.obs = np.concatenate(chunks)
        return np.concatenate(rewards), np.concatenate(dones)

    def drain(self):
        stats, self.done_stats = self.done_stats, []
        return stats

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
