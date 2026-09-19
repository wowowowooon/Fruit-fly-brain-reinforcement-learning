"""라이다와 초파리 회로를 잇는 통역기.

라이다 빔 108개를 그대로 넣지 않고, 초파리 시각 뉴런이 원래 읽는 두 가지로 바꾼다.
  근접도  벽이 얼마나 가까운가          HS 계열이 보는 시야 흐름에 해당
  접근율  그 벽이 얼마나 빨리 다가오나   LC4, LPLC2가 보는 looming에 해당
왼쪽 시야는 좌반구 뉴런에, 오른쪽은 우반구 뉴런에 꽂는다.
"""

import numpy as np
import pandas as pd
import torch

from build_wiring import FlyController, build_wiring
from f1tenth_sim import CONTROL_DT, LIDAR_BEAMS, LIDAR_RANGE

N_SECTORS = 9          # 한쪽 시야를 몇 구역으로 나눌지
PROX_QUANTILE = 0.1    # 구역 대표 거리. 최솟값 대신 하위 10%를 쓴다
LOOM_SMOOTH = 0.6      # 접근율 저역통과 계수. 0이면 생값, 1에 가까울수록 둔해짐


def sensor_channels(n_sectors=N_SECTORS):
    """컨트롤러에 들어갈 입력 채널 목록. (이름, 종류, 반구, 구역)"""
    channels = []
    for side in ("L", "R"):
        for sector in range(n_sectors):
            channels.append((f"prox_{side}{sector}", "prox", side, sector))
            channels.append((f"loom_{side}{sector}", "loom", side, sector))
    channels.append(("speed", "speed", "both", None))
    return channels


class LidarEncoder:
    """라이다 스캔을 근접도와 접근율로 바꾼다.

    거리를 그대로 쓰면 값이 좁은 구간에 뭉쳐서 신호가 죽는다.
    수집한 데이터의 평균과 폭으로 펴준다.

    빔 개수와는 무관하게 동작한다. 108개든 실차의 1081개든 같은 37채널이 나온다.
    """

    def __init__(self, n_beams=LIDAR_BEAMS, n_sectors=N_SECTORS,
                 lidar_range=LIDAR_RANGE, quantile=PROX_QUANTILE,
                 loom_smooth=LOOM_SMOOTH):
        self.n_sectors = n_sectors
        self.lidar_range = lidar_range
        self.loom_smooth = loom_smooth

        # 빔을 좌우로 가르고, 각 쪽을 시야 구역으로 쪼갠다.
        # 양쪽 다 구역 0이 정면, 번호가 커질수록 뒤쪽이 되게 맞춘다.
        # (R을 뒤집지 않으면 같은 번호가 좌우에서 반대 방향을 가리킨다)
        half = n_beams // 2
        sides = (("L", np.arange(half, n_beams)), ("R", np.arange(half)[::-1]))
        self.sector_beams = []
        for _, beams in sides:
            self.sector_beams.extend(np.array_split(beams, n_sectors))
        # 구역 대표 거리로 쓸 순위. 최솟값은 빔 하나만 튀어도 같이 튄다.
        self.ranks = [
            min(int(quantile * len(beams)), len(beams) - 1)
            for beams in self.sector_beams
        ]

        self.prev_prox = None
        self.loom = np.zeros(len(self.sector_beams), dtype=np.float64)
        self.mean = None
        self.std = None

    def set_stats(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), 1e-3)

    def normalize(self, values):
        if self.mean is None:
            return values
        return ((values - self.mean) / self.std).astype(np.float32)

    def reset(self):
        self.prev_prox = None
        self.loom[:] = 0.0

    def raw(self, scan, speed_norm, dt=CONTROL_DT):
        """정규화 전 특징. 구역별 근접도와 그 변화율."""
        scan = np.minimum(scan, self.lidar_range)
        near = np.array([
            np.partition(scan[beams], rank)[rank]
            for beams, rank in zip(self.sector_beams, self.ranks)
        ])
        prox = 1.0 / (1.0 + near)    # 가까울수록 크게, 멀면 0에 수렴

        if self.prev_prox is None:
            self.prev_prox = prox

        # 차분은 노이즈를 1/dt배로 키운다. 그대로 쓰면 접근율이 잡음투성이가 된다.
        raw_loom = (prox - self.prev_prox) / dt
        self.loom = self.loom_smooth * self.loom + (1 - self.loom_smooth) * raw_loom
        self.prev_prox = prox

        # 채널 순서: 구역마다 [근접도, 접근율], 맨 끝에 속도
        values = np.empty(2 * len(prox) + 1, dtype=np.float32)
        values[0:-1:2] = prox
        values[1:-1:2] = self.loom
        values[-1] = speed_norm
        return values

    def encode(self, scan, speed_norm, dt=CONTROL_DT):
        return self.normalize(self.raw(scan, speed_norm, dt))


class FlyDriver:
    """관측을 받아 조향과 속도를 내놓는다."""

    name = "초파리 배선 (CfC)"

    def __init__(self, n_beams=LIDAR_BEAMS, device="cpu",
                 lidar_range=LIDAR_RANGE, dt=CONTROL_DT):
        neurons = pd.read_csv("subcircuit_neurons.csv")
        connections = pd.read_csv("subcircuit_connections.csv")

        self.channels = sensor_channels()
        self.wiring, self.ordered, self.info = build_wiring(
            neurons, connections, self.channels, N_SECTORS
        )
        self.model = FlyController(
            self.wiring, len(self.channels),
            self.info["n_drive"], self.info["n_steering"],
        ).to(device)

        self.encoder = LidarEncoder(n_beams, lidar_range=lidar_range)
        self.lidar_range = lidar_range
        self.dt = dt
        self.device = device
        self.hidden = None
        # 뷰어에서 들여다볼 수 있게 마지막 입력을 남겨둔다.
        self.last_features = np.zeros(len(self.channels), dtype=np.float32)

    def reset(self):
        self.encoder.reset()
        self.hidden = None
        self.last_features[:] = 0.0

    def save(self, path="fly_model.pt"):
        # 하드웨어 사양을 같이 넣어 둔다. 사거리가 다르면 입력 분포가 달라서
        # 같은 가중치를 써도 엉뚱하게 달린다. 실차로 옮길 때 확인용.
        torch.save({
            "model": self.model.state_dict(),
            "mean": self.encoder.mean,
            "std": self.encoder.std,
            "lidar_range": self.lidar_range,
            "n_sectors": N_SECTORS,
            "dt": self.dt,
        }, path)

    def load(self, path="fly_model.pt"):
        bundle = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(bundle["model"])
        if bundle["mean"] is not None:
            self.encoder.set_stats(bundle["mean"], bundle["std"])

        trained_range = bundle.get("lidar_range")
        if trained_range is not None and abs(trained_range - self.lidar_range) > 1e-6:
            print(f"경고: {trained_range:.1f}m 라이다로 학습된 모델을 "
                  f"{self.lidar_range:.1f}m 로 쓰고 있습니다. 다시 학습하세요.")
        self.model.eval()
        return self

    @torch.no_grad()
    def act(self, obs):
        scan = obs[:-1] * self.lidar_range
        features = self.encoder.encode(scan, obs[-1], self.dt)
        self.last_features = features
        x = torch.from_numpy(features).view(1, 1, -1).to(self.device)
        command, self.hidden = self.model(x, self.hidden)
        return command[0, 0].cpu().numpy()

    def summary(self):
        synapses = int(np.abs(self.wiring.adjacency_matrix).sum())
        inhib = int((self.wiring.adjacency_matrix < 0).sum())
        params = sum(p.numel() for p in self.model.parameters())
        kinds = self.info["kinds"]
        return (
            f"뉴런 {self.wiring.units}개, 배선 {synapses}개 "
            f"(억제 {inhib / synapses:.0%})\n"
            f"시각 뉴런  좁은시야 물체 {kinds.get('prox', 0)}개 / "
            f"좁은시야 접근 {kinds.get('loom', 0)}개 / "
            f"광역 {kinds.get('wide', 0)}개\n"
            f"입력 채널 {len(self.channels)}개 -> 센서 배선 {self.info['links']}개\n"
            f"운동 뉴런  조향 {self.info['n_steering']}개 -> 조향축, "
            f"구동 {self.info['n_drive']}개 -> 속도축\n"
            f"학습 파라미터 {params:,}개"
        )


if __name__ == "__main__":
    from f1tenth_sim import F1TenthEnv

    env = F1TenthEnv()
    driver = FlyDriver(n_beams=env.n_beams, lidar_range=env.lidar_range, dt=env.dt)
    print(driver.summary(), "\n")

    obs, _ = env.reset(seed=0)
    driver.reset()

    total = 0.0
    for _ in range(400):
        action = driver.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        total += reward
        if done or truncated:
            break

    print(f"{env.steps}스텝   진행도 {info['progress']:.1%}")
    print(f"충돌 {info['crashed']}   보상 {total:.0f}")
    print("\n(학습 전이라 제대로 못 달리는 게 정상)")
