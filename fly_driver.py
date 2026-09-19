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

N_SECTORS = 9          # 한쪽 시야를 몇 구역으로 나눌지
LIDAR_RANGE = 30.0


def sensor_channels(n_sectors=N_SECTORS):
    """컨트롤러에 들어갈 입력 채널 목록."""
    channels = []
    for side in ("L", "R"):
        for sector in range(n_sectors):
            channels.append((f"prox_{side}{sector}", side, sector))
            channels.append((f"loom_{side}{sector}", side, sector))
    channels.append(("speed", "both", None))
    return channels


class LidarEncoder:
    """라이다 스캔을 근접도와 접근율로 바꾼다.

    거리를 그대로 쓰면 값이 좁은 구간에 뭉쳐서 신호가 죽는다.
    수집한 데이터의 평균과 폭으로 펴준다.
    """

    def __init__(self, n_beams, n_sectors=N_SECTORS):
        self.n_sectors = n_sectors
        # 빔을 좌우로 가르고, 각 쪽을 시야 구역으로 쪼갠다.
        half = n_beams // 2
        self.slices = {}
        for side, beams in (("R", np.arange(half)), ("L", np.arange(half, n_beams))):
            for sector, chunk in enumerate(np.array_split(beams, n_sectors)):
                self.slices[(side, sector)] = chunk
        self.prev_prox = None
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

    def raw(self, scan, speed_norm, dt=0.05):
        """정규화 전 특징. 구역별 근접도와 그 변화율."""
        prox = {}
        for key, beams in self.slices.items():
            nearest = min(scan[beams].min(), LIDAR_RANGE)
            prox[key] = 1.0 / (1.0 + nearest)   # 가까울수록 크게, 멀면 0에 수렴

        if self.prev_prox is None:
            self.prev_prox = prox

        values = []
        for side in ("L", "R"):
            for sector in range(self.n_sectors):
                key = (side, sector)
                values.append(prox[key])
                values.append((prox[key] - self.prev_prox[key]) / dt)
        values.append(speed_norm)

        self.prev_prox = prox
        return np.array(values, dtype=np.float32)

    def encode(self, scan, speed_norm, dt=0.05):
        return self.normalize(self.raw(scan, speed_norm, dt))


class FlyDriver:
    """관측을 받아 조향과 속도를 내놓는다."""

    def __init__(self, n_beams=108, device="cpu"):
        neurons = pd.read_csv("subcircuit_neurons.csv")
        connections = pd.read_csv("subcircuit_connections.csv")

        self.channels = sensor_channels()
        self.wiring, self.ordered, self.links = build_wiring(
            neurons, connections, self.channels, N_SECTORS
        )
        self.model = FlyController(
            self.wiring, len(self.channels), self.wiring.output_dim
        ).to(device)

        self.encoder = LidarEncoder(n_beams)
        self.device = device
        self.hidden = None
        # 뷰어에서 들여다볼 수 있게 마지막 입력을 남겨둔다.
        self.last_features = np.zeros(len(self.channels), dtype=np.float32)

    def reset(self):
        self.encoder.reset()
        self.hidden = None
        self.last_features[:] = 0.0

    def save(self, path="fly_model.pt"):
        torch.save({
            "model": self.model.state_dict(),
            "mean": self.encoder.mean,
            "std": self.encoder.std,
        }, path)

    def load(self, path="fly_model.pt"):
        bundle = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(bundle["model"])
        if bundle["mean"] is not None:
            self.encoder.set_stats(bundle["mean"], bundle["std"])
        self.model.eval()
        return self

    @torch.no_grad()
    def act(self, obs):
        scan = obs[:-1] * LIDAR_RANGE
        features = self.encoder.encode(scan, obs[-1])
        self.last_features = features
        x = torch.from_numpy(features).view(1, 1, -1).to(self.device)
        command, self.hidden = self.model(x, self.hidden)
        return command[0, 0].cpu().numpy()

    def summary(self):
        synapses = int(np.abs(self.wiring.adjacency_matrix).sum())
        inhib = int((self.wiring.adjacency_matrix < 0).sum())
        params = sum(p.numel() for p in self.model.parameters())
        return (
            f"뉴런 {self.wiring.units}개, 배선 {synapses}개 "
            f"(억제 {inhib / synapses:.0%})\n"
            f"입력 채널 {len(self.channels)}개 -> 센서 배선 {self.links}개\n"
            f"출력 뉴런 {self.wiring.output_dim}개 -> [조향, 속도]\n"
            f"학습 파라미터 {params:,}개"
        )


if __name__ == "__main__":
    from f1tenth_sim import F1TenthEnv

    env = F1TenthEnv(n_beams=108)
    driver = FlyDriver(n_beams=108)
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
