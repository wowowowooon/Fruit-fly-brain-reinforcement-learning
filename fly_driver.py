"""라이다와 초파리 회로를 잇는 통역기.

라이다 빔 108개를 그대로 넣지 않고, 초파리 시각 뉴런이 원래 읽는 두 가지로 바꾼다.
  근접도  벽이 얼마나 가까운가          HS 계열이 보는 시야 흐름에 해당
  접근율  그 벽이 얼마나 빨리 다가오나   LC4, LPLC2가 보는 looming에 해당
왼쪽 시야는 좌반구 뉴런에, 오른쪽은 우반구 뉴런에 꽂는다.
"""

import numpy as np
import pandas as pd
import torch

from build_wiring import (FlyController, apply_fly_wiring, build_wiring,
                          enforce_polarity, polarity_kept, wiring_report)
from paths import DATA, model as model_path
from f1tenth_sim import CONTROL_DT, LIDAR_BEAMS, LIDAR_RANGE

N_SECTORS = 9          # 한쪽 시야를 몇 구역으로 나눌지
PROX_QUANTILE = 0.1    # 구역 대표 거리. 최솟값 대신 하위 10%를 쓴다
LOOM_SMOOTH = 0.6      # 접근율 저역통과 계수. 0이면 생값, 1에 가까울수록 둔해짐

# 접근율의 물리적 한계.
#   prox = 1/(1+d) 이므로 d(prox)/dt = -(접근속도)/(1+d)^2
#   가장 심한 경우(d=0.3m, 7m/s 로 정면 접근)라도 4 를 넘지 않는다.
# 이보다 큰 값은 벽이 실제로 다가온 게 아니라, 코너를 돌면서 구역의 시야가
# 뚝 끊겨 거리가 건너뛴 것이다. 실제 스캔 맵에서 40배까지 튀는 것을 확인했다.
LOOM_LIMIT = 4.0

# 정규화 후 허용 범위. 학습 때 본 적 없는 크기의 입력은 잘라서 넣는다.
# 처음 보는 코스에서 모델이 엉뚱한 값을 받고 무너지는 것을 막는다.
FEATURE_LIMIT = 5.0


def sensor_channels(n_sectors=N_SECTORS):
    """컨트롤러에 들어갈 입력 채널 목록. (이름, 종류, 반구, 구역)"""
    channels = []
    for side in ("L", "R"):
        for sector in range(n_sectors):
            channels.append((f"prox_{side}{sector}", "prox", side, sector))
            channels.append((f"loom_{side}{sector}", "loom", side, sector))
    # 자기 움직임(IMU). 초파리에서 HS/LPT 광역 세포가 하는 일이 바로
    # "내가 지금 얼마나 돌고 있나"를 읽는 것이다. 자이로가 재는 값과 같다.
    channels.append(("speed", "speed", "both", None))
    channels.append(("yaw_rate", "speed", "both", None))
    channels.append(("accel", "speed", "both", None))
    return channels


# 라이다 뒤에 붙는 자기 움직임 값의 개수 (속도, 각속도, 가속도)
N_MOTION = 3


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
        z = (values - self.mean) / self.std
        return np.clip(z, -FEATURE_LIMIT, FEATURE_LIMIT).astype(np.float32)

    def reset(self):
        self.prev_prox = None
        self.loom[:] = 0.0

    def raw(self, scan, motion, dt=CONTROL_DT):
        """정규화 전 특징. 구역별 근접도, 그 변화율, 그리고 자기 움직임.

        motion: [속도, 각속도, 가속도] 를 정규화한 값 (IMU)
        """
        scan = np.minimum(scan, self.lidar_range)
        near = np.array([
            np.partition(scan[beams], rank)[rank]
            for beams, rank in zip(self.sector_beams, self.ranks)
        ])
        prox = 1.0 / (1.0 + near)    # 가까울수록 크게, 멀면 0에 수렴

        if self.prev_prox is None:
            self.prev_prox = prox

        # 차분은 노이즈를 1/dt배로 키운다. 그대로 쓰면 접근율이 잡음투성이가 된다.
        # 물리적으로 불가능한 값은 시야가 끊긴 것이므로 먼저 잘라낸다.
        raw_loom = np.clip((prox - self.prev_prox) / dt, -LOOM_LIMIT, LOOM_LIMIT)
        self.loom = self.loom_smooth * self.loom + (1 - self.loom_smooth) * raw_loom
        self.prev_prox = prox

        # 채널 순서: 구역마다 [근접도, 접근율], 맨 끝에 자기 움직임 3개
        motion = np.atleast_1d(np.asarray(motion, dtype=np.float32))
        values = np.empty(2 * len(prox) + len(motion), dtype=np.float32)
        body = values[:-len(motion)]
        body[0::2] = prox
        body[1::2] = self.loom
        values[-len(motion):] = motion
        return values

    def encode(self, scan, motion, dt=CONTROL_DT):
        return self.normalize(self.raw(scan, motion, dt))


def load_eye_position():
    """시각 뉴런이 시야의 어느 방향을 보는지. 없으면 None."""
    path = DATA / "sensory_position.csv"
    if not path.exists():
        print(f"경고: {path.name} 이 없습니다. 시각 뉴런을 시야 구역에 "
              f"임의로 나눠 넣습니다. tools/fetch_eye_position.py 를 "
              f"먼저 돌리세요.")
        return None
    frame = pd.read_csv(path)
    return dict(zip(frame.bodyId, frame.front_to_side))


def check_wiring(model, wiring, fly_wiring):
    """배선이 의도대로 들어갔는지 확인하고, 아니면 멈춘다.

    ncps 가 초파리 배선을 조용히 무시하고 밀집 신경망을 만들어 버린 적이
    있다. 7시간짜리 학습을 돌리고 나서야 알았다. 그 뒤로는 학습을 시작하기
    전에 반드시 센다.
    """
    counts = wiring_report(model, wiring)
    if fly_wiring:
        if counts["links"] != counts["expected"]:
            raise RuntimeError(
                f"초파리 배선이 안 들어갔습니다. "
                f"연결 {counts['links']:,}개 (있어야 할 값 "
                f"{counts['expected']:,}개)"
            )
        if counts["inhibitory"] == 0:
            raise RuntimeError("억제성 연결이 하나도 없습니다. 극성 누락")
        kept = polarity_kept(model, wiring)
        if kept < 0.999:
            raise RuntimeError(
                f"초파리 극성이 지켜지지 않습니다. 유지율 {kept:.1%} "
                f"(가중치가 음수로 가면 억제성이 흥분성으로 뒤집힌다)")
        counts["polarity"] = kept
    elif counts["links"] != counts["dense_units"]:
        raise RuntimeError(
            f"밀집이어야 하는데 연결이 {counts['links']:,}개뿐입니다 "
            f"(있어야 할 값 {counts['dense_units']:,}개)"
        )
    return counts


class FlyDriver:
    """관측을 받아 조향과 속도를 내놓는다.

    fly_wiring=True  초파리 배선 (C안). 연결 9,992개
    fly_wiring=False 전부 연결 (B안, 비교용). 연결 758,641개
    """

    name = "초파리 배선 (CfC)"

    def __init__(self, n_beams=LIDAR_BEAMS, device="cpu",
                 lidar_range=LIDAR_RANGE, dt=CONTROL_DT, fly_wiring=True):
        neurons = pd.read_csv(DATA / "subcircuit_neurons.csv")
        connections = pd.read_csv(DATA / "subcircuit_connections.csv")

        self.channels = sensor_channels()
        self.wiring, self.ordered, self.info = build_wiring(
            neurons, connections, self.channels, N_SECTORS,
            position=load_eye_position(),
        )
        self.model = FlyController(
            self.wiring, len(self.channels),
            self.info["n_drive"], self.info["n_steering"],
            steer_sign=self.info["steer_sign"],
        ).to(device)
        if fly_wiring:
            apply_fly_wiring(self.model, self.wiring, self.info["strength"])
        self.fly_wiring = fly_wiring
        check_wiring(self.model, self.wiring, fly_wiring)

        self.encoder = LidarEncoder(n_beams, lidar_range=lidar_range)
        self.lidar_range = lidar_range
        self.dt = dt
        self.device = device
        self.hidden = None
        # 뷰어에서 들여다볼 수 있게 마지막 입력을 남겨둔다.
        self.last_features = np.zeros(len(self.channels), dtype=np.float32)

    def hold_polarity(self):
        """학습 스텝마다 불러서 초파리 극성을 유지한다."""
        if self.fly_wiring:
            enforce_polarity(self.model, self.wiring)

    def reset(self):
        self.encoder.reset()
        self.hidden = None
        self.last_features[:] = 0.0

    def save(self, path="fly_model.pt"):
        path = model_path(path)
        # 하드웨어 사양을 같이 넣어 둔다. 사거리가 다르면 입력 분포가 달라서
        # 같은 가중치를 써도 엉뚱하게 달린다. 실차로 옮길 때 확인용.
        torch.save({
            "model": self.model.state_dict(),
            "mean": self.encoder.mean,
            "std": self.encoder.std,
            "lidar_range": self.lidar_range,
            "n_sectors": N_SECTORS,
            "dt": self.dt,
            # 어떤 배선으로 학습된 모델인지. 섞이면 비교가 무의미해진다.
            "fly_wiring": self.fly_wiring,
            "n_motion": N_MOTION,
            "n_channel": len(self.channels),
        }, path)

    def load(self, path="fly_model.pt"):
        path = model_path(path)
        bundle = torch.load(path, map_location=self.device, weights_only=False)

        trained_wiring = bundle.get("fly_wiring")
        if trained_wiring is not None and trained_wiring != self.fly_wiring:
            raise RuntimeError(
                f"배선이 다른 모델입니다. 파일은 "
                f"{'초파리' if trained_wiring else '밀집'}로 학습됐는데 "
                f"{'초파리' if self.fly_wiring else '밀집'}로 불러오려 합니다."
            )
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
        scan = obs[:-N_MOTION] * self.lidar_range
        features = self.encoder.encode(scan, obs[-N_MOTION:], self.dt)
        self.last_features = features
        x = torch.from_numpy(features).view(1, 1, -1).to(self.device)
        command, self.hidden = self.model(x, self.hidden)
        return command[0, 0].cpu().numpy()

    def effective_params(self):
        """실제로 살아 있는 연결만 센다.

        마스크로 꺼 놓은 자리에도 파라미터는 남아 있다. 저장 용량은 같지만
        계산에 쓰이는 것은 켜진 자리뿐이라, 비교할 때는 이쪽을 봐야 한다.
        """
        mask = self.model.rnn.rnn_cell.layer_0.sparsity_mask.data
        live = int((mask != 0).sum())
        # 연결 하나마다 ff1/ff2/time_a/time_b 네 벌
        return live * 4 + self.wiring.units * 4 + 14

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
            f"배선 {'초파리' if self.fly_wiring else '전부 연결 (비교군)'}\n"
            f"파라미터 저장 {params:,}개, 실제 사용 "
            f"{self.effective_params():,}개"
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
