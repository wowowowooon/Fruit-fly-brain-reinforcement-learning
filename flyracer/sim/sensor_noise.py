"""라이다 관측을 망가뜨린다.

두 군데에 쓴다.
  1. 강화학습 중 (도메인 랜덤화) — 망가진 상태로 배우면 강건해진다
  2. 비교 실험 — 같은 손상을 여러 컨트롤러에 걸어 누가 버티는지 본다

중요: 환경 안이 아니라 밖에서 건다. 물리와 충돌 판정은 깨끗한 값으로
계산해야 한다. 센서가 망가진 것이지 세상이 망가진 게 아니다.

관측은 F1TenthEnv 규격 그대로 [라이다 0~1 정규화 ..., 속도 0~1] 이다.
"""

import numpy as np

from flyracer.sim.f1tenth_sim import MAX_ACCEL, YAW_RATE_MAX
from flyracer.brain.fly_driver import N_MOTION

# Hokuyo UST-10LX 실측 오차는 약 ±3cm. 그보다 넉넉히 잡아 흔든다.
REAL_NOISE_M = 0.03

# IMU 도 깨끗하지 않다. 특히 1:10 차는 모터와 노면 때문에 진동이 커서
# 가속도계가 많이 튄다. 자이로는 영점이 주행 내내 조금씩 틀어져 있다.
# 완벽한 IMU 만 보고 배우면 실차에서 그 값을 그대로 믿다가 무너진다.
REAL_GYRO_NOISE = 0.02      # [rad/s]
REAL_ACCEL_NOISE = 0.30     # [m/s^2]. 진동이 지배적이다
REAL_GYRO_BIAS = 0.03       # [rad/s]


class SensorFault:
    """라이다 손상 한 가지. 같은 씨앗이면 항상 같은 손상이 재현된다."""

    def __init__(self, name, noise_m=0.0, dropout=0.0, blind=None,
                 max_range=None, stuck=0.0, gyro=0.0, accel=0.0, bias=0.0):
        self.name = name
        self.noise_m = noise_m        # 거리에 섞는 잡음 [m]
        self.dropout = dropout        # 측정 실패 비율 (사거리 끝으로 튄다)
        self.blind = blind            # ('L'|'R'|'front', 각도폭) 완전 차단
        self.max_range = max_range    # 실제보다 짧게 보이는 라이다
        self.stuck = stuck            # 값이 갱신 안 되고 굳는 비율
        self.gyro = gyro              # 각속도 잡음 [rad/s]
        self.accel = accel            # 가속도 잡음 [m/s^2]. 차체 진동이 크다
        self.bias = bias              # 자이로 영점 오차 [rad/s]. 주행 내내 일정

    def __repr__(self):
        return f"SensorFault({self.name})"


# 실험에 쓸 손상 목록. 왼쪽이 약하고 오른쪽으로 갈수록 심하다.
def fault_sweep(lidar_range=10.0):
    return {
        "없음": [SensorFault("깨끗")],
        "거리 잡음": [
            SensorFault(f"±{m * 100:.0f}cm", noise_m=m)
            for m in (0.03, 0.10, 0.25, 0.50, 1.00)
        ],
        "빔 손실": [
            SensorFault(f"{int(p * 100)}%", dropout=p)
            for p in (0.1, 0.3, 0.5, 0.7, 0.9)
        ],
        "한쪽 시야 차단": [
            SensorFault(f"{side} {deg}도", blind=(side, deg))
            for side in ("L", "R") for deg in (45, 90, 135)
        ],
        "사거리 축소": [
            SensorFault(f"{r:.0f}m", max_range=r)
            for r in (7.0, 5.0, 3.0, 2.0, 1.0)
        ],
        "값 굳음": [
            SensorFault(f"{int(p * 100)}%", stuck=p)
            for p in (0.1, 0.3, 0.5, 0.7)
        ],
    }


class NoisyLidar:
    """관측을 받아 망가뜨려 돌려준다. 원본은 건드리지 않는다."""

    def __init__(self, beam_angles, lidar_range, fault=None, rng=None,
                 motion=N_MOTION):
        # 관측 뒤에 붙은 자기 움직임 채널 수. 모델마다 다르다
        # (R안 3, T안 6). 모듈 상수로 박아 두면 T안에서 라이다를
        # 세 칸 더 잘라 먹어 그대로 터진다.
        self.motion = int(motion)
        self.beam_angles = beam_angles
        self.lidar_range = lidar_range
        self.fault = fault or SensorFault("깨끗")
        self.rng = rng or np.random.default_rng(0)
        self.held = None
        self.gyro_bias = 0.0
        self._blind_mask = self._make_blind_mask()

    def _make_blind_mask(self):
        if self.fault.blind is None:
            return None
        side, degrees = self.fault.blind
        span = np.deg2rad(degrees)
        if side == "L":
            return self.beam_angles > 0
        if side == "R":
            return self.beam_angles < 0
        return np.abs(self.beam_angles) <= span / 2      # front

    def reset(self):
        self.held = None
        # 영점 오차는 매 스텝 흔들리는 게 아니라 주행 내내 일정하다
        self.gyro_bias = float(self.rng.normal(0.0, self.fault.bias))

    def __call__(self, obs):
        fault = self.fault
        scan = obs[:-self.motion] * self.lidar_range

        if fault.max_range is not None:
            scan = np.minimum(scan, fault.max_range)

        if fault.noise_m > 0:
            scan = scan + self.rng.normal(0.0, fault.noise_m, scan.shape)

        if fault.dropout > 0:
            # 측정 실패는 0이 아니라 '끝까지 안 맞음'으로 나타난다
            lost = self.rng.random(scan.shape) < fault.dropout
            scan = np.where(lost, self.lidar_range, scan)

        if self._blind_mask is not None:
            scan = np.where(self._blind_mask, self.lidar_range, scan)

        if fault.stuck > 0:
            if self.held is None:
                self.held = scan.copy()
            frozen = self.rng.random(scan.shape) < fault.stuck
            scan = np.where(frozen, self.held, scan)
            self.held = scan.copy()

        scan = np.clip(scan, 0.0, self.lidar_range)

        # 관측 뒤쪽은 [속도, 각속도, 가속도] 다. obs[-1:] 로 자르면
        # IMU 두 개가 통째로 사라진다.
        motion = obs[-self.motion:].astype(np.float32).copy()
        if self.motion >= 3:
            if fault.gyro or fault.bias:
                motion[1] += float(
                    self.rng.normal(0.0, fault.gyro) + self.gyro_bias
                ) / YAW_RATE_MAX
            if fault.accel:
                motion[2] += float(
                    self.rng.normal(0.0, fault.accel)) / MAX_ACCEL
            motion[1:] = np.clip(motion[1:], -1.5, 1.5)
        return np.concatenate([scan / self.lidar_range, motion]).astype(np.float32)


class DomainRandomizer:
    """학습 중 매 에피소드마다 손상 정도를 새로 뽑는다.

    한 가지 손상에만 맞추면 그것만 잘하게 된다. 매번 다르게 걸어야
    '센서를 덜 믿는' 쪽으로 배운다.
    """

    def __init__(self, beam_angles, lidar_range, rng=None, strength=1.0):
        self.beam_angles = beam_angles
        self.lidar_range = lidar_range
        self.rng = rng or np.random.default_rng(0)
        self.strength = strength
        self.lidar = NoisyLidar(beam_angles, lidar_range, rng=self.rng)

    def reset(self):
        r = self.rng
        s = self.strength
        # 절반은 거의 깨끗하게 둔다. 깨끗할 때도 잘 달려야 한다.
        if r.random() < 0.5:
            fault = SensorFault(
                "약함", noise_m=REAL_NOISE_M * r.uniform(0.5, 2.0),
                gyro=REAL_GYRO_NOISE, accel=REAL_ACCEL_NOISE,
                bias=REAL_GYRO_BIAS)
        else:
            fault = SensorFault(
                "랜덤",
                noise_m=REAL_NOISE_M + r.exponential(0.08) * s,
                dropout=r.random() * 0.35 * s,
                max_range=self.lidar_range * r.uniform(0.6, 1.0),
                stuck=r.random() * 0.15 * s,
                gyro=REAL_GYRO_NOISE * r.uniform(1.0, 4.0) * s,
                accel=REAL_ACCEL_NOISE * r.uniform(1.0, 3.0) * s,
                bias=REAL_GYRO_BIAS * r.uniform(1.0, 3.0) * s,
            )
        self.lidar.fault = fault
        self.lidar._blind_mask = self.lidar._make_blind_mask()
        self.lidar.reset()
        return fault

    def __call__(self, obs):
        return self.lidar(obs)
