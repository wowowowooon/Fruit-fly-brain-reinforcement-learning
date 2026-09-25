"""속도 제어를 고쳐 본다. 지금 이것 때문에 대회 코스를 하나 잃는다.

test_loop 에서 나온 것
    기준          4/4  9.13초  여유 +0.238m
    서보 지연만    4/4  9.12초  여유 +0.235m     <- 서보는 무해하다
    속도 PI 만     3/4  9.83초  여유 -0.001m     <- 여기가 범인
    둘 다          3/4  9.90초  여유 -0.025m

정책은 '목표 속도' 를 내는데 시뮬에서는 그게 바로 반영된다. 실차는
duty -> 모터 -> 속도로 늦게 따라온다. 코너 앞에서 감속을 명령해도 속도가
안 떨어진 채로 들어가 벽에 붙는다.

세 가지를 시험한다.
    빠른PI    이득을 올려 더 빨리 쫓는다
    제동      느려져야 할 때 duty 를 음수로 준다 (역토크)
    앞먹임    감속을 원하면 PI 를 기다리지 않고 바로 duty 를 낮춘다

모터 시정수(tau)는 실측이 없어 가정한 값이다. 어떤 설정이 tau 하나에서만
통하면 그건 푼 게 아니다. 그래서 tau 를 같이 흔들어 본다.

  python deploy/tune_speed.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from deploy.obs_bridge import build_obs
from deploy.test_loop import EspServo, Motor
from flyracer.brain.fly_driver import FlyDriver
from flyracer.sim.f1tenth_sim import (BODY_RADIUS, LIDAR_RANGE, MAX_SPEED,
                                      F1TenthEnv)
from flyracer.sim.map_start import find_start
from flyracer.sim.track_map import MapEnv, MapTrack

MAP_YAML = r"C:\Users\woong\Desktop\ifac_roboracer_maps\ifac_roboracer.yaml"
MODEL = "T_fly871_final.pt"
MOTION, BEAMS = 6, 541
GRIPS = (0.75, 0.88, 1.02, 1.15)


class Speed:
    """목표 속도를 duty 로. 제동과 앞먹임을 켜고 끌 수 있다.

    brake=True  면 목표보다 빠를 때 duty 를 음수로 준다. VESC 는 음수
                duty 를 역방향 토크로 쓰므로 실제로 감속한다. 안 쓰면
                타력으로만 줄어 코너에 속도를 안고 들어간다.
    ff          목표 속도에 비례해 미리 넣는 duty. 클수록 PI 가 할 일이
                줄어 반응이 빨라지지만, 너무 크면 목표를 넘긴다.
    """

    def __init__(self, kp=0.06, ki=0.25, ff=0.075, max_duty=0.60,
                 brake=False, brake_gain=0.0):
        self.kp, self.ki, self.ff = kp, ki, ff
        self.max_duty = max_duty
        self.brake, self.brake_gain = brake, brake_gain
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def step(self, target, actual, dt):
        if target <= 0.02:
            self.integral = 0.0
            return -self.brake_gain if self.brake else 0.0
        error = target - actual
        self.integral += error * dt
        limit = self.max_duty / max(self.ki, 1e-6)
        self.integral = max(-limit, min(limit, self.integral))
        duty = self.ff * target + self.kp * error + self.ki * self.integral
        low = -self.max_duty if self.brake else 0.0
        if self.brake and error < 0:
            # 너무 빠르다. 오차에 비례해 역토크를 준다.
            duty = min(duty, self.brake_gain * error)
        return max(low, min(self.max_duty, duty))


# 앞먹임 ff 는 '목표 속도 / 모터 이득' 이 맞는 값이다. 실측 이득 13.3 이면
# 1/13.3 = 0.075. 이것만 제대로 넣어도 PI 가 거의 할 일이 없다.
FF = 1.0 / 13.3
SETUPS = {
    "지금 것":     dict(kp=0.06, ki=0.25, ff=0.035),
    "앞먹임만":    dict(kp=0.06, ki=0.25, ff=FF),
    "앞먹임+빠른PI": dict(kp=0.25, ki=1.2, ff=FF),
    "앞먹임+제동":  dict(kp=0.06, ki=0.25, ff=FF, brake=True, brake_gain=0.12),
    "전부":        dict(kp=0.25, ki=1.2, ff=FF, brake=True, brake_gain=0.20),
}


def run(driver, env, steps, seed, setup, tau):
    env.reset(seed=seed)
    driver.reset()
    esp, motor = EspServo(), Motor(tau=tau)
    esp.reset()
    motor.reset(env.speed)
    hold = Speed(**setup)
    last_action = [0.0, 0.0]
    gaps = []
    info = {"progress": 0.0, "lap_time": 0.0, "crashed": True}

    for step in range(steps):
        obs = build_obs(env._scan(), env.speed, env.yaw_rate, env.accel,
                        env.steer, last_action, MOTION)
        action = driver.act(obs)
        a0 = float(np.clip(action[0], -1.0, 1.0))
        a1 = float(np.clip(action[1], -1.0, 1.0))
        last_action = [a0, a1]

        steer_cmd = esp.step(a0, env.dt)
        want = (a1 + 1.0) / 2.0 * MAX_SPEED
        got = motor.step(hold.step(want, env.speed, env.dt), env.dt)
        speed_cmd = np.clip(got / MAX_SPEED * 2.0 - 1.0, -1.0, 1.0)

        _, _, done, tr, info = env.step(np.array([steer_cmd, speed_cmd]))
        gaps.append(info["clearance"] - BODY_RADIUS)
        if done or tr:
            break
    return {"done": not info["crashed"] and info["progress"] > 0.97,
            "lap": (step + 1) * env.dt,
            "gap": float(np.min(gaps)) if gaps else 0.0}


def race(driver, track, setup, tau):
    rows = []
    for grip in GRIPS:
        env = MapEnv(track, n_beams=BEAMS, lidar_range=LIDAR_RANGE,
                     motion=MOTION)
        env.start_pose = find_start(track, env)
        env.track_seed = 1234
        env.friction = grip
        rows.append(run(driver, env, 8000, 1234, setup, tau))
    laps = [r["lap"] for r in rows if r["done"]]
    return (sum(r["done"] for r in rows), len(GRIPS),
            float(np.mean(laps)) if laps else 0.0,
            min(r["gap"] for r in rows))


def normal(driver, setup, tau, seeds=range(700, 715)):
    rows = [run(driver, F1TenthEnv(n_beams=BEAMS, lidar_range=LIDAR_RANGE,
                                   track_seed=s, motion=MOTION),
                3000, s, setup, tau) for s in seeds]
    laps = [r["lap"] for r in rows if r["done"]]
    return (sum(r["done"] for r in rows), len(list(seeds)),
            float(np.mean(laps)) if laps else 0.0)


def main():
    driver = FlyDriver(n_beams=BEAMS, lidar_range=LIDAR_RANGE,
                       fly_wiring=True, motion=MOTION)
    driver.load(MODEL)
    track = MapTrack(MAP_YAML)

    # 모터가 빠른 경우(0.15)부터 느린 경우(0.50)까지. 실측이 없어서 흔든다.
    # 실측 시정수는 0.20초(0.11~0.30). 그 범위를 덮게 잡는다.
    for tau in (0.11, 0.20, 0.30):
        print(f"\n{'='*66}")
        print(f"모터 시정수 {tau:.2f}초  (속도가 목표의 63%에 닿는 시간)")
        print(f"{'='*66}")
        print(f"{'설정':<14}{'대회':>8}{'랩':>9}{'여유':>10}"
              f"{'일반':>9}{'랩':>9}")
        print("-" * 66)
        for name, setup in SETUPS.items():
            d, t, lap, gap = race(driver, track, setup, tau)
            nd, nt, nlap = normal(driver, setup, tau)
            flag = "" if (d == t and gap >= 0.20) else "  <- 미달"
            print(f"{name:<14}{d:>5}/{t}{lap:>8.2f}초{gap:>+9.3f}m"
                  f"{nd:>6}/{nt}{nlap:>8.2f}초{flag}", flush=True)

    print("\n합격선: 대회 4/4 + 여유 +0.20m 이상, 일반 15/15")
    print("시뮬 기준값: 대회 4/4  9.13초  +0.238m   일반 15/15  14.07초")


if __name__ == "__main__":
    main()
