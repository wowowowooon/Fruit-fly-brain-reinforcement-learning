"""실차 명령 경로를 그대로 태워서 대회 코스를 도는지 본다.

지금까지 확인한 것은 조각들이다 — 관측 변환(test_bridge), 시리얼
프로토콜(test_hardware), 추론 속도. 남은 물음은 이것이다.

    정책이 낸 명령이 ESP 서보 지연과 속도 PI 를 거쳐 나왔을 때도
    여전히 대회 코스를 도는가?

시뮬은 서보를 '기울기 제한'(3.2 rad/s)으로 모델링했는데 실차 ESP 는
'지수 지연'(alpha=0.20 @50Hz, 시정수 약 90ms)이다. 모양이 다른 모델이라
과도 구간이 다르다. 속도도 시뮬은 목표를 바로 따라가지만 실차는 duty 를
거쳐 PI 로 맞춘다. 둘 다 정책이 배운 것과 다른 몸이다.

네 경우를 같은 코스에서 비교한다.
    기준        시뮬 그대로
    서보만      ESP 지수 지연을 조향에 얹는다
    속도만      duty + PI 로 속도를 만든다
    둘 다       실차와 같은 경로

  python deploy/test_loop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from deploy.obs_bridge import build_obs
from flyracer.brain.fly_driver import FlyDriver
from flyracer.sim.f1tenth_sim import (BODY_RADIUS, LIDAR_RANGE, MAX_SPEED,
                                      MAX_STEER, F1TenthEnv)
from flyracer.sim.map_start import find_start
from flyracer.sim.track_map import MapEnv, MapTrack

MAP_YAML = r"C:\Users\woong\Desktop\ifac_roboracer_maps\ifac_roboracer.yaml"
MODEL = "T_fly871_final.pt"
MOTION, BEAMS = 6, 541
GRIPS = (0.75, 0.88, 1.02, 1.15)

# ESP32 펌웨어 값 (esp32_steer_rc.ino)
ESP_ALPHA = 0.20
ESP_DT = 0.02


class EspServo:
    """ESP 가 조향 명령을 받아 실제로 서보를 움직이는 모습.

    currentAngle += (target - current) * 0.20   을 20ms 마다.
    한 제어 스텝(dt) 동안 몇 번 도는지 세어 그만큼 반복한다.
    """

    def __init__(self):
        self.now = 0.0

    def reset(self):
        self.now = 0.0

    def step(self, target, dt):
        for _ in range(max(1, int(round(dt / ESP_DT)))):
            self.now += (target - self.now) * ESP_ALPHA
        return self.now


class Motor:
    """duty 를 주면 속도가 따라온다. 실차 속도 계단 시험 14개에서 뽑았다.

        jetson_code/vesc_speed_test_logs/*.csv 를 최소제곱으로 맞춘 값이다.
        ds/dt = (gain*duty - s) / tau
        이득 13.3 m/s per duty (11.7~13.9), 시정수 0.20초 (0.11~0.30)

    여기서 바로 드러나는 것: duty 0.30 이면 정상상태가 4.0 m/s 다.
    T안 최고속 7 m/s 를 내려면 duty 0.53 이 필요하다.
    """

    def __init__(self, gain=13.3, tau=0.20):
        self.gain, self.tau = gain, tau
        self.speed = 0.0

    def reset(self, speed=0.0):
        self.speed = speed

    def step(self, duty, dt):
        self.speed += (duty * self.gain - self.speed) * dt / self.tau
        return max(0.0, self.speed)


def run(driver, env, steps, seed, servo_lag, speed_loop):
    from deploy.speed_ctrl import SpeedControl

    env.reset(seed=seed)
    driver.reset()
    esp, motor, hold = EspServo(), Motor(), SpeedControl(max_duty=0.60)
    esp.reset()
    motor.reset(env.speed)
    last_action = [0.0, 0.0]
    gaps = []
    info = {"progress": 0.0, "lap_time": 0.0, "crashed": True}

    for step in range(steps):
        scan = env._scan()
        obs = build_obs(scan, env.speed, env.yaw_rate, env.accel,
                        env.steer, last_action, MOTION)
        action = driver.act(obs)
        a0 = float(np.clip(action[0], -1.0, 1.0))
        a1 = float(np.clip(action[1], -1.0, 1.0))
        last_action = [a0, a1]

        # 조향: ESP 서보 지연을 거친 값이 실제로 차에 걸린다
        steer_cmd = esp.step(a0, env.dt) if servo_lag else a0

        # 속도: duty -> 모터 -> 속도. 정책이 원한 목표를 PI 가 쫓는다
        if speed_loop:
            want = (a1 + 1.0) / 2.0 * MAX_SPEED
            duty = hold.step(want, env.speed, env.dt)
            got = motor.step(duty, env.dt)
            speed_cmd = np.clip(got / MAX_SPEED * 2.0 - 1.0, -1.0, 1.0)
        else:
            speed_cmd = a1

        _, _, done, tr, info = env.step(np.array([steer_cmd, speed_cmd]))
        gaps.append(info["clearance"] - BODY_RADIUS)
        if done or tr:
            break

    return {"done": not info["crashed"] and info["progress"] > 0.97,
            "lap": (step + 1) * env.dt,
            "gap": float(np.min(gaps)) if gaps else 0.0,
            "progress": float(info["progress"])}


def race(driver, track, servo_lag, speed_loop):
    rows = []
    for grip in GRIPS:
        env = MapEnv(track, n_beams=BEAMS, lidar_range=LIDAR_RANGE,
                     motion=MOTION)
        env.start_pose = find_start(track, env)
        env.track_seed = 1234
        env.friction = grip
        rows.append(run(driver, env, 8000, 1234, servo_lag, speed_loop))
    laps = [r["lap"] for r in rows if r["done"]]
    return {"done": sum(r["done"] for r in rows), "total": len(GRIPS),
            "lap": float(np.mean(laps)) if laps else 0.0,
            "gap": min(r["gap"] for r in rows)}


def normal(driver, servo_lag, speed_loop, seeds=range(700, 715)):
    rows = []
    for s in seeds:
        env = F1TenthEnv(n_beams=BEAMS, lidar_range=LIDAR_RANGE,
                         track_seed=s, motion=MOTION)
        rows.append(run(driver, env, 3000, s, servo_lag, speed_loop))
    laps = [r["lap"] for r in rows if r["done"]]
    return {"done": sum(r["done"] for r in rows), "total": len(list(seeds)),
            "lap": float(np.mean(laps)) if laps else 0.0}


def main():
    driver = FlyDriver(n_beams=BEAMS, lidar_range=LIDAR_RANGE,
                       fly_wiring=True, motion=MOTION)
    driver.load(MODEL)
    track = MapTrack(MAP_YAML)

    cases = (("기준 (시뮬 그대로)", False, False),
             ("서보 지연만", True, False),
             ("속도 PI 만", False, True),
             ("둘 다 (실차 경로)", True, True))

    print("2026 대회 코스 — 노면 4종")
    print(f"{'경우':<22}{'완주':>7}{'랩':>9}{'최소여유':>10}")
    print("-" * 50)
    for name, lag, loop in cases:
        r = race(driver, track, lag, loop)
        print(f"{name:<22}{r['done']:>4}/{r['total']}{r['lap']:>8.2f}초"
              f"{r['gap']:>+9.3f}m", flush=True)

    print("\n일반 15트랙")
    print(f"{'경우':<22}{'완주':>8}{'랩':>9}")
    print("-" * 42)
    for name, lag, loop in cases:
        r = normal(driver, lag, loop)
        print(f"{name:<22}{r['done']:>5}/{r['total']}{r['lap']:>8.2f}초",
              flush=True)


if __name__ == "__main__":
    main()
