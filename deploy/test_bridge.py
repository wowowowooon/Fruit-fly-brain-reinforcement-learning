"""실차 경로로 관측을 만들어도 시뮬과 같은 주행이 나오는지 잰다.

시뮬이 주는 관측을 그대로 넣는 대신, 실차에서 오는 것과 똑같은 모양으로
한 번 풀었다가(=LaserScan 처럼 ranges/angle_min/angle_increment 로 바꿔서)
obs_bridge 로 다시 조립해서 넣는다. 두 주행이 같으면 다리가 멀쩡한 것이다.

세 가지를 같이 본다.
  1. 관측 재조립이 원본과 같은가 (수치)
  2. 그렇게 만든 관측으로 달려도 완주하고 랩이 같은가
  3. 실차 라이다처럼 생긴 스캔(360도, 점 수 다름, inf 섞임)도 되는가

  python deploy/test_bridge.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from deploy.obs_bridge import (action_to_drive, build_obs, model_angles,
                               resample_scan, servo_to_steer)
from flyracer.brain.fly_driver import FlyDriver
from flyracer.sim.f1tenth_sim import (BODY_RADIUS, LIDAR_FOV, LIDAR_RANGE,
                                      MAX_ACCEL, MAX_SPEED, MAX_STEER,
                                      YAW_RATE_MAX, F1TenthEnv)
from flyracer.sim.map_start import find_start
from flyracer.sim.track_map import MapEnv, MapTrack
from flyracer.paths import RACE_MAP

MAP_YAML = str(RACE_MAP)
MODEL = "T_fly871_final.pt"
MOTION = 6
BEAMS = 541


def fake_scan_msg(env, style):
    """시뮬 상태에서 실차 /scan 처럼 생긴 값을 만든다.

    style 로 실차에서 마주칠 모양을 흉내 낸다.
      'same'    학습과 같은 270도 541점
      'coarse'  270도지만 271점 (각분해능 1도)
      'full360' 360도 1440점 — 뒤쪽은 우리 격자 밖이라 버려져야 한다
      'holes'   270도인데 10%가 무반사(inf)
    """
    scan = env._scan()          # 시뮬 라이다 (env.beam_angles 순서)
    angles = env.beam_angles

    if style == "same":
        return scan.copy(), float(angles[0]), float(angles[1] - angles[0]), 0.0

    if style == "coarse":
        want = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, 271)
    elif style == "full360":
        want = np.linspace(-np.pi, np.pi, 1440)
    elif style == "holes":
        want = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, 541)
    else:
        raise ValueError(style)

    out = np.interp(want, angles, scan, left=LIDAR_RANGE, right=LIDAR_RANGE)
    if style == "full360":
        # 시야 밖은 라이다가 못 보는 영역. 실제로는 뒤쪽도 값이 오지만
        # 우리 격자는 ±135도라 버려진다. 버려지는지 보려고 헛값을 넣는다.
        out = np.where(np.abs(want) > LIDAR_FOV / 2, 0.3, out)
    if style == "holes":
        rng = np.random.default_rng(0)
        out = np.where(rng.random(len(out)) < 0.10, np.inf, out)
    inc = float(want[1] - want[0])
    return out, float(want[0]), inc, 0.0


def drive_direct(driver, env, steps, seed):
    """시뮬 관측을 그대로 넣는다 (기준)."""
    obs, _ = env.reset(seed=seed)
    driver.reset()
    gaps, info = [], {"progress": 0.0, "lap_time": 0.0, "crashed": True}
    for step in range(steps):
        obs, _, done, tr, info = env.step(driver.act(obs))
        gaps.append(info["clearance"] - BODY_RADIUS)
        if done or tr:
            break
    return {"done": not info["crashed"] and info["progress"] > 0.97,
            "lap": (step + 1) * env.dt,
            "gap": float(np.min(gaps)) if gaps else 0.0}


def drive_bridged(driver, env, steps, seed, style):
    """실차 경로로 관측을 다시 만들어 넣는다."""
    env.reset(seed=seed)
    driver.reset()
    last_action = [0.0, 0.0]
    gaps, info = [], {"progress": 0.0, "lap_time": 0.0, "crashed": True}
    for step in range(steps):
        ranges, a_min, a_inc, r_min = fake_scan_msg(env, style)
        scan = resample_scan(ranges, a_min, a_inc, BEAMS, r_min)
        # 실차에서 오는 것과 같은 값들
        servo_deg = 90.0 + (env.steer / MAX_STEER) * 50.0
        obs = build_obs(scan, env.speed, env.yaw_rate, env.accel,
                        servo_to_steer(servo_deg), last_action, MOTION)
        action = driver.act(obs)
        steer_rad, speed_mps, last_action = action_to_drive(
            action, steer_scale_rad=0.3735)
        # control_node 가 되돌리는 과정을 그대로 흉내 낸다
        back = np.array([np.clip(steer_rad / 0.3735, -1, 1),
                         speed_mps / MAX_SPEED * 2 - 1])
        _, _, done, tr, info = env.step(back)
        gaps.append(info["clearance"] - BODY_RADIUS)
        if done or tr:
            break
    return {"done": not info["crashed"] and info["progress"] > 0.97,
            "lap": (step + 1) * env.dt,
            "gap": float(np.min(gaps)) if gaps else 0.0}


def main():
    driver = FlyDriver(n_beams=BEAMS, lidar_range=LIDAR_RANGE,
                       fly_wiring=True, motion=MOTION)
    driver.load(MODEL)

    # ---- 1. 관측 재조립이 원본과 같은가 --------------------------------
    print("1) 관측 재조립 정확도")
    env = F1TenthEnv(n_beams=BEAMS, lidar_range=LIDAR_RANGE, track_seed=700,
                     motion=MOTION)
    obs0, _ = env.reset(seed=700)
    for style in ("same", "coarse", "full360", "holes"):
        ranges, a_min, a_inc, r_min = fake_scan_msg(env, style)
        scan = resample_scan(ranges, a_min, a_inc, BEAMS, r_min)
        truth = obs0[:-MOTION] * LIDAR_RANGE
        err = np.abs(scan - truth)
        print(f"   {style:<9} 평균오차 {err.mean():.4f} m   "
              f"최대 {err.max():.3f} m")

    # 조향 환산이 왕복하는가
    for deg in (-40.0, -20.0, 0.0, 15.0, 40.0):
        servo = 90.0 + (deg / 40.0) * 50.0
        back = np.rad2deg(servo_to_steer(servo))
        assert abs(back - deg) < 1e-6, (deg, back)
    print("   조향 환산 왕복  OK")

    # ---- 2. 대회 코스에서 두 경로가 같은가 ------------------------------
    print("\n2) 2026 대회 코스 — 시뮬 경로 vs 실차 경로")
    track = MapTrack(MAP_YAML)
    for style in ("same", "coarse", "full360", "holes"):
        env = MapEnv(track, n_beams=BEAMS, lidar_range=LIDAR_RANGE,
                     motion=MOTION)
        env.start_pose = find_start(track, env)
        env.track_seed = 1234
        env.friction = 1.0
        if style == "same":
            base = drive_direct(driver, env, 8000, 1234)
            print(f"   기준(시뮬 경로)  {'완주' if base['done'] else '실패'}  "
                  f"{base['lap']:.2f}초  여유 {base['gap']:+.3f} m")
        env2 = MapEnv(track, n_beams=BEAMS, lidar_range=LIDAR_RANGE,
                      motion=MOTION)
        env2.start_pose = find_start(track, env2)
        env2.track_seed = 1234
        env2.friction = 1.0
        r = drive_bridged(driver, env2, 8000, 1234, style)
        print(f"   {style:<9}        {'완주' if r['done'] else '실패'}  "
              f"{r['lap']:.2f}초  여유 {r['gap']:+.3f} m")

    # ---- 3. 일반 트랙 여러 개 -------------------------------------------
    print("\n3) 일반 15트랙")
    for style in ("same", "coarse", "holes"):
        done, laps = 0, []
        for seed in range(700, 715):
            env = F1TenthEnv(n_beams=BEAMS, lidar_range=LIDAR_RANGE,
                             track_seed=seed, motion=MOTION)
            r = drive_bridged(driver, env, 3000, seed, style)
            done += r["done"]
            if r["done"]:
                laps.append(r["lap"])
        print(f"   {style:<9} {done}/15 완주  "
              f"{np.mean(laps) if laps else 0:.2f}초")


if __name__ == "__main__":
    main()
