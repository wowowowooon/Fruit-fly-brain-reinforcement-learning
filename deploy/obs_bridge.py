"""실차 센서 값을 모델이 기대하는 관측으로 바꾼다.

ROS 를 안 쓴다. 노드는 메시지에서 숫자만 꺼내 여기로 넘기고, 여기서
모델이 먹을 형태를 만든다. 이렇게 갈라 놓으면 PC 에서 시뮬 스캔을 넣어
"실차 경로로 넣어도 같은 관측이 나오는지" 를 잴 수 있다.

관측이 한 칸이라도 어긋나면 조용히 틀린다. 실제로 sensor_noise 가
자기 움직임 채널을 3으로 박아 둬서 T안(6채널) 라이다를 세 칸 잘라
먹은 적이 있다. 그때는 터져서 알았지만, 안 터지는 어긋남이 더 무섭다.
"""

import math

import numpy as np

from flyracer.sim.f1tenth_sim import (LIDAR_FOV, LIDAR_RANGE, MAX_ACCEL,
                                      MAX_SPEED, MAX_STEER, YAW_RATE_MAX)

# ESP32 서보 규격 (esp32_steer_rc.ino 와 같아야 한다)
SERVO_CENTER_DEG = 90.0
SERVO_HALF_DEG = 50.0          # 40~140 -> 중립에서 ±50도


def model_angles(beams):
    """모델이 기대하는 빔 각도. 정면 0, 왼쪽이 양수."""
    return np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, int(beams))


def resample_scan(ranges, angle_min, angle_increment, beams,
                  range_min=0.0, scan_sign=1.0, lidar_range=LIDAR_RANGE):
    """실제 라이다 스캔을 모델 격자로 옮긴다.

    점 개수도, 시야도, 시작 각도도 장비마다 다르다. 메시지가 알려 주는
    angle_min 과 angle_increment 를 믿고 각도로 맞춘다. 그러면 270도든
    360도든, 몇 점이든 맞는다.

    안 맞은 빔(inf, nan, 0, 최소거리 미만)은 '끝까지 안 맞음' 으로 바꾼다.
    0 으로 두면 벽이 코앞에 있는 것처럼 보여 차가 급정거한다.
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    n = len(ranges)
    if n < 8:
        return None
    inc = float(angle_increment)
    if not math.isfinite(inc) or abs(inc) < 1e-9:
        return None

    have = float(angle_min) + np.arange(n) * inc
    bad = ~np.isfinite(ranges)
    bad |= ranges <= max(float(range_min), 1e-3)
    ranges = np.where(bad, lidar_range, ranges)
    ranges = np.clip(ranges, 0.0, lidar_range)

    if scan_sign < 0:
        have = -have
    order = np.argsort(have)
    have, ranges = have[order], ranges[order]

    # 우리 격자 밖(뒤쪽)은 '안 맞음' 으로 채운다
    out = np.interp(model_angles(beams), have, ranges,
                    left=lidar_range, right=lidar_range)
    return out.astype(np.float32)


def servo_to_steer(servo_command_deg, steer_sign=1.0):
    """서보 혼 각도(40~140) -> 앞바퀴 각도 [rad].

    조향 센서가 따로 없다. ESP 가 실제로 서보에 쓴 각도가 가장 가까운
    값이다. 스무딩(alpha=0.2)까지 거친 값이라 명령보다 실제에 가깝다.

    혼은 중립 90 에서 ±50도 움직이고, 링크비 때문에 앞바퀴는 ±40도까지
    간다(각도기 실측). 그래서 비율로 환산한 뒤 MAX_STEER 를 곱한다.
    """
    horn = float(servo_command_deg) - SERVO_CENTER_DEG
    frac = float(np.clip(horn / SERVO_HALF_DEG, -1.0, 1.0))
    return steer_sign * frac * MAX_STEER


def build_obs(scan, speed, yaw_rate, accel, steer, last_action, motion,
              lidar_range=LIDAR_RANGE):
    """모델이 먹는 관측을 만든다.

    순서는 f1tenth_sim._observe 와 반드시 같아야 한다.
      [라이다/사거리] + [속도, 각속도, 가속도, 조향각, 직전조향, 직전속도]
    """
    row = [speed / MAX_SPEED,
           yaw_rate / YAW_RATE_MAX,
           accel / MAX_ACCEL]
    if motion >= 4:
        row.append(steer / MAX_STEER)
    if motion >= 6:
        row.extend(last_action[:2])
    return np.concatenate([np.asarray(scan, dtype=np.float64) / lidar_range,
                           np.asarray(row[:motion])]).astype(np.float32)


def action_to_drive(action, steer_scale_rad, steer_sign=1.0,
                    speed_scale=1.0, speed_cap=MAX_SPEED):
    """정책 출력 -> /drive 에 실을 값.

    a[0] 은 '풀락 대비 비율' 이다. ESP 도 S∈[-1,1] 을 풀락 비율로 읽으므로
    둘은 같은 뜻이다. 다만 control_node 가 중간에서
    S = steering_angle / max_steering_angle_rad 로 정규화하므로, 그 분모를
    그대로 곱해 보내야 S 가 a[0] 이 된다. steer_scale_rad 가 그 분모다.
    """
    a0 = float(np.clip(action[0], -1.0, 1.0))
    a1 = float(np.clip(action[1], -1.0, 1.0))
    steer_rad = steer_sign * a0 * steer_scale_rad
    speed_mps = min((a1 + 1.0) / 2.0 * MAX_SPEED * speed_scale, speed_cap)
    return steer_rad, speed_mps, (a0, a1)
