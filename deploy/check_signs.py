#!/usr/bin/env python3
"""차에 올리기 전, 부호 네 개를 30분 안에 확정한다.

fly_drive_node 의 CHECK 에 있는 값들이다. 전부 +1 아니면 -1 이라
틀리면 차가 반대로 간다. 조용히 틀리지 않고 대놓고 틀리지만, 그때는
이미 벽이다. 띄워 놓고 먼저 본다.

  1) 자이로     차를 손으로 왼쪽(반시계)으로 돌린다
  2) 가속도     차를 앞으로 민다
  3) 라이다     차 왼쪽 1m 쯤에 상자를 놓는다
  4) 조향       **바퀴를 띄우고** 조향 명령을 준다

젯슨에서:
    python3 deploy/check_signs.py            # 1~3 만 (안전)
    python3 deploy/check_signs.py --steer    # 4 까지 (바퀴 띄우고!)
"""

import argparse
import math
import sys
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32, Float64

STEER_TEST_RAD = 0.20      # 약 11도. 풀락 근처까지 안 간다.


class SignChecker(Node):
    def __init__(self, do_steer):
        super().__init__("check_signs")
        self.do_steer = do_steer
        self.imu = None
        self.scan = None
        self.speed = None
        self.servo = None
        self.create_subscription(Imu, "/imu/data", self._imu, 10)
        self.create_subscription(LaserScan, "/scan", self._scan, 10)
        self.create_subscription(Float64, "/vehicle/speed_mps",
                                 lambda m: setattr(self, "speed", m.data), 10)
        self.create_subscription(Float32, "/esp32/servo_command_deg",
                                 lambda m: setattr(self, "servo", m.data), 10)
        self.drive = self.create_publisher(AckermannDriveStamped, "/drive", 10)

    def _imu(self, msg):
        self.imu = msg

    def _scan(self, msg):
        self.scan = msg

    def _send(self, steer_rad, speed=0.0):
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.drive.steering_angle = float(steer_rad)
        m.drive.speed = float(speed)
        self.drive.publish(m)

    def wait(self, what, seconds=5.0):
        end = time.time() + seconds
        while time.time() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if getattr(self, what) is not None:
                return True
        return False


def peak(node, field, seconds, getter):
    """그 시간 동안 절댓값이 가장 컸던 값을 돌려준다."""
    best, end = 0.0, time.time() + seconds
    while time.time() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.02)
        v = getter()
        if v is not None and abs(v) > abs(best):
            best = v
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steer", action="store_true",
                        help="조향 시험까지 한다. 반드시 바퀴를 띄우고!")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = SignChecker(args.steer)
    result = {}

    print("=" * 60)
    print("센서 연결 확인")
    print("=" * 60)
    for topic, field in (("/imu/data", "imu"), ("/scan", "scan")):
        ok = node.wait(field, 5.0)
        print(f"  {topic:<26} {'들어옴' if ok else '안 들어옴 ← 먼저 해결'}")
        if not ok:
            rclpy.try_shutdown()
            return
    if node.scan is not None:
        s = node.scan
        n = len(s.ranges)
        print(f"\n  스캔 점 {n}개   "
              f"{math.degrees(s.angle_min):.1f}도 ~ "
              f"{math.degrees(s.angle_max):.1f}도   "
              f"({math.degrees(s.angle_increment):.3f}도 간격)")
        print(f"  거리 {s.range_min:.2f} ~ {s.range_max:.2f} m")
        bad = sum(1 for r in s.ranges if not math.isfinite(r) or r <= 0)
        print(f"  못 맞은 빔 {bad}개 ({bad / n:.0%})")

    print("\n" + "=" * 60)
    print("1) 자이로 부호 — 차를 손으로 왼쪽(반시계)으로 5초간 돌리세요")
    print("=" * 60)
    input("  준비됐으면 엔터: ")
    v = peak(node, "imu", 5.0,
             lambda: node.imu.angular_velocity.z if node.imu else None)
    result["gyro_sign"] = +1.0 if v > 0 else -1.0
    print(f"  최대 {v:+.3f} rad/s  ->  gyro_sign = {result['gyro_sign']:+.0f}")

    print("\n" + "=" * 60)
    print("2) 가속도 부호 — 차를 앞으로 5초간 미세요")
    print("=" * 60)
    input("  준비됐으면 엔터: ")
    v = peak(node, "imu", 5.0,
             lambda: node.imu.linear_acceleration.x if node.imu else None)
    result["accel_sign"] = +1.0 if v > 0 else -1.0
    print(f"  최대 {v:+.3f} m/s^2  ->  accel_sign = {result['accel_sign']:+.0f}")

    print("\n" + "=" * 60)
    print("3) 라이다 좌우 — 차 왼쪽 1m 쯤에 상자를 놓으세요")
    print("=" * 60)
    input("  놓았으면 엔터: ")
    rclpy.spin_once(node, timeout_sec=1.0)
    s = node.scan
    r = np.asarray(s.ranges, dtype=float)
    ang = s.angle_min + np.arange(len(r)) * s.angle_increment
    near = np.isfinite(r) & (r > s.range_min) & (r < 2.0)
    # 정면 ±90도 안에서 가장 가까운 것
    near &= np.abs(ang) < math.pi / 2
    if near.sum() == 0:
        print("  2m 안에 아무것도 안 보입니다. 상자를 더 가까이.")
        result["scan_sign"] = +1.0
    else:
        i = int(np.argmin(np.where(near, r, np.inf)))
        result["scan_sign"] = +1.0 if ang[i] > 0 else -1.0
        print(f"  가장 가까운 물체 {r[i]:.2f} m, 각도 "
              f"{math.degrees(ang[i]):+.1f}도  ->  "
              f"scan_sign = {result['scan_sign']:+.0f}")
        print("  (왼쪽에 놓았는데 각도가 음수로 나오면 -1 이 맞습니다)")

    if args.steer:
        print("\n" + "=" * 60)
        print("4) 조향 부호 — 바퀴가 떠 있는지 확인하세요!")
        print("=" * 60)
        ans = input("  바퀴가 땅에서 떠 있습니까? (yes 입력): ")
        if ans.strip().lower() != "yes":
            print("  건너뜁니다.")
        else:
            print(f"  왼쪽 명령 {STEER_TEST_RAD:+.2f} rad 를 3초간 보냅니다")
            end = time.time() + 3.0
            while time.time() < end and rclpy.ok():
                node._send(+STEER_TEST_RAD, 0.0)
                rclpy.spin_once(node, timeout_sec=0.02)
            horn = node.servo
            node._send(0.0, 0.0)
            if horn is None:
                print("  /esp32/servo_command_deg 가 안 옵니다. "
                      "눈으로 보고 정하세요.")
            else:
                print(f"  서보 혼 {horn:.0f}도 (중립 90)")
                seen = input("  앞바퀴가 왼쪽으로 돌았습니까? (y/n): ")
                result["steer_sign"] = +1.0 if seen.strip().lower() \
                    .startswith("y") else -1.0
                print(f"  ->  steer_sign = {result['steer_sign']:+.0f}")

    print("\n" + "=" * 60)
    print("fly_drive_node.py 의 CHECK 에 그대로 넣으세요")
    print("=" * 60)
    for k, v in result.items():
        print(f'    "{k}": {v:+.1f},')

    node._send(0.0, 0.0)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
