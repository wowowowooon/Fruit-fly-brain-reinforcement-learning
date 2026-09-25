#!/usr/bin/env python3
"""T안을 실차에서 굴리는 노드. 이것 하나로 끝난다.

    /scan (LaserScan)  ─┐
                        ├→ [이 노드] → ESP32 시리얼 "S:<-1~1>"  (조향)
    /imu/data (Imu)    ─┘              VESC 시리얼 duty 패킷    (구동)

기존 스택에서 안 쓰는 것
    stanley / local_planner / fgm / static_obstacle   경로를 안 쓴다
    cartographer / amcl / tf_manager                  지도·위치추정을 안 쓴다
    control_node                                      여기에 흡수했다

T안은 라이다와 IMU 만 보고 바로 조향·속도를 낸다. 그래서 중간 단계가
전부 필요 없다. 남은 것은 센서 드라이버 둘과 이 노드뿐이다.

안전장치는 그대로 가져왔다
    RC CH5 로 사람이 언제든 수동으로 뺏어간다 (자율은 CH5 >= 1700 일 때만)
    스캔이 0.3초 끊기면 정지
    RC 가 0.3초 끊기면 정지 (송신기가 꺼졌거나 선이 빠진 것)
    Ctrl-C 나 예외에도 반드시 duty 0 + 조향 중립

띄우는 법
    ros2 run sllidar_ros2 sllidar_node --ros-args --params-file <sensor_params.yaml>
    ros2 run ebimu_pkg   ebimu_driver  --ros-args --params-file <sensor_params.yaml>
    python3 deploy/fly_drive_node.py --speed-scale 0.3      # 첫 주행

차에서 확인할 값은 CHECK 에 모아 뒀다. 전부 부호 하나다.
    python3 deploy/check_signs.py --steer      # 바퀴 띄우고
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import Imu, LaserScan

from deploy.hardware import Drive, Steering, find_ports
from deploy.speed_ctrl import SpeedControl
from deploy.obs_bridge import build_obs, resample_scan
from flyracer.drive.overtake import Overtaker
from flyracer.brain.fly_driver import FlyDriver
from flyracer.sim.f1tenth_sim import LIDAR_RANGE, MAX_SPEED, MAX_STEER

# ============================================================
# 차에서 확인할 값 — 전부 부호 하나다. check_signs.py 가 찍어 준다.
# ============================================================
CHECK = {
    # 차를 왼쪽(반시계)으로 돌렸을 때 angular_velocity.z 가 양수면 +1
    "gyro_sign": +1.0,
    # 차를 앞으로 밀 때 linear_acceleration.x 가 양수면 +1
    "accel_sign": +1.0,
    # 차 왼쪽에 놓은 물체가 양수 각도로 찍히면 +1 (ROS 규약은 보통 +1)
    "scan_sign": +1.0,
    # 바퀴를 띄우고 S=+0.3 을 줬을 때 앞바퀴가 왼쪽으로 돌면 +1
    "steer_sign": +1.0,
}

CFG = {
    "scan_topic": "/scan",
    "imu_topic": "/imu/data",
    # 학습을 40Hz 로 했다. CfC 는 dt 를 직접 쓰는 순환망이라 주기가 어긋나면
    # 같은 가중치라도 다르게 움직인다. 스캔이 오는 대로 돌리되 dt 는 실제
    # 경과시간을 재서 넘긴다.
    "fallback_dt": 1.0 / 40.0,
    "scan_timeout": 0.30,
    "rc_timeout": 0.30,
    # VESC duty 상한.
    # 실차 속도 계단 시험 14개에서 모터 이득이 13.3 m/s per duty 로 나왔다.
    # 0.30 으로 두면 정상상태가 4.0 m/s 라 T안이 7 m/s 를 요구해도 못 낸다.
    # 7 m/s 를 내려면 0.53 이 필요하다. 여유를 조금 둬 0.60.
    # 첫 주행은 --max-duty 0.25 처럼 낮춰서 본다.
    "max_duty": 0.60,
    # VESC 값 요청 주기 (속도를 여기서 얻는다)
    "vesc_poll_hz": 50.0,
}


class FlyDriveNode(Node):
    def __init__(self, args):
        super().__init__("fly_drive_node")
        self.args = args

        self.driver = FlyDriver(n_beams=args.beams, lidar_range=LIDAR_RANGE,
                                fly_wiring=True, motion=args.motion)
        self.driver.load(args.model)

        esp_port, vesc_port = find_ports(args.esp_port, args.vesc_port)
        self.get_logger().info(f"ESP32 {esp_port}   VESC {vesc_port}")
        self.steer_io = Steering(esp_port)
        self.drive_io = Drive(vesc_port, max_duty=args.max_duty)
        self.hold = SpeedControl(max_duty=args.max_duty)
        # 앞차를 지나갈 자리가 없을 때 참는 층. 기본은 꺼 둔다.
        # 대회 코스처럼 좁은 데서는 박는 횟수를 절반쯤 줄이지만
        # 랩타임을 6~10% 쓰고, 넓은 트랙에서는 이득이 없다.
        self.over = (Overtaker(lidar_range=LIDAR_RANGE)
                     if args.patience else None)

        self.scan = None
        self.yaw_rate = 0.0
        self.accel = 0.0
        self.last_action = [0.0, 0.0]
        self.last_tick = None
        self.last_scan_at = 0.0
        self.last_poll = 0.0
        self.was_auto = False
        self.said = set()
        # 참는 층이 쓸 빔 각도. 모델 격자와 같다.
        from deploy.obs_bridge import model_angles
        self._angles = model_angles(args.beams)

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                         durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(LaserScan, CFG["scan_topic"], self._on_scan,
                                 qos)
        self.create_subscription(Imu, CFG["imu_topic"], self._on_imu, qos)
        # 시리얼은 스캔과 무관하게 계속 읽어야 RC 가 끊긴 걸 알 수 있다
        self.create_timer(0.01, self._tick)
        self.get_logger().info(
            f"모델 {args.model}  빔 {args.beams}  속도배율 {args.speed_scale}")
        self.get_logger().info("CH5 를 자율(위쪽)로 올려야 움직인다")

    # ---- 센서 ----------------------------------------------------------
    def _on_imu(self, msg):
        self.yaw_rate = CHECK["gyro_sign"] * float(msg.angular_velocity.z)
        self.accel = CHECK["accel_sign"] * float(msg.linear_acceleration.x)

    def _on_scan(self, msg):
        self.scan = msg
        self.last_scan_at = time.monotonic()

    # ---- 본체: 10ms 마다 돈다 -------------------------------------------
    def _tick(self):
        now = time.monotonic()
        self.steer_io.poll()
        self.drive_io.poll()
        if now - self.last_poll > 1.0 / CFG["vesc_poll_hz"]:
            self.drive_io.request_values()
            self.last_poll = now

        why = self._blocked(now)
        if why:
            self._stop(why)
            return
        if not self.was_auto:
            self.get_logger().info("자율 시작")
            self.hold.reset()
            self.driver.reset()
            if self.over is not None:
                self.over.reset()
            self.last_tick = None
            self.was_auto = True

        scan = resample_scan(self.scan.ranges, self.scan.angle_min,
                             self.scan.angle_increment, self.args.beams,
                             self.scan.range_min, CHECK["scan_sign"])
        if scan is None:
            self._stop("스캔이 이상하다")
            return

        dt = CFG["fallback_dt"]
        if self.last_tick is not None:
            gap = now - self.last_tick
            if 1e-4 < gap < 0.5:
                dt = gap
        self.last_tick = now
        self.driver.dt = dt

        speed = self.drive_io.speed_mps
        steer_now = CHECK["steer_sign"] * self.steer_io.steer_frac
        obs = build_obs(scan, speed, self.yaw_rate, self.accel,
                        steer_now * MAX_STEER, self.last_action,
                        self.args.motion)

        action = self.driver.act(obs)
        if self.over is not None:
            action = self.over.apply(action, scan, self._angles, speed, dt)
        a0 = float(np.clip(action[0], -1.0, 1.0))
        a1 = float(np.clip(action[1], -1.0, 1.0))
        self.last_action = [a0, a1]

        # a[0] 은 풀락 비율. ESP 의 S 도 풀락 비율이다. 그대로 보낸다.
        self.steer_io.send(CHECK["steer_sign"] * a0)

        target = (a1 + 1.0) / 2.0 * MAX_SPEED * self.args.speed_scale
        target = min(target, self.args.speed_cap)
        self.drive_io.set_duty(self.hold.step(target, speed, dt))

    def _blocked(self, now):
        """달리면 안 되는 이유. None 이면 달려도 된다.

        하나라도 걸리면 무조건 막는다. 여기서 조건을 느슨하게 하면 사람이
        수동으로 뺏어 온 상태에서도 차가 움직인다.
        """
        if not self.steer_io.rc_fresh(CFG["rc_timeout"]):
            return "RC 가 안 온다"
        if not self.steer_io.autonomous:
            return "수동 모드 (CH5 를 올리세요)"
        if self.scan is None:
            return "스캔을 기다린다"
        if now - self.last_scan_at > CFG["scan_timeout"]:
            return "스캔이 끊겼다"
        return None

    def _stop(self, why):
        self.drive_io.set_duty(0.0)
        self.steer_io.send(0.0)
        self.hold.reset()
        if self.was_auto:
            # 달리다 멈춘 것은 항상 알린다
            self.get_logger().warn(f"정지: {why}")
            self.was_auto = False
            self.said.clear()
        elif why not in self.said:
            # 가만히 있을 때는 이유가 바뀔 때만 한 번씩
            self.said.add(why)
            self.get_logger().info(f"대기: {why}")

    def shutdown(self):
        self.drive_io.close()
        self.steer_io.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="T_fly871_final.pt")
    parser.add_argument("--motion", type=int, default=6)
    parser.add_argument("--beams", type=int, default=541,
                        help="모델이 학습한 빔 수. 실제 스캔은 이 수로 다시 묶인다")
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        help="첫 주행은 0.3 으로")
    parser.add_argument("--speed-cap", type=float, default=7.0)
    parser.add_argument("--max-duty", type=float, default=CFG["max_duty"])
    parser.add_argument("--patience", action="store_true",
                        help="앞차를 지나갈 자리가 없으면 참는다. 좁은 코스에서 "
                             "박는 횟수가 줄지만 랩타임을 6~10%% 쓴다")
    parser.add_argument("--esp-port", default="auto")
    parser.add_argument("--vesc-port", default="auto")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = FlyDriveNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
