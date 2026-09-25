"""ESP32(조향)와 VESC(구동)에 직접 말한다. ROS 없이 돌아간다.

기존 스택의 control_node 는 1683줄인데 대부분 Stanley 스케일 환산, CSV
경로, 튜닝용 곁가지다. T안은 그걸 하나도 안 쓴다. 필요한 것만 남겼다.

프로토콜은 그쪽 코드와 ESP 펌웨어에서 그대로 가져왔다.

  ESP32 (USB-C, 115200)
      보낼 때   "S:<-1~1>\\n"   -> normToAngle 이 서보 40~140도로 편다
      받을 때   "RC,ch1,ch2,ch5,ch6,target_deg,servo_deg,..."
                ch5 <= 1300 수동 / >= 1700 자율

  VESC (115200)
      duty 명령 = 0x02 | len | [5, int32(duty*100000)] | crc16 | 0x03
      값 요청   = 같은 틀에 payload [4] (COMM_GET_VALUES)
      응답에서  erpm = payload[23:27] (>i)

속도는 erpm 에서 되돌린다.
      바퀴 rpm = erpm / (극쌍 7 * 기어비 11)
      속도     = 바퀴 rpm * 2*pi*0.05 / 60
"""

import math
import struct
import time

# serial 은 쓸 때 불러온다. CRC 나 속도 PI 같은 순수 계산은 pyserial 이
# 없는 PC 에서도 써야 하는데, 최상단에서 import 하면 그것까지 막힌다.

# ---- 차 제원 (jetson_code 의 metadata.yaml 실측) ----
POLE_PAIRS = 7.0
GEAR_RATIO = 11.0
WHEEL_RADIUS_M = 0.05
WHEEL_CIRC_M = 2.0 * math.pi * WHEEL_RADIUS_M

# ---- ESP32 서보 (esp32_steer_rc.ino 와 같아야 한다) ----
SERVO_CENTER_DEG = 90.0
SERVO_HALF_DEG = 50.0          # 40~140

# ---- RC 모드 (ESP 펌웨어와 같아야 한다) ----
MODE_MANUAL_US = 1300
MODE_AUTO_US = 1700


def find_ports(esp="auto", vesc="auto"):
    """시리얼 포트를 찾는다. 꽂는 순서에 따라 번호가 바뀌므로 이름으로 찾는다.

    /dev/serial/by-id 는 장치가 스스로 알리는 이름이라 재부팅해도 같다.
    VESC 는 ChibiOS 로, ESP32 는 보드 USB 칩 이름으로 잡힌다.
    """
    import glob
    import os

    def pick(want, keys):
        if want != "auto":
            return want
        for path in sorted(glob.glob("/dev/serial/by-id/*")):
            low = os.path.basename(path).lower()
            if any(k in low for k in keys):
                return path
        raise RuntimeError(
            f"{keys[0]} 포트를 못 찾았다. ls /dev/serial/by-id 로 확인하고 "
            f"--esp-port / --vesc-port 로 직접 지정할 것")

    return (pick(esp, ("esp", "cp210", "ch340", "ch910", "usb-jtag")),
            pick(vesc, ("chibios", "vesc", "stmicro")))


def crc16_ccitt(data):
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 \
                else (crc << 1) & 0xFFFF
    return crc


def vesc_packet(payload):
    out = bytearray([0x02, len(payload)])
    out.extend(payload)
    crc = crc16_ccitt(payload)
    out.append((crc >> 8) & 0xFF)
    out.append(crc & 0xFF)
    out.append(0x03)
    return bytes(out)


class Steering:
    """ESP32. 조향 명령을 보내고 RC 상태를 읽는다."""

    def __init__(self, port, baud=115200):
        import serial
        self._serial = serial
        self.ser = serial.Serial(port, baud, timeout=0)
        self.buf = bytearray()
        self.ch = [0, 0, 0, 0]          # ch1, ch2, ch5, ch6
        self.servo_deg = SERVO_CENTER_DEG
        self.last_rc = 0.0

    def send(self, s):
        """s 는 -1~1. 그대로 앞바퀴 풀락 비율이다."""
        s = max(-1.0, min(1.0, float(s)))
        self.ser.write(f"S:{s:.4f}\n".encode())

    def poll(self):
        """ESP 가 보내는 RC 줄을 읽는다. 수동/자율 판단에 쓴다."""
        try:
            self.buf.extend(self.ser.read(512))
        except self._serial.SerialException:
            return
        while b"\n" in self.buf:
            line, _, rest = self.buf.partition(b"\n")
            self.buf = bytearray(rest)
            text = line.decode("ascii", "ignore").strip()
            if not text.startswith("RC,"):
                continue
            parts = text.split(",")
            try:
                self.ch = [int(float(parts[i])) for i in (1, 2, 3, 4)]
                if len(parts) >= 7:
                    self.servo_deg = float(parts[6])
            except (ValueError, IndexError):
                continue
            self.last_rc = time.monotonic()

    @property
    def steer_frac(self):
        """서보가 실제로 간 각도 -> 풀락 비율. 조향 센서 대용이다."""
        horn = self.servo_deg - SERVO_CENTER_DEG
        return max(-1.0, min(1.0, horn / SERVO_HALF_DEG))

    @property
    def autonomous(self):
        """CH5 가 자율 쪽인가. 사람이 언제든 뺏어갈 수 있어야 한다."""
        return self.ch[2] >= MODE_AUTO_US

    def rc_fresh(self, timeout=0.3):
        return (time.monotonic() - self.last_rc) < timeout

    def close(self):
        try:
            self.send(0.0)
            self.ser.close()
        except Exception:
            pass


class Drive:
    """VESC. duty 를 보내고 erpm 을 읽어 속도를 낸다."""

    def __init__(self, port, baud=115200, max_duty=0.30):
        import serial
        self._serial = serial
        self.ser = serial.Serial(port, baud, timeout=0)
        self.buf = bytearray()
        self.max_duty = float(max_duty)
        self.erpm = 0.0
        self.voltage = 0.0
        self.last_rx = 0.0
        self._last_int = None
        self._last_packet = None

    def set_duty(self, duty):
        duty = max(-self.max_duty, min(self.max_duty, float(duty)))
        value = int(duty * 100000)
        if value != self._last_int:
            payload = bytearray([5])
            payload.extend(struct.pack(">i", value))
            self._last_packet = vesc_packet(payload)
            self._last_int = value
        self.ser.write(self._last_packet)

    def request_values(self):
        self.ser.write(vesc_packet(bytearray([4])))

    def poll(self):
        try:
            self.buf.extend(self.ser.read(1024))
        except self._serial.SerialException:
            return
        while len(self.buf) >= 5:
            if self.buf[0] != 0x02:
                del self.buf[0]
                continue
            n = self.buf[1]
            total = n + 5
            if len(self.buf) < total:
                return
            packet = bytes(self.buf[:total])
            del self.buf[:total]
            payload = packet[2:2 + n]
            got = (packet[2 + n] << 8) | packet[3 + n]
            if crc16_ccitt(payload) != got or not payload or payload[0] != 4:
                continue
            try:
                self.erpm = float(struct.unpack(">i", payload[23:27])[0])
                self.voltage = struct.unpack(">h", payload[27:29])[0] / 10.0
                self.last_rx = time.monotonic()
            except struct.error:
                pass

    @property
    def speed_mps(self):
        wheel_rpm = self.erpm / (POLE_PAIRS * GEAR_RATIO)
        return wheel_rpm * WHEEL_CIRC_M / 60.0

    def fresh(self, timeout=0.3):
        return (time.monotonic() - self.last_rx) < timeout

    def close(self):
        try:
            self.set_duty(0.0)
            self.ser.close()
        except Exception:
            pass


class SpeedHold:
    """목표 속도를 duty 로 바꾸는 PI. 모터 특성을 모르니 피드백으로 맞춘다."""

    def __init__(self, kp=0.06, ki=0.25, max_duty=0.30, ff=0.035):
        self.kp, self.ki, self.ff = kp, ki, ff
        self.max_duty = max_duty
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def step(self, target, actual, dt):
        if target <= 0.02:
            self.integral = 0.0
            return 0.0
        error = target - actual
        self.integral += error * dt
        # 적분이 쌓여 풀리지 않는 것(와인드업)을 막는다
        limit = self.max_duty / max(self.ki, 1e-6)
        self.integral = max(-limit, min(limit, self.integral))
        duty = self.ff * target + self.kp * error + self.ki * self.integral
        return max(0.0, min(self.max_duty, duty))
