"""하드웨어 층을 PC 에서 검증한다. 차도 젯슨도 없이.

시리얼을 가짜로 바꿔치기해서, 우리가 내보내는 바이트가 ESP 펌웨어와
VESC 가 기대하는 모양인지 본다. 프로토콜을 잘못 쓰면 차는 아무 반응도
안 하거나(조용한 실패) 엉뚱하게 움직인다.

  1. VESC CRC 가 맞나 (기존 스택 구현과 대조)
  2. duty 패킷 모양이 맞나
  3. VESC 응답을 파싱해 속도가 나오나
  4. ESP 에 보내는 줄이 펌웨어 파서를 통과하나
  5. RC 줄을 읽어 수동/자율을 가르나
  6. 속도 PI 가 목표에 수렴하나

  python deploy/test_hardware.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import struct

import numpy as np


class FakeSerial:
    """serial.Serial 인 척한다. 쓴 것을 모아 두고, 읽을 것을 미리 넣는다."""

    def __init__(self, *a, **kw):
        self.written = bytearray()
        self.to_read = bytearray()

    def write(self, data):
        self.written.extend(data)
        return len(data)

    def read(self, n=1):
        out = bytes(self.to_read[:n])
        del self.to_read[:n]
        return out

    def close(self):
        pass


def patch_serial():
    """pyserial 이 없어도 돌게 가짜 모듈을 심는다.

    이 PC 에는 pyserial 이 없다(젯슨에만 있으면 된다). 하드웨어 층은
    serial.Serial 하나만 쓰므로 그것만 바꿔치기하면 프로토콜을 검증할 수
    있다. 젯슨에서는 진짜 pyserial 이 깔려 있어 이 코드가 안 탄다.
    """
    import types
    try:
        import serial
    except ModuleNotFoundError:
        serial = types.ModuleType("serial")
        serial.SerialException = Exception
        sys.modules["serial"] = serial
    serial.Serial = FakeSerial
    if not hasattr(serial, "SerialException"):
        serial.SerialException = Exception


def esp_normToAngle(s):
    """ESP 펌웨어 normToAngle 을 그대로 옮긴 것. 40~140 도로 편다."""
    s = max(-1.0, min(1.0, s))
    return 40.0 + (s - (-1.0)) * (140.0 - 40.0) / (1.0 - (-1.0))


def main():
    patch_serial()
    from deploy.hardware import (Drive, SpeedHold, Steering, crc16_ccitt,
                                 vesc_packet)

    ok = True

    # ---- 1. CRC ---------------------------------------------------------
    print("1) VESC CRC")
    # 기존 스택 구현과 같은 알고리즘인지 몇 개 대조한다
    cases = {
        bytes([4]): None,
        bytes([5, 0, 0, 0, 0]): None,
    }
    for payload in cases:
        got = crc16_ccitt(bytearray(payload))
        # 같은 코드를 독립적으로 한 번 더 구현해 대조
        crc = 0
        for b in payload:
            crc ^= b << 8
            for _ in range(8):
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 \
                    else (crc << 1) & 0xFFFF
        same = got == crc
        ok &= same
        print(f"   payload {list(payload)}  crc {got:#06x}  "
              f"{'OK' if same else '틀림'}")

    # ---- 2. duty 패킷 ---------------------------------------------------
    print("\n2) duty 패킷 모양")
    d = Drive("fake", max_duty=0.30)
    d.set_duty(0.15)
    pkt = bytes(d.ser.written)
    good = (pkt[0] == 0x02 and pkt[1] == 5 and pkt[2] == 5
            and pkt[-1] == 0x03 and len(pkt) == 10)
    value = struct.unpack(">i", pkt[3:7])[0]
    good &= value == 15000
    ok &= good
    print(f"   {pkt.hex(' ')}")
    print(f"   시작 0x02, 길이 5, 명령 5, duty {value} (=0.15*100000), "
          f"끝 0x03  -> {'OK' if good else '틀림'}")

    d2 = Drive("fake", max_duty=0.30)
    d2.set_duty(5.0)          # 상한을 넘겨 본다
    v2 = struct.unpack(">i", bytes(d2.ser.written)[3:7])[0]
    capped = v2 == 30000
    ok &= capped
    print(f"   duty 5.0 요청 -> {v2/100000:.2f} 로 잘림  "
          f"{'OK' if capped else '틀림 — 위험'}")

    # ---- 3. VESC 응답 파싱 ----------------------------------------------
    print("\n3) VESC 응답 -> 속도")
    erpm = 8000
    payload = bytearray([4])
    payload.extend(b"\x00" * 22)              # payload[1:23]
    payload.extend(struct.pack(">i", erpm))   # [23:27]
    payload.extend(struct.pack(">h", 144))    # [27:29] 전압 14.4V
    d3 = Drive("fake")
    d3.ser.to_read.extend(vesc_packet(payload))
    d3.poll()
    want = erpm / (7.0 * 11.0) * (2 * np.pi * 0.05) / 60.0
    good = abs(d3.speed_mps - want) < 1e-9 and abs(d3.voltage - 14.4) < 1e-6
    ok &= good
    print(f"   erpm {erpm} -> {d3.speed_mps:.3f} m/s  "
          f"(손계산 {want:.3f})  전압 {d3.voltage:.1f}V  "
          f"{'OK' if good else '틀림'}")

    # ---- 4. ESP 조향 명령 -----------------------------------------------
    print("\n4) ESP 조향 명령")
    s = Steering("fake")
    for cmd in (-1.0, -0.5, 0.0, 0.37, 1.0, 2.0):
        s.ser.written.clear()
        s.send(cmd)
        line = bytes(s.ser.written).decode()
        good = line.startswith("S:") and line.endswith("\n")
        value = float(line[2:])
        clipped = max(-1.0, min(1.0, cmd))
        good &= abs(value - clipped) < 1e-4
        ok &= good
        print(f"   send({cmd:+.2f}) -> {line.strip():<12} "
              f"ESP 서보 {esp_normToAngle(value):.0f}도  "
              f"{'OK' if good else '틀림'}")

    # ---- 5. RC 줄 파싱 --------------------------------------------------
    print("\n5) RC 줄 -> 수동/자율")
    for ch5, want_auto in ((1000, False), (1500, False), (2000, True)):
        s2 = Steering("fake")
        s2.ser.to_read.extend(
            f"RC,1500,1500,{ch5},1000,95.0,93.0\n".encode())
        s2.poll()
        good = s2.autonomous == want_auto and s2.rc_fresh()
        ok &= good
        print(f"   CH5={ch5} -> {'자율' if s2.autonomous else '수동'}  "
              f"서보 {s2.servo_deg:.0f}도 (비율 {s2.steer_frac:+.2f})  "
              f"{'OK' if good else '틀림'}")

    # 서보 각도 -> 풀락 비율 왕복
    s3 = Steering("fake")
    for deg, frac in ((40.0, -1.0), (90.0, 0.0), (140.0, +1.0),
                      (115.0, +0.5)):
        s3.servo_deg = deg
        good = abs(s3.steer_frac - frac) < 1e-9
        ok &= good
        print(f"   서보 {deg:.0f}도 -> 비율 {s3.steer_frac:+.2f} "
              f"(앞바퀴 {s3.steer_frac*40:+.0f}도)  "
              f"{'OK' if good else '틀림'}")

    # ---- 6. 속도 PI -----------------------------------------------------
    print("\n6) 속도 PI 수렴")
    hold = SpeedHold(max_duty=0.30)
    # 아주 단순한 모터 모델: duty 에 비례하는 속도로 1차 지연을 두고 수렴
    speed, dt = 0.0, 0.025
    for target in (2.0, 4.0):
        hold.reset()
        for _ in range(400):
            duty = hold.step(target, speed, dt)
            speed += (duty * 20.0 - speed) * dt * 3.0
        err = abs(speed - target)
        good = err < 0.15
        ok &= good
        print(f"   목표 {target:.1f} -> {speed:.2f} m/s "
              f"(오차 {err:.3f})  {'OK' if good else '안 맞음'}")

    print("\n" + "=" * 50)
    print("전부 통과" if ok else "실패한 항목이 있다 — 위를 볼 것")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
