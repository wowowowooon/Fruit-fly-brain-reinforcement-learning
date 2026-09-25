"""목표 속도를 VESC duty 로 바꾼다. 모터 이득을 알고 쓴다.

왜 새로 쓰나
-----------
처음엔 흔한 PI 로 짰는데 시험에서 목표를 3.7 m/s 나 넘겼다. 6.5 를
요구했는데 10.2 가 나온다. 코너에서 그러면 그대로 벽이다.

원인은 제어기가 모터 이득을 몰랐다는 것이다.

    ff(0.075) x 6.5 = 0.49 에 kp 보정이 더해져 duty 0.76
    -> 상한 0.60 으로 잘림 -> 13.3 x 0.60 = 8.0 m/s 로 수렴
    게다가 올라가는 동안 적분이 쌓여 감속 명령에도 안 풀린다

그래서 이렇게 고쳤다.

    1. duty 의 몸통은 target / gain 이다. 이득을 실측했으니 계산하면 된다.
       (실차 속도 계단 시험 14개에서 13.3 m/s per duty, 0.20초)
    2. 피드백은 '보정' 일 뿐이라 +-CORRECT_MAX 로 묶는다. 이득이 좀 틀려도
       메울 만큼이면 충분하고, 그 이상 벌어지면 계산이 틀린 것이다.
    3. 포화 중에는 적분을 안 쌓는다 (안티와인드업). 안 하면 감속할 때
       쌓인 적분이 안 풀려 계속 밀어붙인다.
    4. 목표보다 빠르면 duty 를 음수로 준다. VESC 는 음수 duty 를 역토크로
       쓴다. 타력으로만 줄이면 코너에 속도를 안고 들어간다.

이득은 배터리 전압과 노면에 따라 변한다. 그래서 보정을 남겨 둔다 —
계산만으로 맞추려 하면 전압이 떨어질 때 느려진다.
"""

# 실차 속도 계단 시험 14개에서 뽑은 값 (vesc_speed_test_logs/*.csv)
#   ds/dt = (GAIN * duty - s) / TAU
GAIN_MPS_PER_DUTY = 13.3        # 11.7 ~ 13.9 로 일관됐다
TAU_SEC = 0.20                  # 0.11 ~ 0.30

# 피드백이 건드릴 수 있는 duty 폭. 이득이 20% 틀려도 메운다.
CORRECT_MAX = 0.12
# 목표보다 빠를 때 거는 역토크. duty per (m/s) 초과.
BRAKE_PER_MPS = 0.045
BRAKE_MAX = 0.20


class SpeedControl:
    """목표 속도 -> duty.

        ctrl = SpeedControl(max_duty=0.60)
        duty = ctrl.step(target_mps, measured_mps, dt)
    """

    def __init__(self, max_duty=0.60, gain=GAIN_MPS_PER_DUTY,
                 kp=0.030, ki=0.20, correct_max=CORRECT_MAX,
                 brake=True, brake_per_mps=BRAKE_PER_MPS,
                 brake_max=BRAKE_MAX):
        self.max_duty = float(max_duty)
        self.gain = float(gain)
        self.kp, self.ki = float(kp), float(ki)
        self.correct_max = float(correct_max)
        self.brake = bool(brake)
        self.brake_per_mps = float(brake_per_mps)
        self.brake_max = float(brake_max)
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def step(self, target, actual, dt):
        if target <= 0.02:
            self.integral = 0.0
            # 정지 명령이면 역토크로 세운다
            return -min(self.brake_max, self.brake_per_mps * max(actual, 0.0)) \
                if self.brake else 0.0

        error = target - actual

        # 목표보다 빠르다 -> 역토크. 앞먹임을 끄고 브레이크만 건다.
        if self.brake and error < -0.15:
            self.integral = 0.0
            return -min(self.brake_max, self.brake_per_mps * (-error))

        # duty 의 몸통. 이득을 알면 계산으로 나온다.
        base = target / self.gain
        # 보정은 좁게 묶는다
        correct = self.kp * error + self.ki * self.integral
        correct = max(-self.correct_max, min(self.correct_max, correct))
        duty = base + correct

        # 목표가 요구하는 duty 보다 크게 벌어지지 못하게 한다.
        # 안 막으면 보정만으로 목표보다 높은 속도에 수렴한다 — 예를 들어
        # 목표 6.5(duty 0.489)에 보정 +0.12 가 붙으면 0.609 로 잘려
        # 13.3 x 0.60 = 8.0 m/s 에 앉는다. 이득이 틀릴 여지(35%)만 남긴다.
        cap = min(self.max_duty, base * 1.35 + 0.02)

        # 포화 중에는 적분을 안 쌓는다. 쌓으면 감속 때 안 풀린다.
        saturated = duty >= cap or duty <= 0.0
        if not (saturated and error * (duty - base) > 0):
            self.integral += error * dt
        limit = self.correct_max / max(self.ki, 1e-6)
        self.integral = max(-limit, min(limit, self.integral))

        return max(0.0, min(cap, duty))
