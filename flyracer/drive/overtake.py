"""앞차를 지나갈 자리가 없으면 참게 만든다. 조향은 건드리지 않는다.

측정이 설계를 세 번 고쳤다
-------------------------
1차 — '옆으로 더 밀면 추월이 되겠지' 로 만들었다. 성적이 떨어졌다
      (상대차 38->30, 일반 45->42). 왜 그런지 몰라서 박기 직전 3초를 찍었다.

        느린 상대, 박은 22번의 평균
                        상대거리  상대각   내속도  벽여유
            3~2초 전     2.67m   +25도    4.94   +0.352m
            2~1초 전     1.34m   +47도    2.09   +0.257m
            1~0초 전     0.88m   +50도    3.05   -0.017m
            박는 순간    0.47m   +97도

      상대각 +97도 — 상대가 정확히 옆에 있다. T안은 갇힌 게 아니라 이미
      추월을 시도하다 벽에 박았다. 그러니 필요한 건 미는 힘이 아니라
      참는 힘이다. 방향이 정반대였다.

2차 — 속도만 묶는 '참는 층' 으로 바꿨다. 망가뜨리진 않았지만(장애물만
      있을 때 0회 유지, 일반 45/45 유지) 발동을 한 번도 안 했다.
      조건별로 세 보니 통과 폭 판정이 늘 상한(2.00m)을 냈다. 모서리에서
      바깥으로 나가며 '빔이 덩어리보다 가까워질 때까지' 를 셌는데, 상대차
      너머의 벽은 당연히 더 멀어서 한계로 잡히지 않았다. 거리끼리 비교하면
      안 되고 가로 거리를 재야 했다.

3차 — 가로 거리로 고치니 0.96m 로 현실적인 값이 나왔다. 그런데 여전히
      두 군데가 남았다.

        기준 0.70m 이 느슨하다. 차는 폭 0.30m 지만 길이가 0.60m 라
        코너에서 휩쓰는 폭이 0.53m 다 (반경 0.9m 에서).

        느린 상대는 검출이 91% 실패했다. 좁은 구간에서 상대가 벽에 붙으면
        덩어리 분절이 벽으로 오인한다.

      그래서 지금은 분절에 의존하지 않는다. 정면 원뿔의 최근접 거리를
      프레임마다 이어 '앞의 것이 얼마나 천천히 가까워지나' 를 본다. 벽과
      콘은 우리 속도만큼 가까워지고, 앞차는 그보다 느리게 가까워진다.
      그게 상대 속도다. 분절이 실패해도 이건 된다.

이 층이 지키는 것
    조향을 절대 건드리지 않는다. 벽으로 밀 위험이 원천적으로 없다.
    속도 상한만 건다. 최악이라도 느려질 뿐이다.
    움직이는 것에만 발동한다. 정지 장애물은 T안이 이미 완벽하다
    (장애물 2개 놓고 17.8바퀴 무사고). 거기 손대면 망가진다.
"""

import numpy as np

from flyracer.sim.f1tenth_sim import CAR_WIDTH, FRONT_OVERHANG, MAX_SPEED

# --- 정면 원뿔 (검출) ----------------------------------------------------
FRONT_ANGLE = np.deg2rad(22)   # 이 안의 최근접 거리를 앞의 것으로 본다
# 2.6m 로는 늦다. 진단에서 박기 3초 전 상대가 2.67m 였고 우리는 4.94 m/s
# 였다. 그 거리에서 속도를 묶기 시작하면 구동계가 따라오기 전에 붙는다.
# 4.5m 면 같은 상황에서 1.5초 앞서 손을 뗀다.
BLOCK_RANGE = 4.5              # [m]
MOVING_MIN = 0.9               # [m/s] 이보다 빠르면 '움직이는 것'
SLOW_RATIO = 1.05              # 상대가 내 속도의 이 비율보다 느릴 때
# 상대 속도 추정을 매끄럽게. 낮을수록 안정, 높을수록 빠르게 반응.
BLEND = 0.25
# 증거를 모아서 켠다. 연속 N프레임을 요구하면(4차에서 10프레임) 코너 벽
# 오인은 33%->1% 로 잡히지만 진짜 상대에도 발동이 0% 가 됐다. 네 조건을
# 매 프레임 동시에 만족하기를 바라는 게 무리다 — 통과 폭 판정이 프레임마다
# 흔들린다. 그래서 맞을 때 올리고 틀릴 때 내리는 저울로 바꾼다.
EVIDENCE_UP = 1.0
EVIDENCE_DOWN = 0.7
EVIDENCE_ON = 6.0              # 이만큼 모이면 켠다 (약 0.2초)
EVIDENCE_MAX = 14.0

# --- 지나갈 수 있나 ------------------------------------------------------
SLAB = 0.45                # [m] 그 깊이 앞뒤로 이만큼을 한 층으로
# 차는 폭 0.30m 지만 길이가 0.50m(앞) 라 코너에서 휩쓰는 폭이 더 크다.
#   반경 0.9m -> 0.526   1.5m -> 0.448   직선 -> 0.300
# 추월은 보통 완만한 코너에서 일어나므로 반경 1.5m 를 기준으로 잡는다.
SWEPT_WIDTH = 2.0 * (np.hypot(FRONT_OVERHANG, 1.5 + CAR_WIDTH / 2) - 1.5)
PASS_CLEAR = 0.22          # [m] 양옆에 남겨 둘 여유
PASS_NEED = SWEPT_WIDTH + 2 * PASS_CLEAR        # 약 0.89 m

# --- 참을 때 ------------------------------------------------------------
FOLLOW_GAP = 0.9           # [m] 상대 뒤에 이만큼은 띄운다
CAP_RELAX = 0.12           # 상한을 풀 때 (천천히)
CAP_TIGHTEN = 0.5          # 상한을 걸 때 (빠르게)
CAP_FLOOR = 0.8            # [m/s] 이보다 낮게는 안 묶는다. 서 버리면 안 된다.


def front_range(scan, angles, lidar_range):
    """정면 원뿔 안에서 가장 가까운 거리 [m]."""
    cone = np.abs(angles) <= FRONT_ANGLE
    if not cone.any():
        return lidar_range
    return float(np.min(scan[cone]))


def pass_room(scan, angles, depth):
    """그 깊이에서 지나갈 수 있는 가장 넓은 가로 틈 [m].

    빔을 차 좌표(x 앞, y 왼쪽)로 펴고 그 깊이의 한 층만 잘라 본다. 그 층에
    찍힌 점들의 y 를 늘어놓으면, 이웃한 두 점 사이가 곧 빈 틈이다. 그 중
    가장 넓은 것이 우리가 노릴 수 있는 통로다.

    덩어리를 따로 안 가른다. 분절이 실패해도 이 계산은 된다.
    """
    x = scan * np.cos(angles)
    y = scan * np.sin(angles)
    slab = np.abs(x - depth) <= SLAB
    ys = np.sort(y[slab])
    if len(ys) < 2:
        return 2.0
    return float(min(2.0, np.max(np.diff(ys))))


class Overtaker:
    """지나갈 자리가 없으면 속도 상한을 걸어 참게 한다.

        over = Overtaker()
        action = over.apply(driver.act(obs), scan, angles, speed, dt)

    조향은 손대지 않는다. 상한이 안 걸릴 때는 action 을 그대로 준다.
    """

    def __init__(self, lidar_range=10.0, follow_gap=FOLLOW_GAP,
                 pass_need=PASS_NEED):
        self.lidar_range = float(lidar_range)
        self.follow_gap = float(follow_gap)
        self.pass_need = float(pass_need)
        self.reset()

    def reset(self):
        self.cap = None            # 걸린 속도 상한 [m/s]. None 이면 안 걸림
        self.front = None          # 직전 정면 거리
        self.closing = 0.0         # 앞의 것이 가까워지는 속도 [m/s]
        self.evidence = 0.0        # '앞차를 못 지나간다' 는 증거
        self.holding = False       # 지금 참는 중인가 (이력)
        self.state = "없음"

    def _rival_speed(self, front, speed, dt):
        """앞의 것이 내는 속도. 분절 없이 정면 거리 변화로 얻는다.

        벽과 콘은 우리 속도만큼 가까워진다. 앞차는 그보다 느리게 가까워진다.
            (우리 속도) - (가까워지는 속도) = 상대가 내는 속도
        """
        if self.front is None:
            self.front, self.closing = front, speed
            return speed - self.closing
        rate = (self.front - front) / max(dt, 1e-3)
        # 코너에서 벽이 갑자기 나타나면 변화가 튄다. 매끄럽게 섞는다.
        rate = float(np.clip(rate, -2.0 * MAX_SPEED, 2.0 * MAX_SPEED))
        self.closing = (1 - BLEND) * self.closing + BLEND * rate
        self.front = front
        return speed - self.closing

    def apply(self, action, scan, angles, speed, dt):
        front = front_range(scan, angles, self.lidar_range)
        rival = self._rival_speed(front, speed, dt)

        clear = front > BLOCK_RANGE
        faster = rival >= SLOW_RATIO * max(speed, 0.5)
        moving = rival > MOVING_MIN
        tight = moving and not clear and             pass_room(scan, angles, front) < self.pass_need

        # 증거를 모은다. 프레임마다 흔들려도 흐름은 남는다.
        self.evidence = float(np.clip(
            self.evidence + (EVIDENCE_UP if tight else -EVIDENCE_DOWN),
            0.0, EVIDENCE_MAX))

        # 한번 참기로 했으면 길이 진짜 비워질 때까지 안 푼다.
        #
        # 이게 4차까지 빠져 있던 것이다. 진단을 보면 T안은 앞차를 보고
        # 스스로 4.94 -> 2.09 m/s 로 줄인다. 그런데 1초 뒤 3.05 로 다시
        # 올리며 옆으로 붙어 벽에 박는다. 상한이 그 순간 풀려 있으면
        # 아무것도 못 막는다. 그래서 끄는 조건을 켜는 조건과 따로 둔다 —
        # 통과 폭이 잠깐 넓게 읽혀도 풀지 않는다.
        if self.holding:
            if clear or faster or not moving:
                self.holding = False
                self.evidence = 0.0
        elif self.evidence >= EVIDENCE_ON:
            self.holding = True

        if self.holding:
            self.state = "참음"
            gap = max(front - self.follow_gap, 0.0)
            want = max(CAP_FLOOR, min(rival, rival + gap))
        else:
            want = None
            self.state = ("앞이 비었다" if clear else
                          "정지물" if not moving else
                          "상대가 빠름" if faster else
                          "확인 중" if self.evidence > 0 else "자리 있음")

        if want is None:
            if self.cap is not None:
                self.cap = self.cap + (MAX_SPEED - self.cap) * CAP_RELAX
                if self.cap >= MAX_SPEED - 0.1:
                    self.cap = None
        else:
            self.cap = want if self.cap is None else                 self.cap + (want - self.cap) * CAP_TIGHTEN

        if self.cap is None:
            return action              # 정책 출력 그대로

        out = np.array(action, dtype=float, copy=True)
        # 행동[1] 은 (a+1)/2*MAX_SPEED 가 목표 속도다. 상한을 그 자리로 옮긴다.
        out[1] = min(float(out[1]), self.cap / MAX_SPEED * 2.0 - 1.0)
        return out
