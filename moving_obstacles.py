"""움직이는 장애물 — 앞서 달리는 상대 차량.

지금 시뮬의 장애물은 트랙을 만들 때 바닥에 박아 넣는 정지 덩어리라서
'나에게 다가오는 것'이 하나도 없다. 초파리 회로에서 가장 유명한 부분이
접근 감지(LC4, LPLC2, LPLC1)인데 시험해 볼 자리 자체가 없었다.

실제 대회에서 만나는 움직이는 물체는 앞서 달리는 다른 차다. 그래서
우리보다 조금 느리게 도는 차를 놓는다. 추월하거나, 뒤에 붙어서 속도를
잃거나, 받는다.

상대 차도 '운전' 을 한다. 코너에서는 감속하고 안쪽을 파고들며, 직선에서는
속도를 올린다. 처음에는 한 바퀴 내내 같은 속도로 같은 자리를 가게 만들었는데,
그러면 코너가 4.8 m/s 를 허용하는 자리를 1.7 m/s 로 기어가는 굴러다니는
장애물이 된다. 추월해야 할 '차' 가 아니라서 배울 것이 없다.

B·C 비교에서는 학습에 쓰지 않았다. 배운 적 없는 상황에서 어느 배선이
덜 무너지는지를 보려고 평가에만 썼다. 레이싱용 재학습부터는
vec_env 의 rivals 옵션으로 학습에도 넣는다.

  env = MovingObstacleEnv(n_beams=541, track_seed=700)
"""

import numpy as np

from f1tenth_sim import (BODY_BACK, BODY_FRONT, CAR_LENGTH, CAR_WIDTH,
                         MAX_SPEED, F1TenthEnv, as_segments, body_clearance)
from raceline import CURVE_SMOOTH, path_curvature
from racing_teacher import speed_plan

RIVAL_COUNT = 2              # 트랙 하나에 놓을 상대 차 수
# 상대 차가 낼 속도 / 그 자리에서 물리적으로 낼 수 있는 속도.
# 우리보다 느려야 따라잡히지만, 너무 느리면 추월이 아니라 정지 장애물이 된다.
RIVAL_PACE = (0.55, 0.85)
RIVAL_HUG = (0.35, 1.0)      # 코너 안쪽을 얼마나 파고드는지 (1이면 끝까지)
PASS_GAP = 0.90              # 옆으로 남겨 둘 통로 [m]. 차폭은 0.30 m
SPAWN_RANGE = (0.15, 0.85)   # 출발선 기준 어디쯤에 놓을지 (한 바퀴 비율)
APEX_SMOOTH = 21             # 코너 안쪽 방향을 이만큼 평균 내어 부드럽게


def _corner_side(center, smooth=APEX_SMOOTH):
    """각 지점에서 코너 안쪽이 어느 쪽인지. -1(오른쪽) ~ +1(왼쪽).

    안쪽 = sign(곡률) x 법선. 반시계 원으로 숫자를 찍어 확인했다.
    직선에서는 0 에 가까워져서 상대 차가 제 자리를 지킨다.
    """
    kappa = path_curvature(center, CURVE_SMOOTH)
    scale = np.percentile(np.abs(kappa), 75)
    side = np.clip(kappa / max(scale, 1e-6), -1.0, 1.0)
    window = np.ones(smooth) / smooth
    padded = np.concatenate([side[-smooth:], side, side[:smooth]])
    return np.convolve(padded, window, mode="same")[smooth:-smooth]


class MovingObstacleEnv(F1TenthEnv):
    """정지 장애물은 그대로 두고, 움직이는 상대 차를 얹는다."""

    def __init__(self, *args, rivals=RIVAL_COUNT, rival_pace=RIVAL_PACE,
                 **kwargs):
        self.rival_count = rivals
        self.rival_pace = rival_pace
        self.rivals = []
        self.rival_rng = None
        self._rival_cache = None
        super().__init__(*args, **kwargs)

    # ---- 트랙 기하 -------------------------------------------------------
    def _load_track(self, seed):
        super()._load_track(seed)
        tangent = np.roll(self.center, -1, axis=0) - np.roll(self.center, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        self._tangent = tangent
        self._normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        # 상대 차도 코너에서는 감속하고 직선에서는 올린다. 폭 제한은 걸지
        # 않는다 — 상대는 자기 라인을 아는 차라고 본다.
        self._pace_line = speed_plan(self.center, room=None, v_max=MAX_SPEED)
        self._apex = _corner_side(self.center)

    def _place_rivals(self):
        """상대 차를 출발선 앞쪽에 흩어 놓는다. 옆으로 지나갈 틈은 남긴다."""
        self.rivals = []
        # 옆을 지나가려면 차 반폭 + 통로가 필요하다. 못 되면 이 트랙은 포기.
        room = self.track_width / 2 - CAR_WIDTH / 2 - PASS_GAP / 2
        if room <= 0.05 or self.rival_count <= 0:
            return
        steps = len(self.center)
        # 학습에 쓸 때는 바깥에서 rng 를 꽂아 같은 트랙도 매번 다르게 만든다.
        # 평가할 때는 트랙 시드로 고정해 두 모델이 똑같은 상황을 만난다.
        rng = self.rival_rng or np.random.default_rng([self.track_seed, 909])
        low, high = SPAWN_RANGE
        spots = np.linspace(low, high, self.rival_count + 2)[1:-1]
        for order, spot in enumerate(spots):
            sign = 1.0 if order % 2 else -1.0
            self.rivals.append({
                "index": float(spot * steps),
                "pace": float(rng.uniform(*self.rival_pace)),
                "hug": float(rng.uniform(*RIVAL_HUG)),
                "base": float(sign * rng.uniform(0.4, 1.0) * room),
                "room": float(room),
            })
        self._rival_cache = None

    def _rival_side(self, rival, index):
        """그 지점에서 상대 차가 서 있는 좌우 위치.

        코너에서는 안쪽을 파고들고, 직선에서는 제가 잡은 쪽을 지킨다.
        실제 앞차가 코너 안쪽을 막고 있는 상황이 추월의 핵심이다.
        """
        apex = float(self._apex[index])
        side = rival["base"] * (1.0 - abs(apex)) + rival["hug"] * rival["room"] * apex
        return float(np.clip(side, -rival["room"], rival["room"]))

    def _rival_speed(self, rival, index):
        """그 지점에서 상대 차가 내는 속도."""
        return rival["pace"] * float(self._pace_line[index])

    # ---- 상대 차 몸통 ----------------------------------------------------
    def _rival_box(self, rival):
        """상대 차를 트랙 위 직사각형으로 그린다. 방향은 진행 방향."""
        steps = len(self.center)
        i = int(rival["index"]) % steps
        frac = rival["index"] % 1.0
        j = (i + 1) % steps
        middle = self.center[i] * (1 - frac) + self.center[j] * frac
        t, n = self._tangent[i], self._normal[i]
        middle = middle + n * self._rival_side(rival, i)
        half_l, half_w = CAR_LENGTH / 2, CAR_WIDTH / 2
        return np.stack([
            middle + t * half_l + n * half_w,
            middle + t * half_l - n * half_w,
            middle - t * half_l - n * half_w,
            middle - t * half_l + n * half_w,
        ])

    def _rival_segments(self):
        if self._rival_cache is not None:
            return self._rival_cache
        starts, ends = [], []
        for rival in self.rivals:
            a, b = as_segments(self._rival_box(rival))
            starts.append(a)
            ends.append(b)
        if starts:
            self._rival_cache = (np.concatenate(starts), np.concatenate(ends))
        else:
            empty = np.zeros((0, 2))
            self._rival_cache = (empty, empty)
        return self._rival_cache

    def _near_walls(self, origin, reach):
        seg_a, seg_b = super()._near_walls(origin, reach)
        if not self.rivals:
            return seg_a, seg_b
        ra, rb = self._rival_segments()
        return np.concatenate([seg_a, ra]), np.concatenate([seg_b, rb])

    # ---- 시간 진행 -------------------------------------------------------
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._place_rivals()
        # 상대 차를 놓고 나서 다시 재야 라이다에 상대 차가 보인다
        return self._observe(self._scan()), info

    def step(self, action):
        steps = len(self.center)
        for rival in self.rivals:
            here = int(rival["index"]) % steps
            rival["index"] += (self._rival_speed(rival, here) * self.dt
                               / self.step_length)
        self._rival_cache = None
        return super().step(action)

    def rival_states(self, horizon=2.5, samples=6):
        """상대 차의 지금 위치·속도와, 앞으로 갈 자리들.

        상대가 앞뒤로만 움직이면 지금 위치만 알면 된다. 그런데 코너에서
        안쪽을 파고들기 때문에 좌우로도 움직인다. 내가 따라잡았을 때
        그 차가 어디 있을지는 지금 위치로 알 수 없다. 그래서 앞으로 몇
        초간의 자리를 같이 준다.

        교사만 본다. 학생(신경망)은 라이다로만 본다.
        """
        steps = len(self.center)
        states = []
        for rival in self.rivals:
            here = int(rival["index"]) % steps
            future = []
            index = rival["index"]
            step = horizon / samples
            for k in range(1, samples + 1):
                index += self._rival_speed(
                    rival, int(index) % steps) * step / self.step_length
                ghost = dict(rival, index=index)
                future.append((k * step, self._rival_box(ghost).mean(axis=0)))
            states.append({
                "pos": self._rival_box(rival).mean(axis=0),
                "speed": self._rival_speed(rival, here),
                "future": future,
            })
        return states

    def rival_gap(self):
        """차체에서 가장 가까운 상대 차까지의 여유 [m]. 못 놓았으면 무한대.

        충돌 판정과 똑같은 방식으로 잰다(차체 앞뒤를 훑어 선분까지의 거리).
        중심에서 모서리까지를 재면 옆구리를 받았을 때 놓친다.
        """
        if not self.rivals:
            return float("inf")
        heading = self._heading()
        back = self.pos + BODY_BACK * heading
        front = self.pos + BODY_FRONT * heading
        ra, rb = self._rival_segments()
        return body_clearance(back, front, ra, rb)
