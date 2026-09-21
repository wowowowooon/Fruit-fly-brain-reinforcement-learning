"""움직이는 장애물 — 앞서 달리는 상대 차량.

지금 시뮬의 장애물은 트랙을 만들 때 바닥에 박아 넣는 정지 덩어리라서
'나에게 다가오는 것'이 하나도 없다. 초파리 회로에서 가장 유명한 부분이
접근 감지(LC4, LPLC2, LPLC1)인데 시험해 볼 자리 자체가 없었다.

실제 대회에서 만나는 움직이는 물체는 앞서 달리는 다른 차다. 그래서
중심선을 따라 우리보다 느리게 도는 차를 놓는다. 추월하거나, 뒤에
붙어서 속도를 잃거나, 받는다.

B·C 비교에서는 학습에 쓰지 않았다. 배운 적 없는 상황에서 어느 배선이
덜 무너지는지를 보려고 평가에만 썼다. 레이싱용 재학습부터는
vec_env 의 rivals 옵션으로 학습에도 넣는다.

  env = MovingObstacleEnv(n_beams=541, track_seed=700)
"""

import numpy as np

from f1tenth_sim import (BODY_BACK, BODY_FRONT, CAR_LENGTH, CAR_WIDTH,
                         F1TenthEnv, as_segments, body_clearance)

RIVAL_COUNT = 2              # 트랙 하나에 놓을 상대 차 수
RIVAL_SPEED = (1.5, 3.0)     # 상대 차 속도 [m/s]. 우리는 5~6 m/s 로 달린다
PASS_GAP = 0.90              # 옆으로 남겨 둘 통로 [m]. 차폭은 0.30 m
SPAWN_RANGE = (0.15, 0.85)   # 출발선 기준 어디쯤에 놓을지 (한 바퀴 비율)


class MovingObstacleEnv(F1TenthEnv):
    """정지 장애물은 그대로 두고, 움직이는 상대 차를 얹는다."""

    def __init__(self, *args, rivals=RIVAL_COUNT, rival_speed=RIVAL_SPEED,
                 **kwargs):
        self.rival_count = rivals
        self.rival_speed = rival_speed
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
                "speed": float(rng.uniform(*self.rival_speed)),
                "side": float(sign * rng.uniform(0.5, 1.0) * room),
            })
        self._rival_cache = None

    # ---- 상대 차 몸통 ----------------------------------------------------
    def _rival_box(self, rival):
        """상대 차를 트랙 위 직사각형으로 그린다. 방향은 진행 방향."""
        steps = len(self.center)
        i = int(rival["index"]) % steps
        frac = rival["index"] % 1.0
        j = (i + 1) % steps
        middle = self.center[i] * (1 - frac) + self.center[j] * frac
        t, n = self._tangent[i], self._normal[i]
        middle = middle + n * rival["side"]
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
        for rival in self.rivals:
            rival["index"] += rival["speed"] * self.dt / self.step_length
        self._rival_cache = None
        return super().step(action)

    def rival_states(self):
        """상대 차의 지금 위치와 속도. 교사는 이걸 보고 길을 고른다.

        학생(신경망)에게는 주지 않는다. 학생은 라이다로만 본다.
        """
        return [{"pos": self._rival_box(r).mean(axis=0),
                 "speed": r["speed"]} for r in self.rivals]

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
