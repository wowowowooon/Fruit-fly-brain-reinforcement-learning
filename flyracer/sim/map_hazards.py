"""2026 대회 코스 위에 장애물과 상대 차를 얹는다.

지금까지 회피 능력은 우리가 만든 트랙에서만 쟀다. 대회 코스에서는 벽만
있고 장애물도 상대 차도 없었다. 실제로 피하는 모습을 그 코스에서 봐야
한다.

우리 트랙과 대회 코스는 폭이 다르게 생겼다. 우리 트랙은 폭이 일정해서
'트랙 폭' 하나로 장애물 자리를 잡아도 됐다. 대회 코스는 여유가 0.1 m 인
곳부터 0.92 m 인 곳까지 있어서(중앙값 0.50 m), 평균 폭으로 자리를 잡으면
장애물이 벽 안에 박힌다. 그래서 여기서는 그 자리의 실제 여유를 맵에서
직접 읽어 놓는다.

  python race_avoid.py --model T_fly871_final.pt --motion 6
"""

import numpy as np

from flyracer.sim.f1tenth_sim import (BODY_RADIUS, CAR_WIDTH, MAX_SPEED,
                                     OBSTACLE_SIZE, _blob, as_segments)
from flyracer.sim.moving_obstacles import PASS_GAP, MovingObstacleEnv
from flyracer.sim.track_map import MapEnv

# 장애물 옆에 남겨 둘 통로 [m]. 차폭 0.30 m + 여유.
# 우리 트랙 기본값 1.45 m 는 이 코스(폭 1.0~1.8 m)에서는 아무 데도 안 놓인다.
MAP_PASSAGE = 0.80
MAP_OBSTACLES = 6
# 상대 차 옆으로 남겨 둘 통로 [m]. 이 코스는 좁아서 넓은 구간에서만 된다.
MAP_PASS_GAP = 0.75


class MapHazardEnv(MovingObstacleEnv, MapEnv):
    """대회 코스 + 정지 장애물 + 움직이는 상대 차.

    MovingObstacleEnv 가 super() 로 잘 넘겨 주기 때문에 상속 순서만 맞추면
    된다. MapHazardEnv -> MovingObstacleEnv -> MapEnv -> F1TenthEnv 순으로
    _load_track, _near_walls, reset, step 이 겹겹이 불린다.
    """

    def __init__(self, track, hazard_seed=0, n_obstacles=MAP_OBSTACLES,
                 obstacle_passage=MAP_PASSAGE, pass_gap=MAP_PASS_GAP,
                 teacher_rivals=False, spawn_opposite=False, **kwargs):
        self.hazard_seed = int(hazard_seed)
        self.n_obstacles = int(n_obstacles)
        self.obstacle_passage = float(obstacle_passage)
        self.pass_gap = float(pass_gap)
        # 상대 차를 교사 레이싱 라인에 태울지
        self.teacher_rivals = bool(teacher_rivals)
        # 상대 차를 우리 차 정반대(반 바퀴 앞)에 놓을지
        self.spawn_opposite = bool(spawn_opposite)
        self.blobs = []
        self._line_side = None
        super().__init__(track, **kwargs)

    # ---- 맵에서 그 자리의 여유를 직접 읽는다 -----------------------------
    def _room_at(self, points):
        """세계 좌표 점들에서 벽까지의 거리 [m]. 맵 밖이면 0."""
        track = self.track
        rows, cols = track.free.shape
        points = np.atleast_2d(np.asarray(points, dtype=float))
        col = (points[:, 0] - track.origin[0]) / track.resolution
        row = rows - (points[:, 1] - track.origin[1]) / track.resolution
        r = np.rint(row).astype(int)
        c = np.rint(col).astype(int)
        inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        out = np.zeros(len(points))
        out[inside] = track.clearance[r[inside], c[inside]]
        return out

    # ---- 정지 장애물 -----------------------------------------------------
    def _make_map_obstacles(self):
        """중심선 위에, 그 자리 여유가 되는 곳에만 장애물을 놓는다.

        반지름 size 짜리를 한쪽으로 d 만큼 밀어 놓으면 반대쪽에 남는 통로는
        대략 (d - size + room) 이다. 이게 obstacle_passage 이상이면서
        장애물 자체가 벽에 안 닿아야(d + size <= room) 한다. 둘을 같이
        만족하려면 room >= obstacle_passage / 2 + size 여야 한다.
        """
        if self.n_obstacles <= 0:
            return []
        rng = np.random.default_rng([self.hazard_seed, 4242])
        center = self.center
        steps = len(center)
        room = self._room_at(center)
        size_mid = float(np.mean(OBSTACLE_SIZE))
        need = self.obstacle_passage / 2 + size_mid
        # 출발 직후는 비워 둔다. 시작하자마자 장애물이면 볼 게 없다.
        ok = np.where(room >= need)[0]
        ok = ok[(ok > steps * 0.06) & (ok < steps * 0.97)]
        if len(ok) == 0:
            return []

        spacing = max(6, steps // (self.n_obstacles * 3))
        picked, blobs = [], []
        for slot in rng.permutation(ok):
            if len(blobs) >= self.n_obstacles:
                break
            if any(abs(int(slot) - p) < spacing for p in picked):
                continue
            here = room[slot]
            size = float(rng.uniform(*OBSTACLE_SIZE))
            low = self.obstacle_passage - here + size      # 통로를 남기려면
            high = here - size                             # 벽에 안 닿으려면
            if low > high:
                continue
            offset = float(rng.uniform(low, high))
            sign = float(rng.choice([-1.0, 1.0]))
            spot = center[slot] + self._normal[slot] * sign * offset
            blob = _blob(spot, size, rng)
            # 꼭짓점이 하나라도 벽에 걸리면 버린다
            if np.min(self._room_at(blob)) < 0.02:
                continue
            picked.append(int(slot))
            blobs.append(blob)
        return blobs

    def _load_track(self, seed):
        # MovingObstacleEnv -> MapEnv 순으로 돌아 벽과 중심선, 접선이 채워진다
        super()._load_track(seed)
        # 맵의 벽만 따로 들고 있는다. 장애물을 다시 놓을 때 여기서 시작한다.
        self._wall_a, self._wall_b = self.seg_a, self.seg_b
        self._rebuild()

    def rebuild_hazards(self):
        """중심선이 바뀐 뒤 접선·속도표·장애물을 다시 만든다.

        find_start 는 주행 방향을 맞추려고 중심선을 통째로 뒤집는다(이 코스는
        실제로 뒤집힌다). 그러면 _load_track 이 만들어 둔 접선·법선·속도표는
        옛 순서 그대로라, 새 순서로 인덱싱되는 상대 차가 뒤를 보고 서거나
        좌우가 바뀐 자리에 놓인다. 출발 자세를 정한 뒤 한 번 불러 준다.
        """
        tangent = np.roll(self.center, -1, axis=0) - np.roll(self.center, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        self._tangent = tangent
        self._normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        from flyracer.sim.moving_obstacles import _corner_side
        from flyracer.drive.racing_teacher import speed_plan
        self._pace_line = speed_plan(self.center, room=None, v_max=MAX_SPEED)
        self._apex = _corner_side(self.center)
        self._rebuild()
        if self.teacher_rivals:
            # 중심선이 확정된 뒤에 만들어야 라인이 제자리에 앉는다
            self.make_teacher_rival()

    def _rebuild(self):
        """장애물을 새로 놓고 벽 선분에 얹는다."""
        self.blobs = self._make_map_obstacles()
        starts, ends = [self._wall_a], [self._wall_b]
        for blob in self.blobs:
            a, b = as_segments(blob)
            starts.append(a)
            ends.append(b)
        self.seg_a = np.concatenate(starts)
        self.seg_b = np.concatenate(ends)
        self.seg_mid = (self.seg_a + self.seg_b) / 2
        self.seg_reach = (
            np.linalg.norm(self.seg_b - self.seg_a, axis=1) / 2
            + self.lidar_range
        )
        self._rival_cache = None

    # ---- 상대 차 --------------------------------------------------------
    def _place_rivals(self):
        """옆으로 지나갈 틈이 실제로 남는 자리에만 상대 차를 놓는다.

        원래 코드는 '트랙 폭' 하나로 자리를 잡는데, 이 코스는 자리마다
        폭이 두 배 가까이 차이 난다. 좁은 데 놓으면 길을 완전히 막아
        피할 방법이 없는 시험이 된다.
        """
        self.rivals = []
        if self.rival_count <= 0:
            return
        center = self.center
        steps = len(center)
        room = self._room_at(center)
        # 상대 차가 붙어 서 있고 우리가 지나가려면 이만큼은 있어야 한다
        need = (CAR_WIDTH + self.pass_gap) / 2 + BODY_RADIUS
        rng = self.rival_rng or np.random.default_rng([self.hazard_seed, 909])
        ok = np.where(room >= need)[0]
        if not self.spawn_opposite:
            # 무작위로 놓을 때는 출발선 근처를 비운다. 정반대에 놓을 때는
            # '정확히 반 바퀴' 가 목적이라 자르지 않는다.
            ok = ok[(ok > steps * 0.10) & (ok < steps * 0.90)]
        if len(ok) == 0:
            return

        if self.spawn_opposite:
            # 우리 차에서 정확히 반 바퀴 떨어진 자리. 여러 대면 한 바퀴를
            # 고르게 나눈다(우리 자리는 비워 둔다).
            here = int(np.argmin(np.linalg.norm(self.center - self.pos,
                                               axis=1)))
            want = [(here + int(steps * (k + 1) / (self.rival_count + 1)))
                    % steps for k in range(self.rival_count)]
            # 그 자리가 너무 좁으면 통과 가능한 가장 가까운 자리로 민다
            slots = [int(ok[np.argmin(np.abs(ok - w))]) for w in want]
        else:
            slots = [int(c[len(c) // 2])
                     for c in np.array_split(ok, self.rival_count) if len(c)]

        for order, slot in enumerate(slots):
            side_max = float(room[slot]) - CAR_WIDTH / 2
            sign = 1.0 if order % 2 else -1.0
            self.rivals.append({
                "index": float(slot),
                "pace": float(rng.uniform(*self.rival_pace)),
                "hug": float(rng.uniform(0.35, 1.0)),
                "base": float(sign * rng.uniform(0.5, 1.0) * side_max),
                "room": float(side_max),
            })
        self._rival_cache = None

    def _rival_side(self, rival, index):
        """그 자리의 실제 여유 안으로 눌러 넣는다. 안 그러면 벽에 박힌다."""
        if self.teacher_rivals and self._line_side is not None:
            side = float(self._line_side[index])
        else:
            side = super()._rival_side(rival, index)
        here = float(self._room_at(self.center[index])[0])
        limit = max(0.0, here - CAR_WIDTH / 2)
        return float(np.clip(side, -limit, limit))

    # ---- 교사를 상대 차로 -------------------------------------------------
    def make_teacher_rival(self):
        """상대 차를 우리 교사 라인에 태운다.

        기본 상대 차는 중심선을 따라가며 코너에서 안쪽을 조금 파고드는
        단순한 모델이다. 교사는 코리도 DP + 최소곡률로 뽑은 진짜 레이싱
        라인을 그 자리 한계속도로 달린다. 그걸 상대로 쓰면 '제대로 달리는
        차를 어떻게 넘어가나' 를 보게 된다.

        라인을 그대로 쓰지 않고 '중심선에서 옆으로 얼마' 로 바꿔 둔다.
        상대 차 몸통은 중심선 번호로 그려지므로, 그 번호에 맞는 치우침이
        있어야 라인 위에 앉는다.
        """
        from flyracer.drive.racing_teacher import RacingTeacher, speed_plan

        teacher = RacingTeacher(self.center, self.seg_a, self.seg_b,
                                self.track_width / 2, v_max=MAX_SPEED)
        line = teacher.plan.line
        # 라인의 각 점이 중심선 어느 점에서 얼마나 치우쳐 있나
        side = np.zeros(len(self.center))
        for i, point in enumerate(self.center):
            j = int(np.argmin(np.linalg.norm(line - point, axis=1)))
            side[i] = float(np.dot(line[j] - point, self._normal[i]))
        self._line_side = side
        # 교사 속도표도 중심선 번호에 맞춰 옮긴다
        pace = speed_plan(line, teacher.plan.room, MAX_SPEED)
        moved = np.zeros(len(self.center))
        for i, point in enumerate(self.center):
            j = int(np.argmin(np.linalg.norm(line - point, axis=1)))
            moved[i] = pace[j]
        self._pace_line = moved
        return teacher
