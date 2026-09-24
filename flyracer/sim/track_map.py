"""ROS 점유격자 지도(.png + .yaml)를 시뮬 트랙으로 읽어들인다.

대회에서 배포하는 맵이 이 형식이다. 학습에는 쓰지 않고, 학습이 끝난 모델이
처음 보는 실제 코스에서 달릴 수 있는지 확인하는 데 쓴다.

  free_map = MapTrack("ifac_roboracer.yaml")
  env = MapEnv(free_map)
"""

from pathlib import Path

import networkx as nx
import numpy as np
import yaml
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from flyracer.sim.f1tenth_sim import CAR_WIDTH, F1TenthEnv, as_segments


def _thin(mask):
    """면을 1칸 두께의 뼈대로 깎는다 (Zhang-Suen 세선화).

    통로의 뼈대가 곧 한가운데 선이다. 양쪽 벽에서 똑같이 깎아 들어가므로
    가운데만 남는다. 능선을 '이웃보다 벽에서 먼 칸'으로 잡으면 부동소수점
    비교 때문에 점이 거의 안 남는다. 이 방법은 모양만 보므로 그런 문제가 없다.
    """
    image = mask.astype(np.uint8).copy()
    image[0, :] = image[-1, :] = image[:, 0] = image[:, -1] = 0

    def neighbours(img):
        # 시계 방향 8이웃 P2..P9 (P2 가 위)
        return [np.roll(np.roll(img, dy, axis=0), dx, axis=1)
                for dy, dx in ((1, 0), (1, -1), (0, -1), (-1, -1),
                               (-1, 0), (-1, 1), (0, 1), (1, 1))]

    while True:
        removed = 0
        for step in (0, 1):
            p = neighbours(image)
            count = sum(p)                      # 살아 있는 이웃 수
            # 0->1 로 바뀌는 횟수. 1이어야 끊기지 않는다
            ring = p + [p[0]]
            transitions = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8)
                              for i in range(8))
            if step == 0:
                cond = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                cond = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            drop = ((image == 1) & (count >= 2) & (count <= 6)
                    & (transitions == 1) & cond)
            removed += int(drop.sum())
            image[drop] = 0
        if removed == 0:
            return image.astype(bool)


# 이보다 짧은 곁가지·작은 고리는 벽 울퉁불퉁함이 만든 잡음으로 본다 [m]
MIN_BRANCH_M = 1.2


def _skeleton_graph(rows, cols, points):
    """뼈대 칸들을 이어 그래프로. 계단 모양의 군더더기 연결은 뺀다.

    세선화된 선도 격자 위에서는 계단처럼 꺾인다. 상하좌우와 대각선을
    전부 이으면 그 계단마다 삼각형이 생겨 갈림길로 잡힌다(2,076칸짜리
    뼈대에서 갈림길이 384개로 세어졌다).

    대각선은 그 두 칸이 상하좌우로 이미 이어져 있으면 건너뛴다.
    그러면 계단이 하나의 선으로 펴진다.
    """
    index = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))

    for (r, c), i in index.items():
        for dr, dc in ((0, 1), (1, 0)):                 # 상하좌우
            j = index.get((r + dr, c + dc))
            if j is not None:
                graph.add_edge(i, j, weight=float(
                    np.linalg.norm(points[i] - points[j])))
    for (r, c), i in index.items():
        for dr, dc in ((1, 1), (1, -1)):                # 대각선
            j = index.get((r + dr, c + dc))
            if j is None:
                continue
            # 두 칸 사이에 상하좌우로 이어주는 칸이 이미 있으면 군더더기다
            detour = any(
                index.get(side) is not None
                and graph.has_edge(i, index[side])
                and graph.has_edge(index[side], j)
                for side in ((r + dr, c), (r, c + dc))
            )
            if not detour:
                graph.add_edge(i, j, weight=float(
                    np.linalg.norm(points[i] - points[j])))
    return graph


def _drop_short_loops(ring, points, min_span):
    """짧은 곁가지와 작은 고리를 지워 갈림길 수를 줄인다."""
    graph = nx.Graph(ring)
    for _ in range(40):
        junctions = [n for n in graph if graph.degree(n) != 2]
        if not junctions:
            break
        doomed, changed = set(), False
        for start in junctions:
            for first in list(graph.neighbors(start)):
                chain, prev, node = [start, first], start, first
                while graph.degree(node) == 2:
                    nxt = next(n for n in graph.neighbors(node) if n != prev)
                    prev, node = node, nxt
                    chain.append(node)
                    if len(chain) > 4000:
                        break
                span = float(np.linalg.norm(
                    np.diff(points[chain], axis=0), axis=1).sum())
                if span < min_span and len(chain) > 2:
                    doomed.update(chain[1:-1])
                    changed = True
        if not changed:
            break
        graph.remove_nodes_from(doomed)
        # 지우고 나면 새로 생긴 막다른 가지도 친다
        while True:
            leaves = [n for n in graph if graph.degree(n) <= 1]
            if not leaves:
                break
            graph.remove_nodes_from(leaves)
        if not graph:
            raise RuntimeError("가지를 치다 보니 남는 게 없다")
    biggest = max(nx.connected_components(graph), key=len)
    return graph.subgraph(biggest).copy()


def _branch_graph(ring, points):
    """뼈대를 '갈림길과 그 사이 길' 구조로 압축한다.

    뼈대는 점이 수천 개지만 갈림길은 몇 개뿐이다. 갈림길만 꼭짓점으로
    삼고 그 사이 길을 하나의 변으로 묶으면, 남는 그래프가 아주 작아져서
    가장 긴 고리를 제대로 찾을 수 있다.
    """
    junctions = [n for n in ring if ring.degree(n) != 2]
    small = nx.MultiGraph()
    if not junctions:                       # 갈림길이 없으면 통째로 한 고리
        return None, None
    small.add_nodes_from(junctions)

    walked = set()
    paths = {}
    for start in junctions:
        for first in ring.neighbors(start):
            if (start, first) in walked:
                continue
            chain, prev, node = [start, first], start, first
            while ring.degree(node) == 2:
                nxt = next(n for n in ring.neighbors(node) if n != prev)
                prev, node = node, nxt
                chain.append(node)
            walked.add((start, first))
            walked.add((node, chain[-2]))
            span = float(np.linalg.norm(
                np.diff(points[chain], axis=0), axis=1).sum())
            key = small.add_edge(start, node, weight=span)
            paths[(start, node, key)] = chain
            paths[(node, start, key)] = chain[::-1]
    return small, paths


def _longest_cycle(graph, points):
    """뼈대에서 코스 본선 한 바퀴를 뽑는다.

    뼈대에는 막다른 가지와 갈림길이 섞여 있다. 가지를 쳐낸 뒤에도
    안쪽 섬을 도는 짧은 고리가 남아서, 가까운 이웃만 따라가면 본선을
    놓친다(54m 코스에서 37m 만 돌고 나머지를 직선으로 잘라먹었다).

    갈림길 구조로 압축한 뒤 가장 긴 고리를 고른다.
    """
    graph = graph.copy()
    while True:
        leaves = [n for n in graph if graph.degree(n) <= 1]
        if not leaves:
            break
        graph.remove_nodes_from(leaves)
    if not graph:
        raise RuntimeError("중심선에 고리가 없다. 코스가 닫혀 있는지 확인")

    biggest = max(nx.connected_components(graph), key=len)
    ring = graph.subgraph(biggest)

    # 세선화는 벽의 울퉁불퉁함마다 짧은 곁가지와 작은 고리를 남긴다.
    # 그대로 두면 갈림길이 수백 개가 되어 고리 찾기가 터진다.
    # 코스 본선의 한 구간보다 한참 짧은 것은 전부 잡음으로 보고 지운다.
    ring = _drop_short_loops(ring, points, min_span=MIN_BRANCH_M)

    small, paths = _branch_graph(ring, points)
    if small is None:                       # 갈림길 없음 = 이미 한 고리
        order, prev = [next(iter(ring))], None
        while True:
            options = [n for n in ring.neighbors(order[-1]) if n != prev]
            if not options or options[0] == order[0]:
                break
            prev = order[-1]
            order.append(options[0])
        return order

    # 작은 그래프에서 가장 긴 고리를 고른다. 꼭짓점이 몇 개뿐이라 전부 본다.
    best, best_span = None, -1.0
    for cycle in nx.simple_cycles(nx.Graph(small)):
        if len(cycle) < 2:
            continue
        span, chain, ok = 0.0, [], True
        for i, node in enumerate(cycle):
            nxt = cycle[(i + 1) % len(cycle)]
            options = [(k, d["weight"]) for k, d in small[node][nxt].items()] \
                if small.has_edge(node, nxt) else []
            if not options:
                ok = False
                break
            key, weight = max(options, key=lambda kv: kv[1])
            span += weight
            chain.extend(paths[(node, nxt, key)][:-1])
        if ok and span > best_span:
            best, best_span = chain, span
    if best is None:
        raise RuntimeError("코스 본선을 못 찾았다")
    return best


def _wall_segments(free, resolution, origin):
    """주행 가능 영역의 경계를 선분 목록으로 바꾼다.

    격자에서 '뚫린 칸'과 '막힌 칸'이 맞닿는 면이 곧 벽이다.
    그 면들을 그대로 짧은 선분으로 만든다.
    """
    rows, cols = free.shape
    ox, oy = origin[0], origin[1]

    def world(cx, cy):
        # 이미지 y축은 아래로, 월드 y축은 위로 간다
        return np.stack([ox + cx * resolution,
                         oy + (rows - cy) * resolution], axis=1)

    starts, ends = [], []
    # 세로 경계 (왼쪽/오른쪽 이웃과 다른 곳)
    edge = free[:, :-1] != free[:, 1:]
    y, x = np.nonzero(edge)
    starts.append(world(x + 1.0, y))
    ends.append(world(x + 1.0, y + 1.0))
    # 가로 경계 (위/아래 이웃과 다른 곳)
    edge = free[:-1, :] != free[1:, :]
    y, x = np.nonzero(edge)
    starts.append(world(x, y + 1.0))
    ends.append(world(x + 1.0, y + 1.0))

    return np.concatenate(starts), np.concatenate(ends)


class MapTrack:
    """맵 파일에서 벽과 주행 가능 영역을 읽어 둔다."""

    def __init__(self, yaml_path):
        # 윈도우에서 역슬래시 경로가 들어와도 되게 pathlib 으로 다룬다
        yaml_path = Path(yaml_path)
        with yaml_path.open(encoding="utf-8") as handle:
            meta = yaml.safe_load(handle)

        image = Image.open(yaml_path.parent / meta["image"]).convert("L")
        grid = np.asarray(image, dtype=np.float32) / 255.0
        if meta.get("negate", 0):
            grid = 1.0 - grid

        self.name = Path(meta["image"]).stem
        self.resolution = float(meta["resolution"])
        self.origin = meta["origin"]
        # free_thresh 보다 비어 있으면 주행 가능. 회색(미탐색)은 벽으로 본다.
        self.free = grid > (1.0 - float(meta.get("free_thresh", 0.196)))
        self.seg_a, self.seg_b = _wall_segments(
            self.free, self.resolution, self.origin
        )

        # 벽에서 가장 멀리 떨어진 곳이 코스 한가운데다
        self.clearance = ndimage.distance_transform_edt(self.free) * self.resolution

    def to_world(self, row, col):
        rows = self.free.shape[0]
        return np.array([self.origin[0] + col * self.resolution,
                         self.origin[1] + (rows - row) * self.resolution])

    def widest_point(self):
        row, col = np.unravel_index(np.argmax(self.clearance), self.clearance.shape)
        return self.to_world(row, col), float(self.clearance[row, col])

    def centerline(self, spacing=0.10):
        """주행 가능 영역 한가운데를 따라가는 폐곡선.

        전에는 고전 컨트롤러로 한 바퀴 돌려서 그 궤적을 중심선으로 썼다.
        컨트롤러가 중간에 박으면 잘린 경로가 그대로 중심선이 되고, 진행도가
        엉터리가 된다. 실제로 14.4m 만 기록된 중심선 때문에 16.6m 를 달리고
        충돌한 주행이 '진행도 136%, 완주'로 잡혔다.

        컨트롤러에 기대지 않고 맵 모양에서 바로 뽑는다. 벽에서 가장 먼
        점들이 통로 한가운데이므로, 그 능선을 이어 붙이면 중심선이 된다.
        """
        # 차가 못 지나갈 만큼 좁은 곳은 통로로 치지 않는다.
        # 문틈 같은 데로 뼈대가 새는 것을 막는다.
        usable = self.free & (self.clearance > CAR_WIDTH / 2)
        usable = ndimage.binary_opening(usable, np.ones((3, 3)))
        ridge = _thin(usable)

        rows, cols = np.nonzero(ridge)
        if len(rows) < 8:
            raise RuntimeError("중심선을 뽑을 만한 통로를 못 찾았다")
        points = np.stack([
            self.origin[0] + cols * self.resolution,
            self.origin[1] + (self.free.shape[0] - rows) * self.resolution,
        ], axis=1)

        graph = _skeleton_graph(rows, cols, points)
        loop = _longest_cycle(graph, points)
        path = points[loop]

        # 등간격으로 다시 찍는다
        closed = np.vstack([path, path[:1]])
        step = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(step)])
        want = np.arange(0.0, arc[-1], spacing)
        smooth = np.stack(
            [np.interp(want, arc, closed[:, i]) for i in (0, 1)], axis=1)
        # 격자 계단을 살짝 편다
        for _ in range(6):
            smooth = 0.5 * smooth + 0.25 * (np.roll(smooth, 1, axis=0)
                                            + np.roll(smooth, -1, axis=0))
        return smooth

    def summary(self):
        rows, cols = self.free.shape
        drivable = self.free.sum() * self.resolution ** 2
        width = np.percentile(self.clearance[self.free], 90) * 2
        return (
            f"{self.name}  {cols * self.resolution:.1f} x "
            f"{rows * self.resolution:.1f} m\n"
            f"주행 가능 {drivable:.0f} m^2, 통로 폭 약 {width:.1f} m "
            f"(가장 넓은 곳 {self.clearance.max() * 2:.1f} m)\n"
            f"벽 선분 {len(self.seg_a):,}개"
        )


class MapEnv(F1TenthEnv):
    """실제 맵 위에서 달리는 환경. 나머지 물리와 센서는 그대로다."""

    def __init__(self, track, start_pose=None, **kwargs):
        self.track = track
        self.start_pose = start_pose
        super().__init__(track_seed=0, **kwargs)

    def _load_track(self, seed):
        self.seg_a, self.seg_b = self.track.seg_a, self.track.seg_b
        self.seg_mid = (self.seg_a + self.seg_b) / 2
        self.seg_reach = (
            np.linalg.norm(self.seg_b - self.seg_a, axis=1) / 2 + self.lidar_range
        )
        self.inner = self.outer = np.zeros((0, 2))
        self.track_width = float(
            np.percentile(self.track.clearance[self.track.free], 90) * 2
        )
        # 중심선은 맵 모양에서 바로 뽑는다. 예전에는 고전 컨트롤러로 한 바퀴
        # 돌려서 그 궤적을 썼는데, 컨트롤러가 박으면 잘린 경로가 중심선이 되어
        # 진행도가 엉터리가 됐다 (16.6m 달리고 충돌한 주행이 '완주'로 잡혔다).
        # step_length 와 lap_length 를 같이 채우므로 맨 뒤에 둔다.
        self.set_centerline(self.track.centerline())

    def set_centerline(self, path):
        """한 바퀴 분량의 경로를 중심선으로 삼는다. 진행도 계산에 쓴다."""
        self.center = np.asarray(path, dtype=np.float64)
        steps = np.linalg.norm(
            np.roll(self.center, -1, axis=0) - self.center, axis=1
        )
        self.step_length = float(steps.mean())
        self.lap_length = float(steps.sum())

    def reset(self, seed=None, options=None):
        if self.start_pose is None:
            raise RuntimeError("출발 자세를 먼저 정해야 한다 (find_start 사용)")
        position, yaw = self.start_pose
        obs, info = super().reset(seed=seed, options=options)
        self.pos = np.asarray(position, dtype=np.float64).copy()
        self.yaw = float(yaw)
        self.speed = 1.5
        self.prev_index = self._progress()
        self.lap_progress = 0
        return self._observe(self._scan()), info
