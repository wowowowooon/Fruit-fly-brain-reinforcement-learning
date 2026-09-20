"""추려낸 초파리 회로를 학습 가능한 주행 컨트롤러로 변환.

배선 모양은 초파리에서 그대로 가져오고, 신호 세기만 학습으로 정한다.
연결의 흥분/억제는 보내는 쪽 뉴런의 신경전달물질로 결정한다.

라이다 신호를 아무 시각 뉴런에나 꽂지 않는다. 초파리에서 그 뉴런이
원래 무엇을 읽는지에 맞춰 넣는다.
"""

import re

import numpy as np
import pandas as pd
import torch
from ncps.torch import CfC
from ncps.wirings import Wiring
from torch import nn

# 상대를 끄는 물질. 나머지는 켜는 쪽으로 본다.
INHIBITORY = {"gaba", "glutamate"}

# 시각 뉴런이 실제로 읽는 것. 순서대로 검사하므로 좁은 패턴을 앞에 둔다.
#   loom  LC4, LPLC 계열. 물체가 다가오는 속도(충돌 임박)에 반응
#   prox  나머지 LC 계열. 시야 한 구역에 물체가 있는지 (columnar, 좁은 시야)
#   wide  LPT, HS, H2, VS. 넓은 시야에 걸친 흐름 (tangential, 광역)
SENSORY_KINDS = [
    ("loom", [r"^LC4$", r"^LPLC\d*$"]),
    ("prox", [r"^LC\d+$"]),
    ("wide", [r"^LPT", r"^HS[NES]?$", r"^H2$", r"^Hx$", r"^VS\d*$"]),
]


def sensory_kind(cell_type):
    """이 뉴런에 어떤 신호를 넣어야 하는가."""
    if isinstance(cell_type, str):
        for kind, patterns in SENSORY_KINDS:
            if any(re.match(p, cell_type) for p in patterns):
                return kind
    return "wide"   # 분류 못 한 것은 광역으로 본다 (한 구역에 몰아주지 않음)


def side_of(instance):
    """뉴런이 좌반구인지 우반구인지. 조향은 좌우 차이에서 나온다."""
    if not isinstance(instance, str):
        return None
    if instance.endswith("_L"):
        return "L"
    if instance.endswith("_R"):
        return "R"
    return None


def sector_groups(bodies, n_sectors, position=None):
    """좁은 시야 뉴런들을 시야 구역에 배정한다.

    초파리 겹눈은 낱눈이 육각 격자로 박혀 있고 시신경엽 안쪽까지 그 배열이
    이어진다. 커넥톰이 시냅스마다 그 격자 좌표를 갖고 있어서, 한 세포가
    시야의 어느 방향을 보는지 알 수 있다 (tools/fetch_eye_position.py).

    위치를 모르면 그냥 순서대로 잘라 넣는 수밖에 없는데, 그러면 배선을
    그대로 가져와도 엉뚱한 단자에 꽂는 꼴이 된다. 왼쪽 벽을 보는 신호가
    오른쪽 담당 회로로 들어갈 수 있다.

    position: bodyId -> 0(정면)~1(옆) 값. 없으면 순서대로 자른다.
    """
    bodies = list(bodies)
    if not bodies:
        return {sector: [] for sector in range(n_sectors)}

    if position is None:
        return {sector: list(chunk) for sector, chunk
                in enumerate(np.array_split(bodies, n_sectors))}

    # 실제 시야 각도를 구역 번호로 바꾼다
    where = np.array([position.get(body, 0.5) for body in bodies])
    slot = np.clip((where * n_sectors).astype(int), 0, n_sectors - 1)
    groups = {sector: [b for b, s in zip(bodies, slot) if s == sector]
              for sector in range(n_sectors)}

    # 세포 수가 구역 수보다 적으면 빈 구역이 생긴다. 실제 초파리라면
    # 그 방향을 담당하는 세포의 시야가 그만큼 넓은 것이므로, 가장 가까운
    # 세포가 대신 맡게 한다. 비워 두면 그 각도가 사각지대가 된다.
    centers = (np.arange(n_sectors) + 0.5) / n_sectors
    for sector, members in groups.items():
        if not members:
            nearest = int(np.argmin(np.abs(where - centers[sector])))
            groups[sector] = [bodies[nearest]]
    return groups


def sensory_targets(sensory, n_sectors, position=None):
    """시각 뉴런을 (반구, 읽는 신호, 구역)별로 묶는다.

    좁은 시야 세포(LC 계열)는 자기 구역만 본다.
    광역 세포(LPT/HS 계열)는 자기 반구 전체를 본다.
    """
    narrow, wide = {}, {}
    for side in ("L", "R"):
        rows = sensory[sensory.instance.map(side_of) == side]
        for kind in ("prox", "loom"):
            bodies = rows[rows.type.map(sensory_kind) == kind].bodyId.tolist()
            groups = sector_groups(bodies, n_sectors, position)
            for sector, group in groups.items():
                narrow[(side, kind, sector)] = group
        wide[side] = rows[rows.type.map(sensory_kind) == "wide"].bodyId.tolist()
    return narrow, wide


def build_wiring(neurons, connections, channels, n_sectors, position=None):
    """CSV 두 장을 ncps 배선 객체로 바꾼다.

    channels: [(이름, 종류, 'L'/'R'/'both', 섹터 또는 None), ...]
    position: bodyId -> 0(정면)~1(옆). 시야 방향. 없으면 순서대로 나눈다.
    """
    # 출력 뉴런을 앞쪽에 배치한다. ncps는 앞 index를 출력으로 읽는다.
    # axis로 정렬하면 drive가 먼저, steering이 뒤로 간다.
    outputs = neurons[neurons.axis.notna()].sort_values("axis")
    others = neurons[neurons.axis.isna()]
    ordered = pd.concat([outputs, others]).reset_index(drop=True)
    index = {body: i for i, body in enumerate(ordered.bodyId)}
    axis_counts = outputs.axis.value_counts()

    wiring = Wiring(len(ordered))
    wiring.set_output_dim(len(outputs))

    # 뉴런끼리 연결. 극성은 보내는 쪽 물질로 정한다.
    # 접점 수(weight)는 연결이 얼마나 센지에 대한 유일한 단서다.
    # 나중에 초기 가중치로 쓰려고 같이 들고 있는다.
    nt = dict(zip(ordered.bodyId, ordered.nt))
    strength = np.zeros((len(ordered), len(ordered)), dtype=np.float32)
    has_weight = "weight" in connections.columns
    weights = (connections.weight if has_weight
               else pd.Series(1, index=connections.index))
    for pre, post, count in zip(connections.bodyId_pre,
                                connections.bodyId_post, weights):
        if pre not in index or post not in index:
            continue
        polarity = -1 if nt.get(pre) in INHIBITORY else 1
        wiring.add_synapse(index[pre], index[post], polarity)
        strength[index[pre], index[post]] = float(count)

    # 라이다에서 뽑은 값을 시각 뉴런에 꽂는다.
    # 극성은 +1 이다. 라이다 신호는 그 세포를 '자극'하는 쪽이고,
    # 그 세포가 흥분성인지 억제성인지는 내보낼 때 정해진다 (위 polarity).
    wiring.build(len(channels))
    sensory = ordered[ordered.role == "sensory"]
    narrow, wide = sensory_targets(sensory, n_sectors, position)

    links = 0
    for slot, (_, kind, side, sector) in enumerate(channels):
        if side == "both":
            # 속도는 자기 움직임 정보다. 광역 세포 전체가 받는다.
            targets = wide["L"] + wide["R"]
        else:
            # 그 구역을 보는 좁은 시야 세포 + 그 반구의 광역 세포
            targets = narrow[(side, kind, sector)] + wide[side]
        for body in targets:
            wiring.add_sensory_synapse(slot, index[body], 1)
            links += 1

    # 조향은 좌우 하강 뉴런의 '활동 차이'가 회전 속도가 된다.
    # 어느 것이 왼쪽인지 알아야 그 차이를 읽을 수 있다.
    steer_rows = ordered.iloc[axis_counts.get("drive", 0):len(outputs)]
    steer_sign = np.array(
        [1.0 if side_of(inst) == "L" else -1.0 for inst in steer_rows.instance],
        dtype=np.float32,
    )

    info = {
        "n_drive": int(axis_counts.get("drive", 0)),
        "n_steering": int(axis_counts.get("steering", 0)),
        "kinds": sensory.type.map(sensory_kind).value_counts().to_dict(),
        "links": links,
        "steer_sign": steer_sign,
        "strength": strength,
        "positioned": position is not None,
    }
    return wiring, ordered, info


def apply_fly_wiring(model, wiring, strength=None):
    """초파리 배선을 실제로 신경망에 새긴다.

    ncps 는 배선이 층으로 나뉜 경우만 다룬다. 층 사이는 배선대로 잇고,
    같은 층 안은 ones() 로 전부 이어 버린다. 초파리 뇌는 층 구조가 아니라
    뒤엉킨 그물이라 871개가 전부 한 층에 들어가고, 그 결과 연결 9,992개가
    통째로 무시된 채 758,641개짜리 밀집 신경망이 된다.

    그래서 여기서 직접 덮어쓴다. 이게 빠지면 C(초파리)와 B(밀집)가
    같은 모델이 되어 실험 자체가 성립하지 않는다.

    극성(흥분 +1 / 억제 -1)도 같이 넣는다. 가중치에 곱해지므로 초파리가
    억제성이라 한 연결은 억제 방향에서 학습을 시작한다.
    """
    cell = model.rnn.rnn_cell.layer_0
    mask = cell.sparsity_mask.data
    units = wiring.units
    n_in = mask.shape[1] - units

    adjacency = torch.as_tensor(np.asarray(wiring.adjacency_matrix),
                                dtype=mask.dtype)
    # Linear 가중치는 (받는 쪽, 보내는 쪽) 이라 전치한다.
    mask[:, n_in:] = adjacency.t()

    if strength is None:
        return model

    # 접점 수를 초기 가중치로 쓴다. 커넥톰이 주는 유일한 '세기' 단서다.
    # 접점이 10개인 연결과 1111개인 연결을 같게 두고 시작할 이유가 없다.
    # 범위가 100배라 로그로 눌러서 쓴다.
    scale = torch.as_tensor(strength.T, dtype=mask.dtype)
    scale = torch.log1p(scale) / torch.log1p(scale[scale > 0].median())
    for layer in (cell.ff1, cell.ff2, cell.time_a, cell.time_b):
        block = layer.weight.data[:, n_in:]
        layer.weight.data[:, n_in:] = block * torch.where(
            scale > 0, scale, torch.ones_like(scale))
    return model


def wiring_report(model, wiring):
    """배선이 정말 들어갔는지 센다. 학습 전에 반드시 확인한다."""
    mask = model.rnn.rnn_cell.layer_0.sparsity_mask.data
    units = wiring.units
    n_in = mask.shape[1] - units
    recurrent = mask[:, n_in:]
    adjacency = np.asarray(wiring.adjacency_matrix)
    return {
        "links": int((recurrent != 0).sum()),
        "expected": int((adjacency != 0).sum()),
        "inhibitory": int((recurrent < 0).sum()),
        "sensory": int((mask[:, :n_in] != 0).sum()),
        "dense_units": units * units,
    }


class FlyController(nn.Module):
    """초파리 배선으로 신호를 흘리고, 운동 뉴런을 조향/구동으로 읽는다.

    조향 뉴런(DNa01 등)은 조향만, 구동 뉴런(DNg100, MDN 등)은 속도만 만든다.
    회로가 나눠 놓은 축을 마지막 단계에서 뭉개지 않는다.

    조향은 좌우 하강 뉴런의 '활동 차이'로 읽는다. 초파리에서 이 차이가
    회전 속도에 비례한다고 측정됐다. 10개를 그냥 하나로 합치면 그 원리를
    버리는 것이고, 좌우 대칭이던 회로가 방향을 잃는다.
    """

    def __init__(self, wiring, n_channel, n_drive, n_steering,
                 steer_sign=None):
        super().__init__()
        self.rnn = CfC(n_channel, wiring)
        self.n_drive = n_drive
        self.n_output = n_drive + n_steering
        # 배선에서 출력 뉴런은 drive가 앞, steering이 뒤에 놓여 있다.
        self.speed_head = nn.Linear(n_drive, 1)
        if steer_sign is None:            # 좌우를 모르면 예전처럼 합친다
            steer_sign = np.ones(n_steering, dtype=np.float32)
        self.register_buffer("steer_sign",
                             torch.as_tensor(steer_sign, dtype=torch.float32))
        self.steer_gain = nn.Parameter(torch.ones(1))
        self.steer_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, hidden=None):
        states, hidden = self.rnn(x, hidden)
        motor = states[..., : self.n_output]
        drive, steering = motor[..., : self.n_drive], motor[..., self.n_drive:]
        # 왼쪽이 세면 한쪽으로, 오른쪽이 세면 반대로
        turn = (steering * self.steer_sign).mean(-1, keepdim=True)
        command = torch.tanh(torch.cat(
            [self.steer_gain * turn + self.steer_bias, self.speed_head(drive)],
            dim=-1,
        ))
        return command, hidden
