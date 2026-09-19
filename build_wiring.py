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


def sector_groups(bodies, n_sectors):
    """좁은 시야 뉴런들을 시야 구역 수만큼 나눠 배정한다.

    주의: 어느 뉴런이 시야의 어느 방향을 보는지는 커넥톰에서 아직 안 받았다.
    그래서 구역 배정 자체는 고정된 임의 분할이다. 생물학적 근거가 있는 것은
    좌우 반구 구분(instance의 _L/_R)까지다.
    """
    return {
        sector: list(chunk)
        for sector, chunk in enumerate(np.array_split(bodies, n_sectors))
    }


def sensory_targets(sensory, n_sectors):
    """시각 뉴런을 (반구, 읽는 신호, 구역)별로 묶는다.

    좁은 시야 세포(LC 계열)는 자기 구역만 본다.
    광역 세포(LPT/HS 계열)는 자기 반구 전체를 본다.
    """
    narrow, wide = {}, {}
    for side in ("L", "R"):
        rows = sensory[sensory.instance.map(side_of) == side]
        for kind in ("prox", "loom"):
            bodies = rows[rows.type.map(sensory_kind) == kind].bodyId.tolist()
            for sector, group in sector_groups(bodies, n_sectors).items():
                narrow[(side, kind, sector)] = group
        wide[side] = rows[rows.type.map(sensory_kind) == "wide"].bodyId.tolist()
    return narrow, wide


def build_wiring(neurons, connections, channels, n_sectors):
    """CSV 두 장을 ncps 배선 객체로 바꾼다.

    channels: [(이름, 종류, 'L'/'R'/'both', 섹터 또는 None), ...]
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
    nt = dict(zip(ordered.bodyId, ordered.nt))
    for pre, post in zip(connections.bodyId_pre, connections.bodyId_post):
        if pre not in index or post not in index:
            continue
        polarity = -1 if nt.get(pre) in INHIBITORY else 1
        wiring.add_synapse(index[pre], index[post], polarity)

    # 라이다에서 뽑은 값을 시각 뉴런에 꽂는다.
    wiring.build(len(channels))
    sensory = ordered[ordered.role == "sensory"]
    narrow, wide = sensory_targets(sensory, n_sectors)

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

    info = {
        "n_drive": int(axis_counts.get("drive", 0)),
        "n_steering": int(axis_counts.get("steering", 0)),
        "kinds": sensory.type.map(sensory_kind).value_counts().to_dict(),
        "links": links,
    }
    return wiring, ordered, info


class FlyController(nn.Module):
    """초파리 배선으로 신호를 흘리고, 운동 뉴런을 조향/구동으로 읽는다.

    조향 뉴런(DNa01 등)은 조향만, 구동 뉴런(DNg100, MDN 등)은 속도만 만든다.
    회로가 나눠 놓은 축을 마지막 단계에서 뭉개지 않는다.
    """

    def __init__(self, wiring, n_channel, n_drive, n_steering):
        super().__init__()
        self.rnn = CfC(n_channel, wiring)
        self.n_drive = n_drive
        self.n_output = n_drive + n_steering
        # 배선에서 출력 뉴런은 drive가 앞, steering이 뒤에 놓여 있다.
        self.speed_head = nn.Linear(n_drive, 1)
        self.steer_head = nn.Linear(n_steering, 1)

    def forward(self, x, hidden=None):
        states, hidden = self.rnn(x, hidden)
        motor = states[..., : self.n_output]
        drive, steering = motor[..., : self.n_drive], motor[..., self.n_drive:]
        command = torch.tanh(torch.cat(
            [self.steer_head(steering), self.speed_head(drive)], dim=-1
        ))
        return command, hidden
