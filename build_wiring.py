"""추려낸 초파리 회로를 학습 가능한 주행 컨트롤러로 변환.

배선 모양은 초파리에서 그대로 가져오고, 신호 세기만 학습으로 정한다.
연결의 흥분/억제는 보내는 쪽 뉴런의 신경전달물질로 결정한다.
"""

import numpy as np
import pandas as pd
import torch
from ncps.torch import CfC
from ncps.wirings import Wiring
from torch import nn

# 상대를 끄는 물질. 나머지는 켜는 쪽으로 본다.
INHIBITORY = {"gaba", "glutamate"}


def side_of(instance):
    """뉴런이 좌반구인지 우반구인지. 조향은 좌우 차이에서 나온다."""
    if not isinstance(instance, str):
        return None
    if instance.endswith("_L"):
        return "L"
    if instance.endswith("_R"):
        return "R"
    return None


def sector_groups(neurons, n_sectors):
    """시각 뉴런을 좌우로 나누고, 각 쪽을 시야 구역만큼 쪼갠다.

    초파리 시각계는 보는 방향에 따라 뉴런이 늘어서 있다.
    라이다도 방향별 거리이므로 같은 식으로 대응시킨다.
    """
    groups = {}
    for side in ("L", "R"):
        bodies = [
            body
            for body, instance in zip(neurons.bodyId, neurons.instance)
            if side_of(instance) == side
        ]
        for sector, chunk in enumerate(np.array_split(bodies, n_sectors)):
            groups[(side, sector)] = list(chunk)
    return groups


def build_wiring(neurons, connections, channels, n_sectors):
    """CSV 두 장을 ncps 배선 객체로 바꾼다.

    channels: [(이름, 'L'/'R'/'both', 섹터 또는 None), ...]
    """
    # 출력 뉴런을 앞쪽에 배치한다. ncps는 앞 index를 출력으로 읽는다.
    outputs = neurons[neurons.axis.notna()].sort_values("axis")
    others = neurons[neurons.axis.isna()]
    ordered = pd.concat([outputs, others]).reset_index(drop=True)
    index = {body: i for i, body in enumerate(ordered.bodyId)}

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
    groups = sector_groups(sensory, n_sectors)

    links = 0
    for slot, (_, side, sector) in enumerate(channels):
        if side == "both":
            targets = sensory.bodyId.tolist()
        else:
            targets = groups[(side, sector)]
        for body in targets:
            wiring.add_sensory_synapse(slot, index[body], 1)
            links += 1

    return wiring, ordered, links


class FlyController(nn.Module):
    """초파리 배선으로 신호를 흘리고, 운동 뉴런을 조향/구동으로 읽는다."""

    def __init__(self, wiring, n_channel, n_output):
        super().__init__()
        self.rnn = CfC(n_channel, wiring)
        self.n_output = n_output
        # 운동 뉴런 여러 개를 조향 1축, 구동 1축으로 모은다.
        self.head = nn.Linear(n_output, 2)

    def forward(self, x, hidden=None):
        states, hidden = self.rnn(x, hidden)
        command = torch.tanh(self.head(states[..., : self.n_output]))
        return command, hidden
