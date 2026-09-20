"""커넥톰 인접행렬을 그대로 학습 가능한 조향 컨트롤러로 꽂을 수 있는지 검증."""

import sys
from pathlib import Path

# 상위 폴더에 있는 공통 모듈을 찾을 수 있게 한다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import nn
from ncps.torch import CfC
from ncps.wirings import Wiring

rng = np.random.default_rng(0)

N_SENSOR = 4      # 차선offset, 헤딩, 앞차거리, 속도
N_NEURON = 20     # 커넥톰에서 떼온 뉴런 수
N_MOTOR = 2       # 조향, 가감속

# 실제로는 neuPrint 연결표가 들어갈 자리. 지금은 같은 형식의 가짜 배선.
adjacency = (rng.random((N_NEURON, N_NEURON)) < 0.25).astype(int)
np.fill_diagonal(adjacency, 0)
sensory_adjacency = (rng.random((N_SENSOR, N_NEURON)) < 0.3).astype(int)

# polarity: 흥분성 +1 / 억제성 -1. 실제로는 신경전달물질 예측값을 씀.
polarity = rng.choice([1, -1], size=(N_NEURON, N_NEURON))

wiring = Wiring(N_NEURON)
wiring.set_output_dim(N_MOTOR)

for src, dest in zip(*np.nonzero(adjacency)):
    wiring.add_synapse(int(src), int(dest), int(polarity[src, dest]))

wiring.build(N_SENSOR)
for src, dest in zip(*np.nonzero(sensory_adjacency)):
    wiring.add_sensory_synapse(int(src), int(dest), 1)


class ConnectomeController(nn.Module):
    """배선은 커넥톰으로 고정하고, 앞쪽 N_MOTOR개 뉴런만 명령으로 읽는다."""

    def __init__(self, wiring, n_sensor, n_motor):
        super().__init__()
        self.rnn = CfC(n_sensor, wiring)
        self.n_motor = n_motor

    def forward(self, x):
        states, _ = self.rnn(x)
        return states[..., : self.n_motor]


model = ConnectomeController(wiring, N_SENSOR, N_MOTOR)

print(f"neurons        : {N_NEURON}")
print(f"internal synapses: {int(np.abs(wiring.adjacency_matrix).sum())}")
print(f"sensory synapses : {int(np.abs(wiring.sensory_adjacency_matrix).sum())}")
print(f"trainable params : {sum(p.numel() for p in model.parameters()):,}")

sensors = torch.randn(1, 10, N_SENSOR)
output = model(sensors)
print(f"\ninput {tuple(sensors.shape)} -> output {tuple(output.shape)}")
print(f"last step [steer, throttle]: {output[0, -1].detach().numpy().round(3)}")

target = torch.randn(1, 10, N_MOTOR)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
losses = []
for _ in range(100):
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(sensors), target)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f"\nloss {losses[0]:.4f} -> {losses[-1]:.4f}")
print("OK: wiring fixed, weights trainable" if losses[-1] < losses[0] * 0.5 else "NG")

