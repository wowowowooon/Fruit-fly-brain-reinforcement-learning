"""비교용 일반 신경망 (A안).

초파리 배선이 뭔가 값어치를 한다고 말하려면, 흔히 쓰는 방식이 같은
과제에서 어디까지 가는지를 먼저 알아야 한다. 그 기준선이다.

감각 처리(라이다 -> 39채널)는 C안과 똑같이 쓴다. 그래야 바뀌는 것이
'신경망 구조' 하나로 남는다. 다른 것은 전부 같다.

  A  GRU 256칸. 층 구조 + 전결합. 업계 표준
  B  뉴런 871개, 전부 연결 (초파리 뉴런에 일반 배선)
  C  뉴런 871개, 초파리 배선

A 는 B/C 와 뉴런 수가 다르므로 인과 비교가 아니다. "우리 것이 실무
수준과 견줄 만한가"를 보는 참고선이다. 배선의 효과를 가리는 것은
뉴런 수가 같은 B vs C 다.
"""

import torch
from torch import nn


class StandardController(nn.Module):
    """FlyController 와 같은 자리에 꽂히는 일반 순환신경망.

    강화학습 쪽(RacingPolicy)이 .rnn / .command / .n_output / .n_drive 를
    쓰므로 그것만 맞춰 두면 학습 코드를 그대로 쓸 수 있다.
    """

    name = "일반 신경망 (GRU)"

    def __init__(self, n_channel, n_drive, n_steering, width=256):
        super().__init__()
        self.n_drive = n_drive
        self.n_output = n_drive + n_steering
        if width < self.n_output:
            raise ValueError("은닉 칸이 운동 뉴런 수보다 적을 수 없다")
        self.width = width
        self.core = nn.GRU(n_channel, width)
        # 초파리 쪽과 맞추기 위해 앞쪽 칸들을 '운동 뉴런'으로 읽는다.
        self.speed_head = nn.Linear(n_drive, 1)
        # 일반 신경망에는 좌우 대칭 구조가 없다. 그냥 전부 이어서 읽는다.
        self.steer_head = nn.Linear(n_steering, 1)

    def rnn(self, x, hidden=None):
        """(B, T, C) 를 받아 (B, T, width) 와 다음 상태를 돌려준다."""
        states, hidden = self.core(x.transpose(0, 1),
                                   None if hidden is None else hidden)
        return states.transpose(0, 1), hidden

    def command(self, motor):
        drive = motor[..., : self.n_drive]
        steering = motor[..., self.n_drive:]
        return torch.tanh(torch.cat(
            [self.steer_head(steering), self.speed_head(drive)], dim=-1))

    def forward(self, x, hidden=None):
        states, hidden = self.rnn(x, hidden)
        return self.command(states[..., : self.n_output]), hidden

    def effective_params(self):
        """전부 연결이라 저장된 파라미터가 곧 쓰이는 파라미터다."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    from fly_driver import sensor_channels

    net = StandardController(len(sensor_channels()), 12, 10)
    total = sum(p.numel() for p in net.parameters())
    print(f"{net.name}  은닉 {net.width}칸")
    print(f"입력 {len(sensor_channels())}채널 -> 운동 {net.n_output}칸")
    print(f"파라미터 {total:,}개")
    out, _ = net(torch.randn(2, 5, len(sensor_channels())))
    print(f"출력 {tuple(out.shape)}  (조향, 속도)")
