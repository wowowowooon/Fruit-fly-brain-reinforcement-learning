"""강화학습용 포장. 초파리 회로는 그대로 두고 필요한 것만 덧붙인다.

강화학습은 세 가지가 필요하다.
  1. 행동을 확률로 내는 것 — 여러 가지를 시도해 봐야 배운다
  2. 지금 상태가 얼마나 좋은지 점수 매기는 것 (가치 함수)
  3. 그 둘을 비교해 정책을 미는 것

이 중 1번의 '평균 행동'만 초파리 회로가 만든다.
2번은 학습을 돕는 보조 장치일 뿐이라 따로 둔다. 실차에 올릴 때는 버린다.
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class RacingPolicy(nn.Module):
    """초파리 회로를 행동 정책으로, 별도 신경망을 가치 함수로 쓴다."""

    def __init__(self, fly, n_channel, init_std=0.35, critic_width=128):
        super().__init__()
        self.fly = fly                       # FlyController. 배선은 초파리 것
        self.n_output = fly.n_output
        # 얼마나 흔들어 볼지. 학습이 진행되면 스스로 줄어든다.
        self.log_std = nn.Parameter(torch.full((2,), float(np.log(init_std))))
        # 가치 함수는 입력 채널 + 운동 뉴런 상태를 본다.
        # 운동 뉴런 상태는 끊어서(detach) 넣으므로 초파리 회로를 건드리지 않는다.
        self.critic = nn.Sequential(
            nn.Linear(n_channel + self.n_output, critic_width), nn.Tanh(),
            nn.Linear(critic_width, critic_width), nn.Tanh(),
            nn.Linear(critic_width, 1),
        )

    def motor_states(self, x, hidden):
        """한 스텝 굴려서 운동 뉴런 상태와 다음 내부 상태를 얻는다."""
        states, hidden = self.fly.rnn(x, hidden)
        return states[..., : self.n_output], hidden

    def act_from(self, motor):
        # 변환은 컨트롤러가 갖고 있다. 여기 따로 적으면 한쪽만 고쳤을 때
        # 조용히 어긋난다 (조향을 좌우 차이로 바꾸면서 실제로 겪었다).
        return self.fly.command(motor)

    def forward(self, x, hidden):
        """x: (B, 1, C) 한 스텝. 평균 행동, 가치, 다음 상태를 돌려준다."""
        motor, hidden = self.motor_states(x, hidden)
        mean = self.act_from(motor)
        value = self.critic(torch.cat([x, motor.detach()], dim=-1))
        return mean, value.squeeze(-1), hidden

    def distribution(self, mean):
        return Normal(mean, self.log_std.exp())

    @torch.no_grad()
    def sample(self, x, hidden):
        """굴려 보면서 쓸 행동. 평균 주변에서 무작위로 흔든다."""
        mean, value, hidden = self(x, hidden)
        dist = self.distribution(mean)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value, hidden

    def evaluate(self, x, hidden, action):
        """모아 둔 행동이 지금 정책에서 얼마나 그럴듯한지 다시 계산한다.

        평균 행동도 같이 돌려준다. 모방학습 정책과 얼마나 멀어졌는지
        재는 데 쓴다.
        """
        mean, value, hidden = self(x, hidden)
        dist = self.distribution(mean)
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value, mean, hidden

    def policy_parameters(self):
        """초파리 회로 + 출력 헤드 + 흔들림 크기. 가치 함수는 뺀다."""
        critic = set(id(p) for p in self.critic.parameters())
        return [p for p in self.parameters() if id(p) not in critic]
