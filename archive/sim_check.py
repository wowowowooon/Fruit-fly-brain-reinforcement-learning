"""F1TENTH 환경이 제대로 도는지 확인. 벽 따라가는 기본 주행도 같이 본다."""

import sys
from pathlib import Path

# 상위 폴더에 있는 공통 모듈을 찾을 수 있게 한다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import OUTPUTS

import numpy as np
from PIL import Image

from f1tenth_sim import F1TenthEnv

env = F1TenthEnv(n_beams=108)
obs, info = env.reset()

print(f"관측 {obs.shape}  (라이다 {env.n_beams}점 + 속도)")
print(f"행동 {env.action_space.shape}  [조향, 속도]")
print(f"시작 라이다: 최소 {obs[:-1].min() * 30:.2f}m  최대 {obs[:-1].max() * 30:.2f}m\n")


def follow_the_gap(scan):
    """가장 멀리 뚫린 방향으로 핸들을 돌리는 고전 방식. 비교 기준."""
    angles = env.beam_angles
    smoothed = np.convolve(scan, np.ones(9) / 9, mode="same")
    best = int(np.argmax(smoothed))
    steer = angles[best] / 0.42
    speed = 1.0 - 0.7 * abs(steer)
    return np.array([np.clip(steer, -1, 1), speed])


frames = []
total = 0.0
for step in range(600):
    scan = obs[:-1] * 30.0
    obs, reward, done, truncated, info = env.step(follow_the_gap(scan))
    total += reward
    if step % 120 == 0:
        frames.append(Image.fromarray(env.render()))
    if done or truncated:
        break

print(f"{env.steps}스텝 주행")
print(f"진행도 {info['progress']:.1%}   속도 {info['speed']:.1f} m/s")
print(f"충돌 {info['crashed']}   누적 보상 {total:.0f}")

frames.append(Image.fromarray(env.render()))
width, height = frames[0].size
strip = Image.new("RGB", (width * len(frames), height), "white")
for i, frame in enumerate(frames):
    strip.paste(frame, (i * width, 0))
strip.save(OUTPUTS / "f1tenth_preview.png")
print(f"\n저장: outputs/f1tenth_preview.png ({len(frames)}프레임)")

