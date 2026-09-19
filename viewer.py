"""주행을 눈으로 보는 뷰어.

  F  초파리 회로 컨트롤러
  G  Follow the Gap (고전 방식, 비교 기준)
  R  트랙 새로 뽑기
  스페이스  일시정지
  ESC  종료
"""

from pathlib import Path

import numpy as np
import pygame

from f1tenth_sim import LIDAR_RANGE, MAX_STEER, F1TenthEnv
from fly_driver import FlyDriver, sensor_channels

WIDTH, HEIGHT = 1000, 720
TRACK_VIEW = 720
PANEL_X = TRACK_VIEW + 20

BG = (18, 18, 22)
WALL = (110, 115, 130)
BEAM = (230, 140, 60)
CAR = (90, 210, 120)
TEXT = (210, 210, 215)
DIM = (120, 120, 130)
BAR_POS = (90, 170, 240)
BAR_NEG = (220, 90, 90)


def follow_the_gap(env, scan):
    """가장 멀리 뚫린 방향으로 도는 고전 방식."""
    smoothed = np.convolve(scan, np.ones(9) / 9, mode="same")
    best = int(np.argmax(smoothed))
    steer = env.beam_angles[best] / MAX_STEER
    return np.array([np.clip(steer, -1, 1), 1.0 - 0.7 * abs(steer)])


class Camera:
    """월드 좌표를 화면 좌표로."""

    def __init__(self, track):
        span = np.abs(track).max() * 2.2
        self.scale = TRACK_VIEW / span
        self.offset = np.array([TRACK_VIEW / 2, TRACK_VIEW / 2])

    def __call__(self, points):
        pts = np.atleast_2d(points) * self.scale
        pts[:, 1] *= -1
        return pts + self.offset


def draw_loop(surface, camera, loop):
    points = camera(np.vstack([loop, loop[:1]]))
    pygame.draw.lines(surface, WALL, False, [tuple(p) for p in points], 2)


def draw_bar(surface, font, x, y, label, value, width=150):
    """-1 ~ 1 값을 가로 막대로."""
    surface.blit(font.render(label, True, DIM), (x, y - 14))
    mid = x + width // 2
    pygame.draw.rect(surface, (45, 45, 52), (x, y, width, 10))
    pygame.draw.line(surface, (80, 80, 90), (mid, y), (mid, y + 10))
    length = int(np.clip(value, -1, 1) * width / 2)
    color = BAR_POS if length >= 0 else BAR_NEG
    if length >= 0:
        pygame.draw.rect(surface, color, (mid, y, length, 10))
    else:
        pygame.draw.rect(surface, color, (mid + length, y, -length, 10))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("초파리 회로 주행")
    font = pygame.font.SysFont("consolas", 15)
    big = pygame.font.SysFont("consolas", 19, bold=True)
    clock = pygame.time.Clock()

    env = F1TenthEnv(n_beams=108)
    driver = FlyDriver(n_beams=108)
    trained = Path("fly_model.pt").exists()
    if trained:
        driver.load()
        print("학습된 모델을 불러왔습니다.")
    else:
        print("학습 전 모델입니다. train_imitation.py 를 먼저 돌리세요.")
    seed = 0
    obs, _ = env.reset(seed=seed)
    driver.reset()
    camera = Camera(env.outer)

    channels = sensor_channels()
    mode = "fly"
    paused = False
    total = 0.0
    best = 0.0
    features = np.zeros(len(channels), dtype=np.float32)
    action = np.zeros(2)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    mode = "fly"
                elif event.key == pygame.K_g:
                    mode = "gap"
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    seed += 1
                    obs, _ = env.reset(seed=seed)
                    driver.reset()
                    camera = Camera(env.outer)
                    total = 0.0

        if not paused:
            scan = obs[:-1] * LIDAR_RANGE
            if mode == "fly":
                action = driver.act(obs)
                features = driver.encoder.encode(scan, obs[-1])
            else:
                action = follow_the_gap(env, scan)
            obs, reward, done, truncated, info = env.step(action)
            total += reward
            best = max(best, info["progress"])
            if done or truncated:
                obs, _ = env.reset(seed=seed)
                driver.reset()
                total = 0.0

        screen.fill(BG)

        for wall in (env.inner, env.outer):
            draw_loop(screen, camera, wall)

        scan = obs[:-1] * LIDAR_RANGE
        angles = env.beam_angles + env.yaw
        ends = env.pos + scan[:, None] * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )
        origin = tuple(camera(env.pos)[0])
        for end in camera(ends)[::4]:
            pygame.draw.line(screen, BEAM, origin, tuple(end), 1)

        nose = env.pos + 0.9 * np.array([np.cos(env.yaw), np.sin(env.yaw)])
        pygame.draw.circle(screen, CAR, origin, 6)
        pygame.draw.line(screen, CAR, origin, tuple(camera(nose)[0]), 3)

        state = "학습됨" if trained else "학습전"
        label = f"초파리 회로 ({state})" if mode == "fly" else "Follow the Gap"
        lines = [
            (big, f"[{mode.upper()}] {label}"),
            (font, ""),
            (font, f"진행도   {info['progress']:>7.1%}"),
            (font, f"최고     {best:>7.1%}"),
            (font, f"속도     {env.speed:>6.2f} m/s"),
            (font, f"최근접   {scan.min():>6.2f} m"),
            (font, f"보상     {total:>7.0f}"),
            (font, f"스텝     {env.steps:>7d}"),
        ]
        y = 24
        for renderer, text in lines:
            if text:
                screen.blit(renderer.render(text, True, TEXT), (PANEL_X, y))
            y += 24

        y += 10
        screen.blit(font.render("출력", True, DIM), (PANEL_X, y))
        y += 22
        draw_bar(screen, font, PANEL_X, y, "조향", action[0])
        y += 36
        draw_bar(screen, font, PANEL_X, y, "속도", action[1])

        y += 40
        screen.blit(font.render("센서 채널", True, DIM), (PANEL_X, y))
        y += 22
        for (name, _, _), value in zip(channels, features):
            draw_bar(screen, font, PANEL_X, y, name, value, width=150)
            y += 30

        hint = "F 초파리   G 고전   R 새 트랙   SPACE 정지"
        screen.blit(font.render(hint, True, DIM), (16, HEIGHT - 24))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
