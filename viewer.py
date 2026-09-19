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
from fly_driver import N_SECTORS, FlyDriver, sensor_channels

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


def load_font(size, bold=False):
    """한글이 나오는 폰트를 고른다. consolas에는 한글 글자가 없어 네모로 찍힌다."""
    available = pygame.font.get_fonts()
    for name in ("malgungothic", "gulim", "dotum", "notosanskr", "batang"):
        if name in available:
            return pygame.font.SysFont(name, size, bold=bold)
    return pygame.font.SysFont(None, size, bold=bold)


def field_order(channels, n_sectors):
    """시야 구역을 화면 순서로 늘어놓는다.

    왼쪽 끝이 차의 좌측 후방(+135도), 가운데가 정면, 오른쪽 끝이 우측 후방.
    좌우 구역 번호가 서로 반대 방향이라 L은 큰 번호부터, R도 큰 번호부터다.
    """
    index = {name: i for i, (name, _, _) in enumerate(channels)}
    order = [("L", s) for s in range(n_sectors - 1, -1, -1)]
    order += [("R", s) for s in range(n_sectors - 1, -1, -1)]
    return (
        [index[f"prox_{side}{s}"] for side, s in order],
        [index[f"loom_{side}{s}"] for side, s in order],
    )


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
    surface.blit(font.render(label, True, DIM), (x, y - 16))
    mid = x + width // 2
    pygame.draw.rect(surface, (45, 45, 52), (x, y, width, 10))
    pygame.draw.line(surface, (80, 80, 90), (mid, y), (mid, y + 10))
    length = int(np.clip(value, -1, 1) * width / 2)
    color = BAR_POS if length >= 0 else BAR_NEG
    if length >= 0:
        pygame.draw.rect(surface, color, (mid, y, length, 10))
    else:
        pygame.draw.rect(surface, color, (mid + length, y, -length, 10))


def draw_field(surface, font, x, y, label, values, width=220, height=40):
    """구역별 값을 세로 막대로 늘어놓는다. 화면 왼쪽이 차의 왼쪽.

    입력은 이미 평균 0으로 맞춰져 있어서 +-3을 꽉 찬 눈금으로 본다.
    """
    surface.blit(font.render(label, True, DIM), (x, y))
    y += 18
    mid = y + height / 2
    step = width / len(values)
    pygame.draw.line(surface, (70, 70, 80), (x, mid), (x + width, mid))
    for i, value in enumerate(values):
        length = float(np.clip(value / 3.0, -1, 1)) * height / 2
        color = BAR_POS if length >= 0 else BAR_NEG
        top = mid - length if length >= 0 else mid
        pygame.draw.rect(
            surface, color,
            (x + i * step, top, max(2.0, step - 2), abs(length)),
        )
    # 정면이 어디인지 표시
    pygame.draw.line(surface, (90, 90, 100),
                     (x + width / 2, y), (x + width / 2, y + height))
    return y + height + 12


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("초파리 회로 주행")
    font = load_font(15)
    big = load_font(19, bold=True)
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
    prox_idx, loom_idx = field_order(channels, N_SECTORS)
    speed_idx = len(channels) - 1
    mode = "fly"
    paused = False
    total = 0.0
    best = 0.0
    features = np.zeros(len(channels), dtype=np.float32)
    action = np.zeros(2)
    info = {"progress": 0.0}

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
                # act() 안에서 이미 변환했다. 또 부르면 접근율이 0이 된다.
                action = driver.act(obs)
                features = driver.last_features
            else:
                action = follow_the_gap(env, scan)
                features = driver.encoder.encode(scan, obs[-1])
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
        rows = [
            ("진행도", f"{info['progress']:.1%}"),
            ("최고", f"{best:.1%}"),
            ("속도", f"{env.speed:.2f} m/s"),
            ("최근접", f"{scan.min():.2f} m"),
            ("보상", f"{total:.0f}"),
            ("스텝", f"{env.steps}"),
        ]

        y = 24
        screen.blit(big.render(f"[{mode.upper()}] {label}", True, TEXT), (PANEL_X, y))
        y += 40
        # 이름은 왼쪽, 값은 오른쪽 정렬. 폰트 폭과 상관없이 줄이 맞는다.
        for name, value in rows:
            screen.blit(font.render(name, True, DIM), (PANEL_X, y))
            rendered = font.render(value, True, TEXT)
            screen.blit(rendered, (PANEL_X + 220 - rendered.get_width(), y))
            y += 24

        y += 16
        screen.blit(font.render("출력", True, DIM), (PANEL_X, y))
        y += 26
        draw_bar(screen, font, PANEL_X, y, "조향", action[0])
        y += 38
        draw_bar(screen, font, PANEL_X, y, "속도", action[1])

        y += 34
        screen.blit(font.render("시야 (왼쪽 ← 정면 → 오른쪽)", True, DIM), (PANEL_X, y))
        y += 22
        y = draw_field(screen, font, PANEL_X, y, "근접도", features[prox_idx])
        y = draw_field(screen, font, PANEL_X, y, "접근율", features[loom_idx])
        draw_bar(screen, font, PANEL_X, y + 16, "속도 입력", features[speed_idx] / 3.0)

        hint = "F 초파리   G 고전   R 새 트랙   SPACE 정지   ESC 종료"
        screen.blit(font.render(hint, True, DIM), (16, HEIGHT - 26))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
