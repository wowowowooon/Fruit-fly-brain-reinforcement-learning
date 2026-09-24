"""주행을 눈으로 보는 뷰어.

  F  초파리 회로 (학습된 모델)
  G  Follow the Gap (고전 방식, 비교 기준)
  T  교사 (레이싱 라인 + Stanley). 라인도 같이 그린다
  R  트랙 새로 뽑기
  +/-  배속        스페이스  일시정지        ESC  종료

  python viewer.py
  python viewer.py --model fly_model.pt
  python viewer.py --map "C:/.../ifac_roboracer.yaml"
"""

import sys
from pathlib import Path

# 이 파일은 하위 폴더에 있다. flyracer 패키지를 찾으려면
# 저장소 뿌리를 경로에 넣어 줘야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import numpy as np
import pygame

from flyracer.drive.controllers import follow_the_gap
from flyracer.sim.f1tenth_sim import BODY_BACK, BODY_FRONT, CAR_WIDTH, F1TenthEnv
from flyracer.paths import model as model_path
from flyracer.brain.fly_driver import N_MOTION, N_SECTORS, FlyDriver, sensor_channels
from flyracer.drive.raceline import StanleyRacer

WIDTH, HEIGHT = 1000, 720
TRACK_VIEW = 720
PANEL_X = TRACK_VIEW + 20

BG = (18, 18, 22)
WALL = (110, 115, 130)
BEAM = (230, 140, 60)
BLOB = (200, 60, 60)          # 정지 장애물
RIVAL = (240, 150, 40)        # 움직이는 상대 차
CAR = (90, 210, 120)
TEXT = (210, 210, 215)
DIM = (120, 120, 130)
BAR_POS = (90, 170, 240)
BAR_NEG = (220, 90, 90)


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
    양쪽 다 구역 0이 정면이므로 L은 큰 번호부터, R은 0번부터 늘어놓는다.
    """
    index = {channel[0]: i for i, channel in enumerate(channels)}
    order = [("L", s) for s in range(n_sectors - 1, -1, -1)]
    order += [("R", s) for s in range(n_sectors)]
    return (
        [index[f"prox_{side}{s}"] for side, s in order],
        [index[f"loom_{side}{s}"] for side, s in order],
    )


class Camera:
    """월드 좌표를 화면 좌표로. 트랙 전체가 화면에 들어오게 맞춘다."""

    def __init__(self, points):
        points = np.atleast_2d(points)
        low, high = points.min(axis=0), points.max(axis=0)
        span = float(np.max(high - low)) * 1.12
        self.scale = TRACK_VIEW / max(span, 1e-6)
        middle = (low + high) / 2
        self.offset = (np.array([TRACK_VIEW / 2, TRACK_VIEW / 2])
                       - np.array([middle[0], -middle[1]]) * self.scale)

    def __call__(self, points):
        pts = np.atleast_2d(points).astype(float) * self.scale
        pts[:, 1] *= -1
        return pts + self.offset


def draw_walls(camera, seg_a, seg_b):
    """벽은 한 번만 그려 배경으로 들고 있는다. 매 프레임 다시 그리면 느리다."""
    surface = pygame.Surface((TRACK_VIEW, HEIGHT))
    surface.fill(BG)
    for a, b in zip(camera(seg_a), camera(seg_b)):
        pygame.draw.line(surface, WALL, tuple(a), tuple(b), 2)
    return surface


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="C_fly871.pt")
    parser.add_argument("--map", default=None, help="ROS 점유격자 .yaml")
    parser.add_argument("--motion", type=int, default=3,
                        help="관측의 자기 움직임 채널 수. R안 3, T안 6")
    parser.add_argument("--beams", type=int, default=541)
    parser.add_argument("--obstacles", type=int, default=0,
                        help="대회 코스에 놓을 정지 장애물 수")
    parser.add_argument("--rivals", type=int, default=0,
                        help="움직이는 상대 차 수")
    parser.add_argument("--hazard-seed", type=int, default=1)
    parser.add_argument("--laps", type=int, default=0,
                        help="몇 바퀴 돌고 멈출지. 0이면 끌 때까지 계속")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("초파리 회로 주행")
    font = load_font(15)
    big = load_font(19, bold=True)
    clock = pygame.time.Clock()

    if args.map:
        from flyracer.sim.map_start import find_start
        from flyracer.sim.track_map import MapEnv, MapTrack
        track = MapTrack(args.map)
        if args.obstacles or args.rivals:
            from flyracer.sim.map_hazards import MapHazardEnv
            env = MapHazardEnv(track, hazard_seed=args.hazard_seed,
                               n_obstacles=args.obstacles,
                               rivals=args.rivals,
                               n_beams=args.beams, motion=args.motion)
            env.start_pose = find_start(track, env)
            # find_start 가 중심선을 뒤집는다. 접선과 장애물을 다시 만들어야
            # 상대 차가 제 방향으로 제자리에 선다.
            env.rebuild_hazards()
        else:
            env = MapEnv(track, n_beams=args.beams, motion=args.motion)
            env.start_pose = find_start(track, env)
        print(f"실제 코스: {track.name}")
    else:
        env = F1TenthEnv(n_beams=args.beams, motion=args.motion)

    driver = FlyDriver(n_beams=env.n_beams, lidar_range=env.lidar_range,
                       dt=env.dt, motion=args.motion)
    # 켜 둔 동안 계속 돌게 한다. 한 바퀴에서 끊기면 회피하는 순간을
    # 몇 번씩 다시 보기 어렵다. 박았을 때만 처음으로 돌린다.
    endless = args.laps <= 0
    env.laps = 10 ** 6 if endless else args.laps
    env.max_steps = 10 ** 9 if endless else env.max_steps

    trained = model_path(args.model).exists()
    if trained:
        driver.load(args.model)
        print(f"모델: {args.model}")
    else:
        print(f"{args.model} 이 없습니다. 학습 전 모델로 돕니다.")

    seed = 0
    crashes = 0
    obs, _ = env.reset(seed=seed)
    driver.reset()
    camera = Camera(np.concatenate([env.seg_a, env.seg_b]))
    wall_a = getattr(env, "_wall_a", env.seg_a)
    wall_b = getattr(env, "_wall_b", env.seg_b)
    walls = draw_walls(camera, wall_a, wall_b)
    teacher = None
    speed_mult = 1

    channels = sensor_channels()
    prox_idx, loom_idx = field_order(channels, N_SECTORS)
    speed_idx = len(channels) - env.motion
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
                elif event.key == pygame.K_t and not args.map:
                    mode = "teacher"
                    obs, _ = env.reset(seed=seed)
                    teacher = StanleyRacer.for_env(env)
                    obs = env._observe(env._scan())
                    driver.reset()
                    total = 0.0
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed_mult = min(8, speed_mult * 2)
                elif event.key == pygame.K_MINUS:
                    speed_mult = max(1, speed_mult // 2)
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    if not args.map:
                        seed += 1
                    obs, _ = env.reset(seed=seed)
                    driver.reset()
                    teacher = None
                    if mode == "teacher":
                        teacher = StanleyRacer.for_env(env)
                        obs = env._observe(env._scan())
                    camera = Camera(np.concatenate([env.seg_a, env.seg_b]))
                    wall_a = getattr(env, "_wall_a", env.seg_a)
                    wall_b = getattr(env, "_wall_b", env.seg_b)
                    walls = draw_walls(camera, wall_a, wall_b)
                    total = 0.0

        for _ in range(0 if paused else speed_mult):
            scan = obs[:-env.motion] * env.lidar_range
            if mode == "fly":
                # act() 안에서 이미 변환했다. 또 부르면 접근율이 0이 된다.
                action = driver.act(obs)
                features = driver.last_features
            else:
                if mode == "teacher" and teacher is not None:
                    action = teacher.act(env.pos, env.yaw, env.speed)
                else:
                    action = follow_the_gap(scan, env.beam_angles,
                                            env.lidar_range)
                features = driver.encoder.encode(scan, obs[-env.motion:], env.dt)
            obs, reward, done, truncated, info = env.step(action)
            total += reward
            best = max(best, info["progress"])
            if done or truncated:
                if env.crashed:
                    crashes += 1
                obs, _ = env.reset(seed=seed)
                driver.reset()
                if mode == "teacher":
                    teacher = StanleyRacer.for_env(env)
                    obs = env._observe(env._scan())
                total = 0.0
                break

        screen.fill(BG)
        screen.blit(walls, (0, 0))
        for blob in getattr(env, "blobs", []):
            pygame.draw.polygon(screen, BLOB,
                                [tuple(q) for q in camera(blob)])
        for rival in getattr(env, "rivals", []):
            box = camera(env._rival_box(rival))
            pygame.draw.polygon(screen, RIVAL, [tuple(q) for q in box])
        if mode == "teacher" and teacher is not None:
            loop = camera(np.vstack([teacher.line, teacher.line[:1]]))
            pygame.draw.lines(screen, (120, 200, 255), False,
                              [tuple(p) for p in loop], 2)

        scan = obs[:-env.motion] * env.lidar_range
        angles = env.beam_angles + env.yaw
        # 광선은 차 가운데가 아니라 앞쪽 라이다에서 나간다
        lidar = env.lidar_pos
        ends = lidar + scan[:, None] * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )
        origin = tuple(camera(lidar)[0])
        for end in camera(ends)[::16]:
            pygame.draw.line(screen, BEAM, origin, tuple(end), 1)

        # 차체를 길이 있는 물체로 그린다. 뒤가 길다.
        heading = np.array([np.cos(env.yaw), np.sin(env.yaw)])
        body = camera(np.stack([env.pos + BODY_BACK * heading,
                                env.pos + BODY_FRONT * heading]))
        thickness = max(3, int(CAR_WIDTH * camera.scale))
        pygame.draw.line(screen, CAR, tuple(body[0]), tuple(body[1]), thickness)
        pygame.draw.circle(screen, (240, 120, 90), origin, 4)

        state = Path(args.model).stem if trained else "학습전"
        label = {"fly": f"초파리 회로 ({state})",
                 "gap": "고전 Follow the Gap",
                 "teacher": "교사 (레이싱 라인)"}[mode]
        rows = [
            ("바퀴", f"{info['progress']:.2f}"),
            ("박은 횟수", f"{crashes}"),
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

        hint = (f"F 초파리  G 고전  T 교사  R 새트랙  +/- 배속({speed_mult}x)"
                "  SPACE 정지  ESC 종료")
        screen.blit(font.render(hint, True, DIM), (16, HEIGHT - 26))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
