"""2026 대회 코스에서 장애물과 상대 차를 피하는 모습을 그린다.

세 장면을 같은 코스에서 나란히 그린다.

  벽만          기준선. 아무것도 없을 때의 라인과 속도
  정지 장애물    콘이 놓였을 때 어디로 돌아 나가는지
  상대 차        앞차가 있을 때 어떻게 붙고 어떻게 빠지는지

  python race_avoid.py --model T_fly871_final.pt --motion 6
  python race_avoid.py --model R_fly871_v1_keep.pt --motion 3 --out outputs/R_avoid.png
"""

import sys
from pathlib import Path

# 이 파일은 하위 폴더에 있다. flyracer 패키지를 찾으려면
# 저장소 뿌리를 경로에 넣어 줘야 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from flyracer.sim.f1tenth_sim import BODY_RADIUS, LIDAR_RANGE
from flyracer.brain.fly_driver import FlyDriver
from flyracer.sim.map_hazards import MAP_OBSTACLES, MapHazardEnv
from flyracer.paths import OUTPUTS, model as model_path
from flyracer.sim.map_start import find_start
from flyracer.sim.track_map import MapTrack
from flyracer.paths import RACE_MAP

MAP_YAML = str(RACE_MAP)
SEED = 1234


def korean_font():
    """한글이 네모로 나오지 않게 시스템 글꼴을 찾는다."""
    for name in ("Malgun Gothic", "NanumGothic", "Gulim", "Batang"):
        try:
            matplotlib.font_manager.findfont(name, fallback_to_default=False)
        except Exception:
            continue
        plt.rcParams["font.family"] = name
        plt.rcParams["axes.unicode_minus"] = False
        return


def build(track, driver_motion, beams, obstacles, rivals, hazard_seed):
    env = MapHazardEnv(track, hazard_seed=hazard_seed,
                       n_obstacles=obstacles, rivals=rivals,
                       n_beams=beams, lidar_range=LIDAR_RANGE,
                       motion=driver_motion)
    env.start_pose = find_start(track, env)
    # find_start 는 주행 방향을 맞추려고 중심선을 뒤집는다(이 코스는 실제로
    # 뒤집힌다). 접선과 속도표가 옛 순서 그대로면 상대 차가 뒤를 보고 선다.
    env.rebuild_hazards()
    # reset 이 트랙을 다시 읽어 중심선 방향을 되돌리지 못하게 막는다
    env.track_seed = SEED
    return env


def drive(driver, env, max_steps=6000):
    obs, _ = env.reset(seed=SEED)
    driver.reset()
    path, speeds, gaps = [], [], []
    rival_paths = [[] for _ in env.rivals]
    info = {"progress": 0.0, "lap_time": 0.0, "crashed": True}
    for _ in range(max_steps):
        path.append(env.pos.copy())
        speeds.append(env.speed)
        for keep, rival in zip(rival_paths, env.rivals):
            keep.append(env._rival_box(rival).mean(axis=0))
        obs, _, done, truncated, info = env.step(driver.act(obs))
        gaps.append(info["clearance"] - BODY_RADIUS)
        if done or truncated:
            break
    return {
        "path": np.array(path), "speed": np.array(speeds),
        "gap": float(np.min(gaps)) if gaps else 0.0,
        "rival_paths": [np.array(p) for p in rival_paths],
        "blobs": [b.copy() for b in env.blobs],
        "rival_boxes": [env._rival_box(r) for r in env.rivals],
        "info": info,
    }


def draw(ax, track, scene, title):
    rows, cols = track.free.shape
    res = track.resolution
    ax.imshow(track.free, cmap="gray",
              extent=[track.origin[0], track.origin[0] + cols * res,
                      track.origin[1], track.origin[1] + rows * res],
              origin="upper")

    for blob in scene["blobs"]:
        ax.add_patch(Polygon(blob, closed=True, facecolor="#d62728",
                             edgecolor="#7f1013", linewidth=1.0, zorder=4))
    for keep in scene["rival_paths"]:
        if len(keep):
            ax.plot(keep[:, 0], keep[:, 1], color="#ff7f0e",
                    linewidth=1.6, alpha=0.55, zorder=3)
    for box in scene["rival_boxes"]:
        ax.add_patch(Polygon(box, closed=True, facecolor="#ff7f0e",
                             edgecolor="#8c4400", linewidth=1.0,
                             alpha=0.9, zorder=5))

    path, speed = scene["path"], scene["speed"]
    dots = ax.scatter(path[:, 0], path[:, 1], c=speed, cmap="turbo",
                      s=7, vmin=0, vmax=7, zorder=6)
    ax.plot(*path[0], marker="o", color="lime", markersize=9, zorder=7)
    info = scene["info"]
    if info["crashed"]:
        ax.plot(*path[-1], marker="X", color="red", markersize=15, zorder=8)

    ax.set_title(f"{title}\n{info['lap_time']:.2f}초   진행 {info['progress']:.0%}   "
                 f"최소여유 {scene['gap']:+.2f} m   "
                 f"{'충돌' if info['crashed'] else '완주'}", fontsize=12)
    ax.set_xlim(path[:, 0].min() - 2.5, path[:, 0].max() + 2.5)
    ax.set_ylim(path[:, 1].min() - 2.5, path[:, 1].max() + 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return dots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="T_fly871_final.pt")
    parser.add_argument("--motion", type=int, default=6)
    parser.add_argument("--beams", type=int, default=541)
    parser.add_argument("--map", default=MAP_YAML)
    parser.add_argument("--obstacles", type=int, default=MAP_OBSTACLES)
    parser.add_argument("--rivals", type=int, default=2)
    parser.add_argument("--hazard-seed", type=int, default=1)
    parser.add_argument("--out", default=str(OUTPUTS / "race_avoid.png"))
    args = parser.parse_args()

    korean_font()
    track = MapTrack(args.map)
    driver = FlyDriver(n_beams=args.beams, lidar_range=LIDAR_RANGE,
                       fly_wiring=True, motion=args.motion)
    driver.load(args.model)

    scenes = []
    for title, obstacles, rivals in (
            ("벽만 (기준선)", 0, 0),
            (f"정지 장애물 {args.obstacles}개", args.obstacles, 0),
            (f"상대 차 {args.rivals}대", 0, args.rivals)):
        env = build(track, args.motion, args.beams, obstacles, rivals,
                    args.hazard_seed)
        scene = drive(driver, env)
        scenes.append((title, scene))
        info = scene["info"]
        print(f"  {title:<16} {info['lap_time']:>6.2f}초  "
              f"진행 {info['progress']:>6.1%}  "
              f"최소여유 {scene['gap']:+.3f} m  "
              f"장애물 {len(scene['blobs'])}  상대 {len(scene['rival_boxes'])}  "
              f"{'충돌' if info['crashed'] else '완주'}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=100)
    for ax, (title, scene) in zip(axes, scenes):
        dots = draw(ax, track, scene, title)
    fig.colorbar(dots, ax=axes, label="속도 [m/s]", shrink=0.6)
    fig.suptitle(f"{Path(args.model).stem} — 2026 대회 코스에서 피하기  "
                 f"(빨강 = 정지 장애물, 주황 = 상대 차와 그 궤적)", fontsize=15)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
