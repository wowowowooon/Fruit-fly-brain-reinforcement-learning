"""실제 코스 위에 주행 궤적을 그린다. 어디서 왜 실패하는지 눈으로 본다.

  .\\.venv\\Scripts\\python.exe render_map_run.py <맵.yaml> [출력.png]
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from controllers import follow_the_gap
from fly_driver import N_MOTION
from f1tenth_sim import BODY_RADIUS
from test_on_map import find_start, trace_lap
from paths import OUTPUTS
from track_map import MapEnv, MapTrack


def drive(env, controller, max_steps=6000):
    obs, _ = env.reset()
    if hasattr(controller, "reset"):
        controller.reset()
    path, speeds = [], []
    for _ in range(max_steps):
        path.append(env.lidar_pos.copy())
        speeds.append(env.speed)
        action = controller(obs) if callable(controller) else controller.act(obs)
        obs, _, done, truncated, info = env.step(action)
        if done or truncated:
            break
    return np.array(path), np.array(speeds), info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map_yaml")
    parser.add_argument("--out", default=str(OUTPUTS / "map_run.png"))
    parser.add_argument("--model", default="fly_racer.pt")
    args = parser.parse_args()

    track = MapTrack(args.map_yaml)
    env = MapEnv(track)
    env.start_pose = find_start(track, env)
    path, _ = trace_lap(env)
    env.set_centerline(path)

    runs = []
    gap = lambda obs: follow_the_gap(          # noqa: E731
        obs[:-N_MOTION] * env.lidar_range, env.beam_angles, env.lidar_range)
    runs.append(("고전 Follow the Gap", *drive(env, gap)))

    from pathlib import Path
    if Path(args.model).exists():
        from fly_driver import FlyDriver
        driver = FlyDriver(n_beams=env.n_beams, lidar_range=env.lidar_range,
                           dt=env.dt).load(args.model)
        runs.append(("초파리 배선", *drive(env, driver)))

    rows, cols = track.free.shape
    res = track.resolution
    extent = [track.origin[0], track.origin[0] + cols * res,
              track.origin[1], track.origin[1] + rows * res]

    fig, axes = plt.subplots(1, len(runs), figsize=(9 * len(runs), 7), dpi=100)
    axes = np.atleast_1d(axes)
    for ax, (name, path, speeds, info) in zip(axes, runs):
        ax.imshow(track.free, cmap="gray", extent=extent, origin="upper")
        points = ax.scatter(path[:, 0], path[:, 1], c=speeds, cmap="turbo",
                            s=6, vmin=0, vmax=7, zorder=3)
        ax.plot(*path[0], marker="o", color="lime", markersize=10, zorder=4)
        if info["crashed"]:
            ax.plot(*path[-1], marker="X", color="red", markersize=16, zorder=5)
        ax.set_title(
            f"{name}\n{info['lap_time']:.2f}초  진행 {info['progress']:.0%}  "
            f"{'충돌' if info['crashed'] else '완주'}", fontsize=13)
        ax.set_xlim(path[:, 0].min() - 3, path[:, 0].max() + 3)
        ax.set_ylim(path[:, 1].min() - 3, path[:, 1].max() + 3)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.colorbar(points, ax=ax, label="speed [m/s]", shrink=0.7)

    plt.rcParams["font.family"] = "Malgun Gothic"
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"저장: {args.out}")
    for name, path, speeds, info in runs:
        print(f"  {name:<22} {info['lap_time']:>6.2f}초  "
              f"진행 {info['progress']:>6.1%}  "
              f"평균 {speeds.mean():.1f} m/s  "
              f"{'충돌' if info['crashed'] else '완주'}")


if __name__ == "__main__":
    main()
