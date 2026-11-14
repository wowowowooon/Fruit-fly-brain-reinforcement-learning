r"""README 에 들어가는 그림을 만든다. 결과는 docs/images/ 에 저장한다.

  .\.venv\Scripts\python.exe tools\make_figures.py            전부
  .\.venv\Scripts\python.exe tools\make_figures.py circuit    하나만

  circuit   초파리 회로 871개 뉴런과 연결 9,992개 (data/subcircuit_*.csv)
  curve     강화학습 완주율 곡선, 초파리 배선 vs 밀집망 (logs/rl_out.log, logs/rl_B_out.log)
  compare   초파리 배선 vs 밀집망 성적 (models/*.json)
  course    최종 모델이 대회 코스에서 피해 가는 세 장면 (대회 맵, T_fly871_final.pt 필요)
  gif       대회 코스 주행 애니메이션 (대회 맵, T_fly871_final.pt 필요)
"""

import json
import re
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from flyracer.paths import DATA, LOGS, MODELS, RACE_MAP, ROOT

OUT = ROOT / "docs" / "images"

# 색 (dataviz 기본 팔레트, 밝은 배경용)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
FLY = "#2a78d6"      # 초파리 배선
DENSE = "#eb6834"    # 밀집망
SENSORY = "#1baf7a"
MOTOR = "#6250d6"
EXC = "#2a78d6"
INH = "#e34948"


def style():
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            matplotlib.font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"저장 {OUT / name}")


# ---------------------------------------------------------------- 회로
def circuit():
    neurons = pd.read_csv(DATA / "subcircuit_neurons.csv")
    links = pd.read_csv(DATA / "subcircuit_connections.csv")
    inhibitory = {"gaba", "glutamate"}
    nt = dict(zip(neurons.bodyId, neurons.nt))

    # 감각 뉴런에서 몇 단계 떨어졌는지로 가로 위치를 정한다
    nxt = {}
    for a, b in zip(links.bodyId_pre, links.bodyId_post):
        nxt.setdefault(a, []).append(b)
    depth = {n: 0 for n in neurons.bodyId[neurons.role == "sensory"]}
    queue = deque(depth)
    while queue:
        a = queue.popleft()
        for b in nxt.get(a, []):
            if b not in depth:
                depth[b] = depth[a] + 1
                queue.append(b)
    inner_max = max([depth.get(n, 1) for n in neurons.bodyId[neurons.role == "inter"]] + [1])

    rng = np.random.default_rng(0)
    pos = {}
    columns = {}
    for row in neurons.itertuples():
        if row.role == "sensory":
            col = 0
        elif row.role == "motor":
            col = inner_max + 1
        else:
            col = min(max(depth.get(row.bodyId, 1), 1), inner_max)
        columns.setdefault(col, []).append(row)
    for col, rows in columns.items():
        rows.sort(key=lambda r: (str(r.type), r.bodyId))
        ys = np.linspace(-1, 1, len(rows)) if len(rows) > 1 else [0.0]
        for r, y in zip(rows, ys):
            pos[r.bodyId] = (col + rng.uniform(-0.28, 0.28), y)

    fig, ax = plt.subplots(figsize=(12, 6.2))
    for sign, color, alpha in ((1, EXC, 0.05), (-1, INH, 0.07)):
        segs = [(pos[a], pos[b]) for a, b in zip(links.bodyId_pre, links.bodyId_post)
                if (-1 if nt.get(a) in inhibitory else 1) == sign]
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segs, colors=color, linewidths=0.4, alpha=alpha, zorder=1))
    for role, color, size in (("inter", MUTED, 6), ("sensory", SENSORY, 9), ("motor", MOTOR, 9)):
        ids = neurons.bodyId[neurons.role == role]
        xy = np.array([pos[i] for i in ids])
        ax.scatter(xy[:, 0], xy[:, 1], s=size, c=color, edgecolors=SURFACE,
                   linewidths=0.3, zorder=3)

    n_inh = sum(nt.get(a) in inhibitory for a in links.bodyId_pre)
    labels = [
        (0, "감각 뉴런", f"{(neurons.role == 'sensory').sum()}개 · 라이다 시야에 배정"),
        ((inner_max + 1) / 2, "중간 뉴런", f"{(neurons.role == 'inter').sum()}개"),
        (inner_max + 1, "하강 뉴런 (출력)", f"{(neurons.role == 'motor').sum()}개 · 조향·구동"),
    ]
    for x, title, sub in labels:
        ax.text(x, 1.16, title, ha="center", fontsize=12, color=INK, weight="bold")
        ax.text(x, 1.08, sub, ha="center", fontsize=10, color=INK_2)
    ax.plot([], [], color=EXC, lw=2, label=f"흥분성 연결 {len(links) - n_inh:,}")
    ax.plot([], [], color=INH, lw=2, label=f"억제성 연결 {n_inh:,} ({n_inh / len(links):.0%})")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False,
              fontsize=10, labelcolor=INK_2)
    ax.set_xlim(-0.6, inner_max + 1.6)
    ax.set_ylim(-1.08, 1.24)
    ax.axis("off")
    ax.set_title("초파리 커넥톰에서 추출한 주행 회로 — 뉴런 871개, 연결 9,992개",
                 fontsize=14, loc="left", pad=18)
    save(fig, "circuit.png")


# ---------------------------------------------------------------- 학습 곡선
def read_curve(path):
    rows = []
    pat = re.compile(r"^강화\s+(\d+)/\d+.*?완주\s+(\d+)%")
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = pat.match(line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2))))
    df = pd.DataFrame(rows, columns=["it", "done"]).drop_duplicates("it")
    df["smooth"] = df.done.rolling(50, min_periods=1).mean()
    return df


def curve():
    fly = read_curve(LOGS / "rl_out.log")
    dense = read_curve(LOGS / "rl_B_out.log")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axhline(80, color=MUTED, lw=1, ls=(0, (4, 4)), zorder=1)
    ax.text(dense.it.max(), 81.5, "완주율 80%", ha="right", va="bottom", fontsize=9, color=INK_2)
    for df, color, name in ((dense, DENSE, "밀집망 (전부 연결)"), (fly, FLY, "초파리 배선")):
        ax.plot(df.it, df.smooth, color=color, lw=2, label=name, zorder=3)
        hit = df[df.smooth >= 80]
        if len(hit):
            x = int(hit.it.iloc[0])
            ax.plot(x, 80, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=4)
            ax.annotate(f"{x}회차", (x, 80), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=10, color=INK)
            print(f"  {name}: 80% 첫 도달 {x}회차 (50회차 이동평균)")
    ax.set_xlabel("강화학습 회차")
    ax.set_ylabel("학습 트랙 완주율 (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(fly.it.max(), dense.it.max()))
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_2)
    ax.set_title("강화학습 수렴 속도 — 뉴런 수는 같고 배선만 다르다 (50회차 이동평균)",
                 fontsize=13, loc="left")
    save(fig, "learning_curve.png")


# ---------------------------------------------------------------- 성적 비교
def compare():
    def j(name):
        return json.load(open(MODELS / f"{name}.json", encoding="utf-8"))
    c, b = j("C_fly871"), j("B_dense871")
    dc, db = j("dynamic_C_fly871"), j("dynamic_B_dense871")
    rows = [
        ("기본 주행\n(처음 보는 트랙 45개)", c["completed"] / c["tracks"], b["completed"] / b["tracks"],
         f'{c["completed"]}/{c["tracks"]}', f'{b["completed"]}/{b["tracks"]}'),
        ("움직이는 상대 차\n(학습에 없던 조건)", dc["moving_completed"] / dc["tracks"],
         db["moving_completed"] / db["tracks"],
         f'{dc["moving_completed"]}/{dc["tracks"]}', f'{db["moving_completed"]}/{db["tracks"]}'),
        ("제어 지연 150 ms\n(학습의 3배)", 14 / 20, 11 / 20, "14/20", "11/20"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    h = 0.34
    for i, (label, vc, vb, tc, tb) in enumerate(rows):
        y = len(rows) - 1 - i
        for dy, v, t, color in ((h / 2 + 0.01, vc, tc, FLY), (-h / 2 - 0.01, vb, tb, DENSE)):
            ax.barh(y + dy, v * 100, height=h, color=color, zorder=2)
            ax.text(v * 100 + 1.2, y + dy, f"{v:.0%}  ({t})", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("완주율 (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    from matplotlib.patches import Patch
    handles = [Patch(color=FLY, label=f"초파리 배선 · 파라미터 {c['params_used']:,}"),
               Patch(color=DENSE, label=f"밀집망 · 파라미터 {b['params_used']:,}")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
              labelcolor=INK_2)
    ax.set_title("학습한 조건에서는 밀집망, 학습에서 못 본 조건에서는 초파리 배선이 앞섰다",
                 fontsize=13, loc="left")
    save(fig, "wiring_compare.png")


# ---------------------------------------------------------------- 대회 코스
def _race_setup():
    from flyracer.brain.fly_driver import FlyDriver
    from flyracer.sim.f1tenth_sim import LIDAR_RANGE
    from flyracer.sim.track_map import MapTrack
    track = MapTrack(str(RACE_MAP))
    driver = FlyDriver(n_beams=541, lidar_range=LIDAR_RANGE, fly_wiring=True, motion=6)
    driver.load("T_fly871_final.pt")
    return track, driver


def _map_image(ax, track, alpha=1.0):
    rows, cols = track.free.shape
    res = track.resolution
    # 주행 가능 영역은 밝게, 벽 바깥은 어둡게
    img = np.where(track.free, 0.97, 0.16)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, alpha=alpha,
              extent=[track.origin[0], track.origin[0] + cols * res,
                      track.origin[1], track.origin[1] + rows * res], origin="upper")


def course():
    sys.path.insert(0, str(ROOT / "view"))
    from race_avoid import build, drive
    track, driver = _race_setup()
    scenes = []
    for title, obstacles, rivals in (("벽만", 0, 0), ("정지 장애물 6개", 6, 0), ("상대 차 2대", 0, 2)):
        env = build(track, 6, 541, obstacles, rivals, 1)
        scenes.append((title, drive(driver, env)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 11.5))
    for ax, (title, s) in zip(axes, scenes):
        _map_image(ax, track)
        for blob in s["blobs"]:
            ax.add_patch(Polygon(blob, closed=True, facecolor=INH, edgecolor=SURFACE, lw=1, zorder=4))
        for keep in s["rival_paths"]:
            if len(keep):
                ax.plot(keep[:, 0], keep[:, 1], color=DENSE, lw=1.4, alpha=0.6, zorder=3)
        for box in s["rival_boxes"]:
            ax.add_patch(Polygon(box, closed=True, facecolor=DENSE, edgecolor=SURFACE, lw=1, zorder=5))
        p = s["path"]
        dots = ax.scatter(p[:, 0], p[:, 1], c=s["speed"], cmap="viridis", s=6, vmin=0, vmax=7, zorder=6)
        info = s["info"]
        ax.set_title(f"{title}  —  {info['lap_time']:.2f}초 · "
                     f"{'완주' if not info['crashed'] else '충돌'} · 벽까지 최소 {s['gap'] * 100:+.0f} cm",
                     fontsize=12, loc="left")
        ax.set_xlim(p[:, 0].min() - 1.0, p[:, 0].max() + 1.0)
        ax.set_ylim(p[:, 1].min() - 1.0, p[:, 1].max() + 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
    cb = fig.colorbar(dots, ax=axes, shrink=0.35, pad=0.02)
    cb.set_label("속도 (m/s)")
    cb.outline.set_visible(False)
    fig.suptitle("2026 대회 코스 (학습에 쓰지 않은 맵) — 빨강: 정지 장애물, 주황: 상대 차",
                 fontsize=13, x=0.05, y=0.93, ha="left")
    save(fig, "race_course.png")


def gif():
    from PIL import Image
    from flyracer.sim.f1tenth_sim import LIDAR_RANGE
    sys.path.insert(0, str(ROOT / "view"))
    from race_avoid import SEED, build
    track, driver = _race_setup()
    env = build(track, 6, 541, 0, 2, 1)
    obs, _ = env.reset(seed=SEED)
    driver.reset()
    frames_data = []
    for step in range(6000):
        scan = obs[:env.n_beams] * env.lidar_range
        frames_data.append({
            "pos": env.pos.copy(), "yaw": env.yaw, "speed": env.speed,
            "origin": env.lidar_pos.copy(), "angles": env.beam_angles + env.yaw,
            "scan": np.minimum(scan, LIDAR_RANGE),
            "rivals": [env._rival_box(r) for r in env.rivals],
            "t": step * env.dt if hasattr(env, "dt") else None,
        })
        obs, _, done, truncated, info = env.step(driver.act(obs))
        if done or truncated:
            break
    lap = info["lap_time"]
    print(f"  {len(frames_data)}스텝, {lap:.2f}초, {'충돌' if info['crashed'] else '완주'}")

    path = np.array([f["pos"] for f in frames_data])
    speeds = np.array([f["speed"] for f in frames_data])
    every = max(1, len(frames_data) // 150)        # 150장 안팎
    x0, x1 = path[:, 0].min() - 0.8, path[:, 0].max() + 0.8
    y0, y1 = path[:, 1].min() - 0.8, path[:, 1].max() + 0.8
    width = 9.0
    fig = plt.figure(figsize=(width, width * (y1 - y0) / (x1 - x0)), dpi=90)
    ax = fig.add_axes([0, 0, 1, 1])
    images = []
    for k in range(0, len(frames_data), every):
        f = frames_data[k]
        ax.clear()
        _map_image(ax, track)
        ends = f["origin"] + f["scan"][:, None] * np.c_[np.cos(f["angles"]), np.sin(f["angles"])]
        for e in ends[::12]:
            ax.plot([f["origin"][0], e[0]], [f["origin"][1], e[1]], color=FLY, lw=0.5, alpha=0.35, zorder=2)
        hit = f["scan"] < LIDAR_RANGE * 0.99
        ax.scatter(ends[hit, 0], ends[hit, 1], s=1.2, color=FLY, zorder=3)
        ax.plot(path[:k + 1, 0], path[:k + 1, 1], color=INK_2, lw=1.6, zorder=4)
        for box in f["rivals"]:
            ax.add_patch(Polygon(box, closed=True, facecolor=DENSE, edgecolor=SURFACE, lw=1, zorder=5))
        c, s_ = np.cos(f["yaw"]), np.sin(f["yaw"])
        body = np.array([[-0.27, -0.15], [0.33, -0.15], [0.33, 0.15], [-0.27, 0.15]])
        body = body @ np.array([[c, s_], [-s_, c]]) + f["pos"]
        ax.add_patch(Polygon(body, closed=True, facecolor=INK, edgecolor=SURFACE, lw=1, zorder=6))
        t = lap * k / max(len(frames_data) - 1, 1)
        ax.text(0.01, 0.97, f"초파리 회로 주행   {t:4.1f} s   {f['speed']:.1f} m/s",
                transform=ax.transAxes, va="top", fontsize=11, color=INK,
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=3))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.axis("off")
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
    plt.close(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    pal = images[0].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
    ms = int(1000 * lap / len(images))             # 실제 속도로 재생
    frames[0].save(OUT / "drive.gif", save_all=True, append_images=frames[1:],
                   duration=ms, loop=0, optimize=True)
    print(f"저장 {OUT / 'drive.gif'}  ({(OUT / 'drive.gif').stat().st_size / 1e6:.1f} MB)")


JOBS = {"circuit": circuit, "curve": curve, "compare": compare, "course": course, "gif": gif}

if __name__ == "__main__":
    style()
    for name in sys.argv[1:] or JOBS:
        JOBS[name]()
