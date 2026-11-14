r"""README 에 들어가는 그림을 만든다. 결과는 docs/images/ 에 저장한다.

  .\.venv\Scripts\python.exe tools\make_figures.py            전부
  .\.venv\Scripts\python.exe tools\make_figures.py circuit    하나만

  circuit   초파리 회로 871개 뉴런과 연결 9,992개 (data/subcircuit_*.csv)
  curve     강화학습 완주율 곡선, 초파리 배선 vs 밀집망 (logs/rl_out.log, logs/rl_B_out.log)
  compare   초파리 배선 vs 밀집망 성적 (models/*.json)
  course    최종 모델이 대회 코스에서 피해 가는 세 장면 (대회 맵, T_fly871_final.pt 필요)
  gif       대회 코스 주행 애니메이션 (대회 맵, T_fly871_final.pt 필요)

그림마다 GitHub 밝은 테마용(_light)과 어두운 테마용(_dark) 두 장을 만든다.
README 는 <picture> 태그로 보는 사람의 테마에 맞는 쪽을 보여 준다.
정적 그림은 배경을 투명하게 두어 GitHub 배경색이 그대로 비친다.
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
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Polygon

from flyracer.paths import DATA, LOGS, MODELS, RACE_MAP, ROOT

OUT = ROOT / "docs" / "images"

# 글자·선 색은 GitHub 자체 색을 따르고, 계열 색은 색각 이상 검사를 통과한 쌍을 쓴다.
THEMES = {
    "light": dict(
        bg="#ffffff", ink="#1f2328", ink2="#59636e", grid="#d1d9e0",
        fly="#2a78d6", dense="#eb6834", obstacle="#e34948",
        sensory="#1baf7a", inter="#8c959f", motor="#6250d6",
        free="#f6f8fa", wall="#818b98",
        speed=["#b9d3f2", "#2a78d6", "#0b3a75"],
    ),
    "dark": dict(
        bg="#0d1117", ink="#e6edf3", ink2="#9198a1", grid="#30363d",
        fly="#3987e5", dense="#d95926", obstacle="#e66767",
        sensory="#199e70", inter="#6e7681", motor="#9085e9",
        free="#21262d", wall="#0d1117",
        speed=["#1d4579", "#3987e5", "#cfe3ff"],
    ),
}
T = THEMES["light"]     # 지금 그리는 테마


def use_theme(name):
    global T
    T = THEMES[name]
    for font in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams["font.family"] = font
            break
        except Exception:
            continue
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": T["ink"], "axes.labelcolor": T["ink2"],
        "xtick.color": T["ink2"], "ytick.color": T["ink2"],
        "axes.edgecolor": T["grid"], "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "legend.labelcolor": T["ink2"],
    })


def save(fig, name, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_{theme}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.12, transparent=True)
    plt.close(fig)
    print(f"저장 {path}")


# ---------------------------------------------------------------- 회로
def circuit(theme):
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
    pos, columns = {}, {}
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

    fig, ax = plt.subplots(figsize=(12, 5.6))
    alpha = {"light": (0.05, 0.08), "dark": (0.07, 0.10)}[theme]
    for sign, color, a in ((1, T["fly"], alpha[0]), (-1, T["dense"], alpha[1])):
        segs = [(pos[p], pos[q]) for p, q in zip(links.bodyId_pre, links.bodyId_post)
                if (-1 if nt.get(p) in inhibitory else 1) == sign]
        ax.add_collection(LineCollection(segs, colors=color, linewidths=0.4, alpha=a, zorder=1))
    for role, size in (("inter", 6), ("sensory", 9), ("motor", 9)):
        xy = np.array([pos[i] for i in neurons.bodyId[neurons.role == role]])
        ax.scatter(xy[:, 0], xy[:, 1], s=size, c=T[role], linewidths=0, zorder=3)

    n_inh = sum(nt.get(a) in inhibitory for a in links.bodyId_pre)
    for x, title, sub in (
            (0, "감각 뉴런", f"{(neurons.role == 'sensory').sum()}개 · 라이다 시야"),
            ((inner_max + 1) / 2, "중간 뉴런", f"{(neurons.role == 'inter').sum()}개"),
            (inner_max + 1, "하강 뉴런", f"{(neurons.role == 'motor').sum()}개 · 조향·구동")):
        ax.text(x, 1.15, title, ha="center", fontsize=12.5, color=T["ink"], weight="bold")
        ax.text(x, 1.07, sub, ha="center", fontsize=10, color=T["ink2"])
    handles = [plt.Line2D([], [], color=T["fly"], lw=2.5, label=f"흥분성 연결 {len(links) - n_inh:,}"),
               plt.Line2D([], [], color=T["dense"], lw=2.5,
                          label=f"억제성 연결 {n_inh:,} ({n_inh / len(links):.0%})")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=2,
              frameon=False, fontsize=10)
    ax.set_xlim(-0.6, inner_max + 1.6)
    ax.set_ylim(-1.06, 1.22)
    ax.axis("off")
    save(fig, "circuit", theme)


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


def curve(theme):
    fly = read_curve(LOGS / "rl_out.log")
    dense = read_curve(LOGS / "rl_B_out.log")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axhline(80, color=T["ink2"], lw=1, ls=(0, (3, 3)), zorder=1)
    # 80% 첫 도달 표시. 글자는 선이 없는 빈자리에 두고 가는 선으로 잇는다
    spots = {"초파리 배선": (420, 96), "밀집망 (전부 연결)": (1719, 52)}
    for df, color, name in ((dense, T["dense"], "밀집망 (전부 연결)"), (fly, T["fly"], "초파리 배선")):
        ax.plot(df.it, df.smooth, color=color, lw=2, zorder=3)
        hit = df[df.smooth >= 80]
        x = int(hit.it.iloc[0])
        ax.plot(x, 80, "o", ms=8, color=color, mec=T["bg"], mew=2, zorder=5)
        ax.annotate(f"{x:,}회차에 80% 도달", (x, 80), xytext=spots[name], textcoords="data",
                    ha="center", va="center", fontsize=10, color=T["ink"], weight="bold", zorder=6,
                    bbox=dict(facecolor=T["bg"], edgecolor="none", pad=2),
                    arrowprops=dict(arrowstyle="-", color=T["ink2"], lw=0.8, shrinkA=2, shrinkB=5))
        # 선 끝에 이름을 바로 붙인다
        ax.text(df.it.iloc[-1] + 25, df.smooth.iloc[-1], name, va="center", fontsize=10.5, color=T["ink"])
    ax.set_xlabel("강화학습 회차")
    ax.set_ylabel("학습 트랙 완주율 (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(fly.it.max(), dense.it.max()))
    ax.yaxis.grid(True, color=T["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "learning_curve", theme)


# ---------------------------------------------------------------- 성적 비교
def compare(theme):
    def j(name):
        return json.load(open(MODELS / f"{name}.json", encoding="utf-8"))
    c, b = j("C_fly871"), j("B_dense871")
    dc, db = j("dynamic_C_fly871"), j("dynamic_B_dense871")
    rows = [
        ("기본 주행", "처음 보는 트랙 45개", c["completed"], b["completed"], c["tracks"]),
        ("움직이는 상대 차", "학습에 없던 조건", dc["moving_completed"], db["moving_completed"], dc["tracks"]),
        ("제어 지연 150 ms", "학습의 3배", 14, 11, 20),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    h, gap = 0.32, 0.03
    for i, (label, sub, nc, nb, n) in enumerate(rows):
        y = len(rows) - 1 - i
        for dy, k, color in ((h / 2 + gap, nc, T["fly"]), (-h / 2 - gap, nb, T["dense"])):
            ax.barh(y + dy, k / n * 100, height=h, color=color, zorder=2)
            ax.text(k / n * 100 + 1.2, y + dy, f"{k / n:.0%}", va="center", fontsize=10.5,
                    color=T["ink"], weight="bold")
            ax.text(k / n * 100 + 8.5, y + dy, f"{k}/{n}", va="center", fontsize=9.5, color=T["ink2"])
        ax.text(-2, y + 0.07, label, ha="right", va="bottom", fontsize=11, color=T["ink"])
        ax.text(-2, y - 0.07, sub, ha="right", va="top", fontsize=9.5, color=T["ink2"])
    ax.set_yticks([])
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.xaxis.grid(True, color=T["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    handles = [Patch(color=T["fly"], label=f"초파리 배선 · 파라미터 {c['params_used']:,}"),
               Patch(color=T["dense"], label=f"밀집망 · 파라미터 {b['params_used']:,}")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.08), ncol=2, frameon=False)
    save(fig, "wiring_compare", theme)


# ---------------------------------------------------------------- 대회 코스
_RACE = {}


def _race_setup():
    if not _RACE:
        from flyracer.brain.fly_driver import FlyDriver
        from flyracer.sim.f1tenth_sim import LIDAR_RANGE
        from flyracer.sim.track_map import MapTrack
        track = MapTrack(str(RACE_MAP))
        driver = FlyDriver(n_beams=541, lidar_range=LIDAR_RANGE, fly_wiring=True, motion=6)
        driver.load("T_fly871_final.pt")
        _RACE.update(track=track, driver=driver)
    return _RACE["track"], _RACE["driver"]


def _map_image(ax, track):
    rows, cols = track.free.shape
    res = track.resolution
    rgb = np.where(track.free[..., None],
                   np.array(matplotlib.colors.to_rgb(T["free"])),
                   np.array(matplotlib.colors.to_rgb(T["wall"])))
    ax.imshow(rgb, extent=[track.origin[0], track.origin[0] + cols * res,
                           track.origin[1], track.origin[1] + rows * res], origin="upper")


def _scenes():
    if "scenes" not in _RACE:
        sys.path.insert(0, str(ROOT / "view"))
        from race_avoid import build, drive
        track, driver = _race_setup()
        scenes = []
        for title, obstacles, rivals in (("벽만", 0, 0), ("정지 장애물 6개", 6, 0), ("상대 차 2대", 0, 2)):
            env = build(track, 6, 541, obstacles, rivals, 1)
            scenes.append((title, drive(driver, env)))
        _RACE["scenes"] = scenes
    return _RACE["scenes"]


def course(theme):
    track, _ = _race_setup()
    scenes = _scenes()
    cmap = LinearSegmentedColormap.from_list("speed", T["speed"])
    fig, axes = plt.subplots(3, 1, figsize=(10, 10.6))
    for ax, (title, s) in zip(axes, scenes):
        _map_image(ax, track)
        for blob in s["blobs"]:
            ax.add_patch(Polygon(blob, closed=True, facecolor=T["obstacle"], edgecolor="none", zorder=4))
        for keep in s["rival_paths"]:
            if len(keep):
                ax.plot(keep[:, 0], keep[:, 1], color=T["dense"], lw=1.3, alpha=0.6, zorder=3)
        for box in s["rival_boxes"]:
            ax.add_patch(Polygon(box, closed=True, facecolor=T["dense"], edgecolor="none", zorder=5))
        p = s["path"]
        dots = ax.scatter(p[:, 0], p[:, 1], c=s["speed"], cmap=cmap, s=6, vmin=0, vmax=7,
                          linewidths=0, zorder=6)
        info = s["info"]
        ax.set_title(f"{title}", fontsize=12, loc="left", color=T["ink"], weight="bold")
        ax.set_title(f"{info['lap_time']:.2f}초 · {'완주' if not info['crashed'] else '충돌'} · "
                     f"벽까지 최소 {s['gap'] * 100:+.0f} cm", fontsize=10.5, loc="right", color=T["ink2"])
        ax.set_xlim(p[:, 0].min() - 1.0, p[:, 0].max() + 1.0)
        ax.set_ylim(p[:, 1].min() - 1.0, p[:, 1].max() + 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
    cb = fig.colorbar(dots, ax=axes, shrink=0.3, pad=0.02, aspect=18)
    cb.set_label("속도 (m/s)", color=T["ink2"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=T["ink2"], length=0)
    handles = [Patch(color=T["obstacle"], label="정지 장애물"), Patch(color=T["dense"], label="상대 차와 그 궤적")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.45, 0.06))
    save(fig, "race_course", theme)


def _drive_frames():
    if "frames" not in _RACE:
        from flyracer.sim.f1tenth_sim import LIDAR_RANGE
        sys.path.insert(0, str(ROOT / "view"))
        from race_avoid import SEED, build
        track, driver = _race_setup()
        env = build(track, 6, 541, 0, 2, 1)
        obs, _ = env.reset(seed=SEED)
        driver.reset()
        frames = []
        for _ in range(6000):
            frames.append({
                "pos": env.pos.copy(), "yaw": env.yaw, "speed": env.speed,
                "origin": env.lidar_pos.copy(), "angles": env.beam_angles + env.yaw,
                "scan": np.minimum(obs[:env.n_beams] * env.lidar_range, LIDAR_RANGE),
                "rivals": [env._rival_box(r) for r in env.rivals],
            })
            obs, _, done, truncated, info = env.step(driver.act(obs))
            if done or truncated:
                break
        print(f"  {len(frames)}스텝, {info['lap_time']:.2f}초, {'충돌' if info['crashed'] else '완주'}")
        _RACE.update(frames=frames, lap=info["lap_time"], lidar_range=LIDAR_RANGE)
    return _RACE["frames"], _RACE["lap"], _RACE["lidar_range"]


def gif(theme):
    from PIL import Image
    track, _ = _race_setup()
    frames_data, lap, lidar_range = _drive_frames()
    path = np.array([f["pos"] for f in frames_data])
    every = max(1, len(frames_data) // 150)
    x0, x1 = path[:, 0].min() - 0.8, path[:, 0].max() + 0.8
    y0, y1 = path[:, 1].min() - 0.8, path[:, 1].max() + 0.8
    width = 9.0
    fig = plt.figure(figsize=(width, width * (y1 - y0) / (x1 - x0)), dpi=90, facecolor=T["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    images = []
    for k in range(0, len(frames_data), every):
        f = frames_data[k]
        ax.clear()
        ax.set_facecolor(T["bg"])
        _map_image(ax, track)
        ends = f["origin"] + f["scan"][:, None] * np.c_[np.cos(f["angles"]), np.sin(f["angles"])]
        rays = [(f["origin"], e) for e in ends[::12]]
        ax.add_collection(LineCollection(rays, colors=T["fly"], linewidths=0.5, alpha=0.35, zorder=2))
        hit = f["scan"] < lidar_range * 0.99
        ax.scatter(ends[hit, 0], ends[hit, 1], s=1.2, color=T["fly"], zorder=3)
        ax.plot(path[:k + 1, 0], path[:k + 1, 1], color=T["ink2"], lw=1.6, zorder=4)
        for box in f["rivals"]:
            ax.add_patch(Polygon(box, closed=True, facecolor=T["dense"], edgecolor="none", zorder=5))
        c, s_ = np.cos(f["yaw"]), np.sin(f["yaw"])
        body = np.array([[-0.27, -0.15], [0.33, -0.15], [0.33, 0.15], [-0.27, 0.15]])
        body = body @ np.array([[c, s_], [-s_, c]]) + f["pos"]
        ax.add_patch(Polygon(body, closed=True, facecolor=T["ink"], edgecolor="none", zorder=6))
        t = lap * k / max(len(frames_data) - 1, 1)
        ax.text(0.012, 0.96, f"{t:4.1f} s   {f['speed']:.1f} m/s", transform=ax.transAxes,
                va="top", fontsize=11, color=T["ink"], family="monospace",
                bbox=dict(facecolor=T["bg"], edgecolor="none", alpha=0.85, pad=3))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.axis("off")
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
    plt.close(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    pal = images[len(images) // 2].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
    out = OUT / f"drive_{theme}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 * lap / len(images)), loop=0, optimize=True)
    print(f"저장 {out}  ({out.stat().st_size / 1e6:.1f} MB)")


JOBS = {"circuit": circuit, "curve": curve, "compare": compare, "course": course, "gif": gif}

if __name__ == "__main__":
    for name in sys.argv[1:] or JOBS:
        for theme in THEMES:
            use_theme(theme)
            JOBS[name](theme)
