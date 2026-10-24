"""plots.py —— Task2 的结果插图"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import calib as calib_mod

__all__ = ["make_all", "render_bev"]


def _setup_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback",
                 "AR PL UMing CN", "WenQuanYi Zen Hei"]:
        if name in have:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    return plt


def render_bev(frame, cal, X0=-9.0, X1=9.0, Z0=24.0, Z1=42.0, m_per_px=0.05):
    """把透视图按路面度量重采样成鸟瞰图（真逆透视，非近似）。"""
    w = int((X1 - X0) / m_per_px)
    h = int((Z1 - Z0) / m_per_px)
    Xs = X0 + (np.arange(w) + 0.5) * m_per_px
    Zs = Z1 - (np.arange(h) + 0.5) * m_per_px      # 远处在上
    XX, ZZ = np.meshgrid(Xs, Zs)
    U, V = cal.ground_to_pixel(XX, ZZ)
    bev = cv2.remap(frame, U.astype(np.float32), V.astype(np.float32),
                    cv2.INTER_LINEAR, borderValue=(25, 25, 25))
    return bev, (X0, X1, Z0, Z1, m_per_px)


def _gz_to_px(X, Z, ext):
    X0, X1, Z0, Z1, m = ext
    return (np.asarray(X) - X0) / m, (Z1 - np.asarray(Z)) / m


# ==========================================================================
def fig_calibration(plt, cal, poly, frame, path):
    """标定结果：车道线、着色区域、计数线、以及路面米制网格。"""
    import run_task2 as R

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    ax = axes[0]
    vis = R.draw_tint(frame.copy(), poly, 0.20)
    vs = np.linspace(560, 1078, 2)
    for ln, col, name in [(cal.p.line_a, (0, 0, 255), "车道线 A"),
                          (cal.p.line_b, (255, 0, 255), "车道线 B"),
                          (cal.p.line_right, (255, 255, 0), "道路右边缘")]:
        a, b = ln
        cv2.line(vis, (int(a * vs[0] + b), int(vs[0])), (int(a * vs[1] + b), int(vs[1])), col, 2)
    a1, b1 = cal.p.line_a
    ar, br = cal.p.line_right
    cv2.line(vis, (int(a1 * R.COUNT_ROW + b1), R.COUNT_ROW),
             (int(ar * R.COUNT_ROW + br), R.COUNT_ROW), (0, 140, 255), 3)
    cv2.putText(vis, "counting line", (int(a1 * R.COUNT_ROW + b1) + 8, R.COUNT_ROW - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title("(a) 着色区域(绿) + 车道线 + 计数线", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # 路面米制网格投影回图像，直观展示标定的几何
    ax = axes[1]
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for Z in np.arange(24, 45, 2):
        Xs = np.linspace(-9, 9, 60)
        U, V = cal.ground_to_pixel(Xs, np.full_like(Xs, Z))
        ax.plot(U, V, color="cyan", lw=0.7, alpha=0.8)
    for X in np.arange(-8, 9, 2):
        Zs = np.linspace(24, 44, 60)
        U, V = cal.ground_to_pixel(np.full_like(Zs, X), Zs)
        ax.plot(U, V, color="cyan", lw=0.7, alpha=0.8)
    ax.set_xlim(0, 1920); ax.set_ylim(1080, 540)
    ax.set_title(f"(b) 路面 2m×2m 米制网格：h={cal.h:.1f}m, f={cal.f:.0f}px, "
                 f"俯仰={cal.theta_deg:.1f}°, 偏航={cal.yaw_deg:.1f}°", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_bev_tracks(plt, res, rows, cal, path):
    """鸟瞰图上的车辆轨迹。"""
    import run_task2 as R

    cap = cv2.VideoCapture(str(R.VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(40, res["n_frames"] - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = np.zeros((1080, 1920, 3), np.uint8)
    bev, ext = render_bev(frame, cal, X0=-10, X1=10, Z0=23, Z1=43, m_per_px=0.05)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB), extent=[ext[0], ext[1], ext[2], ext[3]])
    speeds = {r["track_id"]: r["speed_kmh"] for r in rows}
    cmap = plt.get_cmap("turbo")
    for t in res["tracks"]:
        arr = np.array(t.track)
        if len(arr) < 4:
            continue
        sp = speeds.get(t.id)
        col = cmap(min(1.0, (sp if isinstance(sp, float) else 0) / 90.0)) if sp != "" else "0.6"
        ax.plot(arr[:, 1], arr[:, 2], "-", color=col, lw=2.0, alpha=0.95)
        ax.plot(arr[0, 1], arr[0, 2], "o", color=col, ms=4)
        ax.annotate(f"#{t.id}", (arr[-1, 1], arr[-1, 2]), color="white", fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("X / m（横向）")
    ax.set_ylabel("Z / m（纵向，越大越远）")
    ax.set_title("鸟瞰图上的车辆轨迹（颜色=速度，蓝→红 由慢到快）", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 90))
    fig.colorbar(sm, ax=ax, label="速度 / (km/h)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_speed(plt, rows, summary, path):
    """速度分布 + 逐车速度。"""
    counted = [r for r in rows if r["crossed_count_line"] and r["speed_kmh"] != ""]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sp = np.array([r["speed_kmh"] for r in counted], dtype=float)
    ax = axes[0]
    if len(sp):
        ax.hist(sp, bins=max(4, int(len(sp) / 1.5)), color="tab:blue", alpha=0.75,
                edgecolor="k")
        ax.axvline(sp.mean(), color="r", ls="--", label=f"均值 {sp.mean():.1f}")
        ax.axvline(np.median(sp), color="g", ls=":", label=f"中位数 {np.median(sp):.1f}")
        ax.legend(fontsize=9)
    ax.set_xlabel("速度 / (km/h)"); ax.set_ylabel("车辆数")
    ax.set_title(f"(a) 越线车辆的速度分布（n={len(sp)}）", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ids = [f"#{r['track_id']}" for r in counted]
    vals = [r["speed_kmh"] for r in counted]
    ax.barh(ids, vals, color="tab:green", alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"{v:.0f}", va="center", fontsize=8)
    ax.set_xlabel("速度 / (km/h)")
    ax.set_title("(b) 逐车速度", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_timeline(plt, res, summary, path):
    """累计计数随时间变化 + 车辆在区域内的时间条。"""
    fps = res["fps"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ev = sorted(res["count_events"], key=lambda e: e["t_cross_s"])
    ts = [e["t_cross_s"] for e in ev]
    axes[0].step([0] + ts + [res["n_frames"] / fps], list(range(len(ts) + 1)) + [len(ts)],
                 where="post", color="tab:red", lw=2)
    axes[0].plot(ts, range(1, len(ts) + 1), "o", color="tab:red", ms=5)
    axes[0].set_ylabel("累计计数")
    axes[0].set_title(f"全年累计计数（共 {len(ts)} 辆）", fontsize=10)
    axes[0].grid(alpha=0.3)

    colors = plt.get_cmap("tab20")
    for i, t in enumerate(res["tracks"]):
        t0 = t.track[0][0] / fps
        t1 = t.track[-1][0] / fps
        axes[1].barh(i, t1 - t0, left=t0, height=0.6, color=colors(i % 20))
        axes[1].text(t0, i, f" #{t.id}", va="center", fontsize=7)
    axes[1].set_xlabel("时间 / s")
    axes[1].set_ylabel("航迹")
    axes[1].set_title("每条航迹在区域内出现的时段", fontsize=10)
    axes[1].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_gallery(plt, res, rows, path, video_path, per_track=5, thumb_h=90):
    """逐车缩略图长廊：用于人工核对计数与轨迹是否可靠。"""
    speeds = {r["track_id"]: r["speed_kmh"] for r in rows}
    counted = {r["track_id"]: r["crossed_count_line"] for r in rows}
    need: dict[int, list[int]] = {}
    for t in res["tracks"]:
        fs = [o[0] for o in t.track]
        if not fs:
            continue
        idx = np.linspace(0, len(fs) - 1, min(per_track, len(fs))).astype(int)
        need[t.id] = [fs[i] for i in idx]

    wanted = {f for v in need.values() for f in v}
    frames: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    fi = 0
    while wanted:
        ok, fr = cap.read()
        if not ok:
            break
        if fi in wanted:
            frames[fi] = fr.copy()
            wanted.discard(fi)
        fi += 1
    cap.release()

    n = len(res["tracks"])
    if n == 0:
        return
    fig, axes = plt.subplots(n, per_track, figsize=(per_track * 2.0, n * 1.55))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, t in enumerate(res["tracks"]):
        sp = speeds.get(t.id)
        tag = "✓计入" if counted.get(t.id) else "—"
        title = f"#{t.id}  {tag}"
        if sp != "" and sp is not None:
            title += f"  {sp:.0f} km/h"
        for j in range(per_track):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(title, fontsize=7.5, rotation=0, ha="right", va="center")
            if j >= len(need[t.id]):
                ax.axis("off")
                continue
            fr = frames.get(need[t.id][j])
            if fr is None:
                ax.axis("off")
                continue
            # 找到该帧的检测框位置
            bb = None
            for o in t.track:
                if o[0] == need[t.id][j]:
                    bb = o
                    break
            u, v = (bb[3], bb[4]) if bb else (960, 900)
            h = thumb_h * 2
            x0 = int(np.clip(u - h * 0.9, 0, fr.shape[1] - 1))
            x1 = int(np.clip(u + h * 0.9, 1, fr.shape[1]))
            y0 = int(np.clip(v - h * 1.6, 0, fr.shape[0] - 1))
            y1 = int(np.clip(v + h * 0.25, 1, fr.shape[0]))
            crop = fr[y0:y1, x0:x1]
            if crop.size == 0:
                ax.axis("off")
                continue
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    fig.suptitle("逐车核对长廊（每行一条航迹；每列 = 时序抽样；✓计入 = 已过计数线）",
                 fontsize=11)
    fig.tight_layout(rect=[0.06, 0, 1, 0.98])
    fig.savefig(path)
    plt.close(fig)


# ==========================================================================
def make_all(res, rows, summary, cal, figdir: Path):
    plt = _setup_font()
    from pathlib import Path as _P
    import run_task2 as R

    figdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(R.VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(40, res["n_frames"] - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = np.zeros((1080, 1920, 3), np.uint8)

    fig_calibration(plt, cal, res["poly"], frame, figdir / "fig_t2_calibration.png")
    fig_bev_tracks(plt, res, rows, cal, figdir / "fig_t2_bev_tracks.png")
    fig_speed(plt, rows, summary, figdir / "fig_t2_speed.png")
    fig_timeline(plt, res, summary, figdir / "fig_t2_timeline.png")
    fig_gallery(plt, res, rows, figdir / "fig_t2_track_gallery.png", R.VIDEO)
