"""viz.py —— 实验插图与结果汇总的输出"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import imops

__all__ = [
    "setup_font",
    "write_csv",
    "write_report",
    "fig01_gray_and_hist",
    "fig02_gt_annotation",
    "fig03_noise_median",
    "fig04_operator_responses",
    "fig05_binary_compare",
    "fig06_hough_accumulator",
    "fig07_line_extraction",
    "fig08_pr_bars",
    "fig09_zoom_line",
    "fig10_wire_profiles",
]


def setup_font() -> None:
    """配置 matplotlib 中文字体与基础样式。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib import font_manager

    prefer = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback",
              "AR PL UMing CN", "WenQuanYi Zen Hei", "SimHei"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in prefer:
        if name in have:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"


def _imshow(ax, img, title, cmap="gray", vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fn = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fn})


# ==========================================================================
def fig01_gray_and_hist(figdir: Path, A, A_ref, B3, B5) -> None:
    """图1：灰度图 A、中值滤波结果 B 及其灰度直方图。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    _imshow(axes[0, 0], A, "(a) 灰度图 A（由 Image1_withnoise.jpg 转换）")
    axes[0, 1].hist(A.ravel(), bins=256, range=(0, 255), color="0.35")
    axes[0, 1].set_title("(b) A 的灰度直方图", fontsize=10)
    axes[0, 1].set_xlim(0, 255)
    _imshow(axes[0, 2], B3, "(c) 中值滤波 3×3 结果 B")

    axes[1, 0].hist(A_ref.ravel(), bins=256, range=(0, 255), color="tab:blue", alpha=0.6, label="无噪原图")
    axes[1, 0].hist(A.ravel(), bins=256, range=(0, 255), color="tab:red", alpha=0.5, label="含噪图 A")
    axes[1, 0].set_title("(d) 含噪前后直方图对比", fontsize=10)
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].hist(B3.ravel(), bins=256, range=(0, 255), color="tab:green", alpha=0.7, label="B (3×3)")
    axes[1, 1].hist(B5.ravel(), bins=256, range=(0, 255), color="tab:orange", alpha=0.5, label="B (5×5)")
    axes[1, 1].set_title("(e) 中值滤波后直方图", fontsize=10)
    axes[1, 1].set_xlim(0, 255)
    axes[1, 1].legend(fontsize=8)

    _imshow(axes[1, 2], B5, "(f) 中值滤波 5×5 结果")
    fig.tight_layout()
    fig.savefig(figdir / "fig01_gray_and_hist.png")
    plt.close(fig)


def fig02_gt_annotation(figdir: Path, A, gt_lines, cfg) -> None:
    """图2：两条电力线的真值标注（含局部放大）。"""
    import matplotlib.pyplot as plt

    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]
    ax.imshow(A, cmap="gray")
    colors = ["#ff2d2d", "#00d26a"]
    for ln, col in zip(gt_lines, colors):
        (xa, ya), (xb, yb) = ln.endpoints()
        ax.plot([xa, xb], [ya, yb], color=col, lw=1.4,
                label=f"{ln.id}: {ln.equation_str()}  (θ={ln.theta_deg:.2f}°, 斜率角={ln.slope_angle_deg:.2f}°)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(a) 人工核定的两条电力线真值（叠加于含噪灰度图 A）", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # 局部放大：取 (x 380..470, y 90..150) 同时穿过两条线
    x0, x1, y0, y1 = 330, 470, 80, 180
    ax2 = axes[1]
    ax2.imshow(A[y0:y1, x0:x1], cmap="gray", extent=[x0, x1, y1, y0])
    for ln, col in zip(gt_lines, colors):
        (xa, ya), (xb, yb) = ln.endpoints()
        ax2.plot([xa, xb], [ya, yb], color=col, lw=1.0, alpha=0.9)
    ax2.set_title("(b) 局部放大（×3 视野）：细线与真值重合情况", fontsize=10)
    ax2.set_xticks([]); ax2.set_yticks([])
    axes[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1.0))
    fig.tight_layout()
    fig.savefig(figdir / "fig02_gt_annotation.png")
    plt.close(fig)


def fig03_noise_median(figdir: Path, A_ref, A, B3, B5) -> None:
    """图3：噪声特性与中值滤波的去噪效果（含局部放大）。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    x0, x1, y0, y1 = 60, 140, 40, 120
    for i, (img, t) in enumerate(zip([A_ref, A, B3, B5],
                                     ["(a) 无噪原图", "(b) 含噪图 A", "(c) 中值 3×3 (B)", "(d) 中值 5×5"])):
        _imshow(axes[0, i], img, t)
        _imshow(axes[1, i], img[y0:y1, x0:x1], t + " - 局部放大", vmin=img[y0:y1, x0:x1].min(),
                vmax=img[y0:y1, x0:x1].max() + 40)

    # 噪声统计：差分直方图
    d = A.astype(np.float32) - A_ref.astype(np.float32)
    print(f"[噪声分析] 差分均值={d.mean():.3f}, 标准差={d.std():.3f}, "
          f"|diff|>50 占比={float((np.abs(d) > 50).mean()):.5f}")
    fig.suptitle(f"含噪图与无噪原图的差分：均值={d.mean():.2f}, 标准差={d.std():.2f}（高斯型为主），"
                 f"另有 |diff|>50 的脉冲点占 {float((np.abs(d) > 50).mean()) * 100:.2f}%", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "fig03_noise_median.png")
    plt.close(fig)


def fig04_operator_responses(figdir: Path, A, respA) -> None:
    """图4：四种算子在 A 上的响应（含 Canny 二值结果与梯度方向）。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    _imshow(axes[0, 0], A, "(a) 灰度图 A")
    _imshow(axes[0, 1], imops.normalize_u8(respA["Robert"]), "(b) Robert 梯度幅值（|Gx|+|Gy|）")
    _imshow(axes[0, 2], imops.normalize_u8(respA["Sobel"]), "(c) Sobel 梯度幅值（sqrt(Gx²+Gy²)）")
    _imshow(axes[1, 0], imops.normalize_u8(respA["Laplacian"]), "(d) Laplacian 幅值（4 邻域，|∇²I|）")
    _imshow(axes[1, 1], imops.canny(A, sigma=1.4, strong_frac=0.03, weak_frac=0.15), "(e) Canny 二值边缘图")
    _imshow(axes[1, 2], imops.gradient_angle(A, sigma=1.4), "(f) Sobel 梯度方向场", cmap="hsv")
    fig.tight_layout()
    fig.savefig(figdir / "fig04_operator_responses.png")
    plt.close(fig)


def fig05_binary_compare(figdir: Path, A, binA, binB) -> None:
    """图5：A / B 两种输入下，四种算子二值边缘图对比。"""
    import matplotlib.pyplot as plt

    names = ["Robert", "Sobel", "Laplacian", "Canny"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for j, n in enumerate(names):
        _imshow(axes[0, j], binA[n], f"{n} —— 输入 A（含噪）", vmin=0, vmax=1)
        _imshow(axes[1, j], binB[n], f"{n} —— 输入 B（中值滤波后）", vmin=0, vmax=1)
    fig.tight_layout()
    fig.savefig(figdir / "fig05_binary_compare.png")
    plt.close(fig)


def fig06_hough_accumulator(figdir: Path, A, edge) -> None:
    """图6：Hough 累加器与峰值（体现'点在线上共线积累'的投票机制）。"""
    import matplotlib.pyplot as plt

    import hough as hg

    acc, rho_vals, theta_vals, ys, xs = hg.hough_accumulator(edge, 1.0, 1.0)
    peaks = hg.find_peaks(acc, rho_vals, theta_vals, threshold=max(25, int(0.25 * 511)),
                          nms_rho=6, nms_theta=4, max_peaks=20)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im = axes[0].imshow(acc, aspect="auto", origin="lower", cmap="inferno",
                        extent=[theta_vals[0], theta_vals[-1], rho_vals[0], rho_vals[-1]])
    axes[0].set_xlabel("θ / (°)"); axes[0].set_ylabel("ρ / (pixel)")
    axes[0].set_title("(a) Hough 累加器 A(ρ, θ)", fontsize=10)
    fig.colorbar(im, ax=axes[0], label="边缘点票数")
    for v, r, t in peaks[:6]:
        axes[0].plot([t], [r], "o", mfc="none", mec="cyan", ms=8)
        axes[0].annotate(f"{v}", (t, r), color="cyan", fontsize=8, xytext=(4, 4), textcoords="offset points")

    _imshow(axes[1], edge, "(b) 用于投票的二值边缘图")
    fig.tight_layout()
    fig.savefig(figdir / "fig06_hough_accumulator.png")
    plt.close(fig)


def fig07_line_extraction(figdir: Path, A, det, gt_lines) -> None:
    """图7：两条电力线的最终提取结果与线段边缘点序列。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    ax = axes[0]
    ax.imshow(A, cmap="gray")
    for ln in gt_lines:
        (xa, ya), (xb, yb) = ln.endpoints()
        ax.plot([xa, xb], [ya, yb], "--", color="yellow", lw=1.0, label=f"真值 {ln.id}")
    for i, d in enumerate(det, 1):
        (xa, ya), (xb, yb) = (0, d.y_at(0)), (511, d.y_at(511))
        ax.plot([xa, xb], [ya, yb], "-", lw=1.4, label=f"提取线 {i}")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(a) 提取结果与真值对比", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.imshow(A, cmap="gray")
    for i, d in enumerate(det, 1):
        if len(d.points):
            ax.scatter(d.points[:, 0], d.points[:, 1], s=2.5, label=f"线 {i}: {len(d.points)} 个边缘点")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(b) 线段边缘点序列（Hough 回溯得到）", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[2]
    ax.imshow(A, cmap="gray")
    colors = ["#ff2d2d", "#00d26a", "#1e90ff", "#ffa500"]
    for i, d in enumerate(det, 1):
        for (sx, sy), (ex, ey) in d.runs:
            ax.plot([sx, ex], [sy, ey], color=colors[(i - 1) % 4], lw=2.0)
    ax.set_title("(c) 线段的连通段（runs）：反映遮挡导致的断裂", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(figdir / "fig07_line_extraction.png")
    plt.close(fig)


def fig08_pr_bars(figdir: Path, rows: list[dict]) -> None:
    """图8：不同算子、不同输入下的 Precision / Recall / F1 对比。"""
    import matplotlib.pyplot as plt

    ops = ["Robert", "Sobel", "Laplacian", "Canny"]
    x = np.arange(len(ops))
    w = 0.13
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        for k, tag in enumerate(["A(含噪)", "B(中值滤波)"]):
            vals = [next(r[metric] for r in rows if r["operator"] == o and r["image"] == tag) for o in ops]
            off = (k - 0.5) * (w * 3)
            bars = ax.bar(x + off, vals, w * 2.6, label=tag)
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
        ax.set_xticks(x); ax.set_xticklabels(ops, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"({metric}) 电力线 ROI 内的 {metric}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig08_pr_bars.png")
    plt.close(fig)


def fig09_zoom_line(figdir: Path, A, B3, binA, binB, det, gt_lines) -> None:
    """图9：电力线区域的局部放大对比（A/B 输入下的边缘检测与提取结果）。"""
    import matplotlib.pyplot as plt

    x0, x1, y0, y1 = 300, 440, 60, 220
    sl = (slice(y0, y1), slice(x0, x1))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    _imshow(axes[0, 0], A[sl], "(a) A（含噪）局部")
    _imshow(axes[0, 1], binA["Sobel"][sl], "(b) A 上 Sobel 二值边缘", vmin=0, vmax=1)
    _imshow(axes[0, 2], B3[sl], "(c) B（中值滤波）局部")

    _imshow(axes[1, 0], binB["Sobel"][sl], "(d) B 上 Sobel 二值边缘", vmin=0, vmax=1)
    _imshow(axes[1, 1], binB["Canny"][sl], "(e) B 上 Canny 二值边缘", vmin=0, vmax=1)
    ax = axes[1, 2]
    ax.imshow(B3[sl], cmap="gray")
    for ln in gt_lines:
        (xa, ya), (xb, yb) = ln.endpoints()
        ax.plot([xa, xb], [ya, yb], "--", color="yellow", lw=1.0)
    for i, d in enumerate(det, 1):
        ax.plot([0, 511], [float(d.y_at(0)), float(d.y_at(511))], "-", lw=1.4, label=f"提取线 {i}")
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.legend(fontsize=8)
    ax.set_title("(f) 提取结果（黄虚线=真值）", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(figdir / "fig09_zoom_line.png")
    plt.close(fig)


def fig10_wire_profiles(figdir: Path, A, det) -> None:
    """图10：候选直线的法向灰度剖面 —— 说明如何区分'细暗线'与'宽明暗结构'。"""
    import matplotlib.pyplot as plt

    from run_experiment import wire_profile

    tests = [(d.rho, d.theta_deg, f"提取线 {i+1}") for i, d in enumerate(det)]
    tests += [(408.0, 77.5, "反例：θ=77.5°, ρ=408（宽浅盆地）"),
              (332.0, 94.5, "反例：θ=94.5°, ρ=332（地形阶跃）")]
    fig, axes = plt.subplots(1, len(tests), figsize=(4.0 * len(tests), 4.2))
    if len(tests) == 1:
        axes = [axes]
    for ax, (rho, th, name) in zip(axes, tests):
        offs, prof = wire_profile(A, rho, th, half=8)
        ax.plot(offs, prof, "o-", ms=4)
        ax.axvline(0, color="r", ls=":", lw=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("法向偏移 / pixel")
        ax.set_ylabel("沿线平均灰度")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "fig10_wire_profiles.png")
    plt.close(fig)


# ==========================================================================
def write_report(path: Path, s: dict) -> None:
    """把关键指标写成 Markdown，便于直接摘抄进实验报告。"""
    L = []
    L.append("# Project #2 实验结果汇总（自动生成）\n")
    L.append(f"- 输入图像：`{Path(s['image']).name}`，尺寸 {s['image_size'][1]}×{s['image_size'][0]}")
    L.append(f"- 匹配容差 tol = {s['tolerance_px']} px；评价 ROI = 真值线 ±{s['roi_halfwidth_px']} px\n")

    L.append("## 1. 电力线真值（人工核定）\n")
    L.append("| 线 | rho | theta(°) | 斜率角(°) | 直线方程 | 端点 |")
    L.append("|---|---|---|---|---|---|")
    for g in s["gt"]:
        (xa, ya), (xb, yb) = g["endpoints"]
        L.append(f"| {g['id']} | {g['rho']:.2f} | {g['theta_deg']:.2f} | {g['slope_angle_deg']:.2f} | "
                 f"`{g['equation']}` | ({xa:.0f},{ya:.1f})-({xb:.0f},{yb:.1f}) |")
    L.append("")

    L.append("## 2. 边缘检测指标（电力线 ROI 内，容差 %.1fpx）\n" % s["tolerance_px"])
    L.append("| 算子 | 输入 | Precision | Recall | F1 | 命中预测像素 | 被覆盖真值像素 | 预测像素总数 | 真值像素总数 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in s["edge_detection"]:
        L.append(f"| {r['operator']} | {r['image']} | {r['precision']:.4f} | {r['recall']:.4f} | "
                 f"{r['f1']:.4f} | {r['tp_pred']} | {r['tp_gt']} | {r['n_pred']} | {r['n_gt']} |")
    L.append("")
    L.append("### 中值滤波带来的提升（同算子，B 相对 A）\n")
    L.append("| 算子 | ΔPrecision | ΔRecall | ΔF1 |")
    L.append("|---|---|---|---|")
    for op in ["Robert", "Sobel", "Laplacian", "Canny"]:
        a = next(r for r in s["edge_detection"] if r["operator"] == op and r["image"] == "A(含噪)")
        b = next(r for r in s["edge_detection"] if r["operator"] == op and r["image"] == "B(中值滤波)")
        L.append(f"| {op} | {b['precision']-a['precision']:+.4f} | {b['recall']-a['recall']:+.4f} | "
                 f"{b['f1']-a['f1']:+.4f} |")
    L.append("")

    L.append("## 3. Hough 直线提取指标\n")
    L.append("| 边缘图来源 | 检出直线数 | 匹配真值数 | 线级 Precision | 线级 Recall |")
    L.append("|---|---|---|---|---|")
    for k, v in s["line_extraction"].items():
        L.append(f"| {k} | {v['n_detected']} | {v['n_matched']} | "
                 f"{v['line_precision']:.3f} | {v['line_recall']:.3f} |")
    L.append("")
    L.append(f"最终选用组合：**{s['final_combination']}**\n")
    L.append("逐条线的参数与像素级指标：\n")
    L.append("| 组合 | 真值线 | rho 偏差 | theta 偏差(°) | 边缘点数 | 段数 | Precision | Recall | F1 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, v in s["line_extraction"].items():
        for e in v["details"]:
            L.append(f"| {k} | {e['gt_id']} | {e['d_rho']:+.2f} | {e['d_theta_deg']:+.3f} | "
                     f"{e['n_points']} | {e['n_runs']} | {e['precision']:.3f} | {e['recall']:.3f} | {e['f1']:.3f} |")
    L.append("")

    L.append("## 4. 最终提取的两条电力线\n")
    for i, d in enumerate(s["final_lines"], 1):
        L.append(f"- 线 {i}：`{d['equation']}`，斜率角 {d['slope_angle_deg']:.2f}°，"
                 f"边缘点 {d['n_points']} 个，连通段 {d['n_runs']} 段")
        for si, (p0, p1) in enumerate(d["runs"], 1):
            L.append(f"    - 段 {si}: ({p0[0]},{p0[1]}) → ({p1[0]},{p1[1]})")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
