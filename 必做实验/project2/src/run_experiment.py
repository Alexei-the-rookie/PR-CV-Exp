"""
run_experiment.py —— Project #2 主实验流程

对应实验指导书 Project #2 的四个步骤：
    1) Image1_withnoise.jpg -> 灰度 A
    2) 自编程序对 A 做中值滤波 -> B
    3) 对 A、B 分别用 Robert / Sobel / Canny / Laplacian 做边缘检测，
       比较优劣，并评价中值滤波能否提升线段边缘检测的 Precision / Recall
    4) 在边缘检测基础上用 Hough 变换完整提取两条电力线：
       给出直线方程参数、线段边缘点序列，并评价检测效果（P/R）

运行：
    python3 src/run_experiment.py
输出：
    results/figures/*.png    全部实验插图
    results/metrics/*.csv|json  全部定量指标
    results/REPORT_RESULTS.md   指标汇总（供撰写实验报告直接引用）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import imops  # noqa: E402
import hough as hg  # noqa: E402
import evaluate as ev  # noqa: E402
import groundtruth as gtmod  # noqa: E402
import viz  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
EXP_DIR = PROJ.parent                     # 必做实验/
RESULTS = PROJ / "results"
FIG = RESULTS / "figures"
MET = RESULTS / "metrics"

NOISY = EXP_DIR / "Image1_withnoise.jpg"
CLEAN = EXP_DIR / "Image1.jpg"
GT_JSON = PROJ / "data" / "ground_truth" / "power_lines.json"


# ==========================================================================
# 工具
# ==========================================================================
def read_image(path: Path) -> np.ndarray:
    """用 PIL 读图并返回 RGB uint8（IO 不参与算法实现）。"""
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def binary_by_otsu(response: np.ndarray, percentile_cap: float | None = None) -> np.ndarray:
    """对微分算子响应做归一化后，用大津法求阈值并二值化。

    参数 percentile_cap：若给定（如 99.0），先对响应做高分位裁剪，
    抑制个别极值对 Otsu 直方图的干扰。
    """
    r = response.astype(np.float32)
    if percentile_cap is not None:
        hi = np.percentile(r, percentile_cap)
        r = np.clip(r, 0, hi)
    u8 = imops.normalize_u8(r)
    t = imops.otsu_threshold(u8)
    return u8 >= t


def wire_profile(gray: np.ndarray, rho: float, theta_deg: float, half: int = 8):
    """沿候选直线求“平均法向灰度剖面”。

    返回 (offsets, profile)；profile[d] 为法向偏移 d 处的沿线平均灰度。
    """
    H, W = gray.shape
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    # 沿线采样参数 t 覆盖整幅图
    corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], dtype=float)
    u = np.array([-s, c])
    n = np.array([c, s])
    t_all = (corners - rho * n) @ u
    ts = np.arange(np.floor(t_all.min()), np.ceil(t_all.max()) + 1, 1.0)
    offsets = np.arange(-half, half + 1)
    prof = np.full(offsets.shape, np.nan, dtype=np.float64)
    for k, d in enumerate(offsets):
        p = rho * n + ts[:, None] * u[None, :] + d * n[None, :]
        xi = np.round(p[:, 0]).astype(int)
        yi = np.round(p[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        prof[k] = gray[yi[ok], xi[ok]].mean() if ok.any() else np.nan
    return offsets, prof


def wire_score(gray: np.ndarray, rho: float, theta_deg: float, half: int = 8) -> dict:
    """细暗线验证：返回对比度、半高全宽、以及是否像“细线”的判定。

    电力线在剖面上表现为**窄而深**的暗谷；
    地物明暗、道路边缘则表现为**宽而浅**或**阶跃**，可借此排除。
    """
    offs, prof = wire_profile(gray, rho, theta_deg, half=half)
    if np.isnan(prof).all():
        return {"contrast": 0.0, "fwhm": 99.0, "min_val": 0.0, "base": 0.0, "profile": prof, "offsets": offs}
    base = float(np.nanmean(prof[np.abs(offs) >= half - 2]))
    mn = float(np.nanmin(prof))
    depth = base - mn
    thr = base - depth / 2.0
    fwhm = int(np.nansum(prof <= thr))
    return {
        "contrast": depth,
        "fwhm": fwhm,
        "min_val": mn,
        "base": base,
        "profile": prof,
        "offsets": offs,
    }


def supporting_points(edge: np.ndarray, rho: float, theta_deg: float, band: int = 2) -> np.ndarray:
    """统计边缘图中落在该直线 ±band 像素内的边缘点（法向精确判据）。"""
    H, W = edge.shape
    ys, xs = np.nonzero(edge)
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    d = np.abs(xs * c + ys * s - rho)
    ok = d <= band
    return np.stack([xs[ok], ys[ok]], axis=1)


def group_peaks(peaks, dtheta: float = 2.0, drho: float = 8.0):
    """把 Hough 峰值按 (theta, rho) 近邻聚成“同一条直线”。

    同一条物理直线（尤其是宽度 >1px 的暗线）往往产生多个相邻峰值：
    其两条边缘对应两条平行线，间距约等于线宽，因此需要聚类合并。
    """
    clusters: list[dict] = []
    for votes, rho, theta in peaks:
        placed = False
        for cl in clusters:
            dth = abs(theta - cl["theta"])
            dth = min(dth, 180.0 - dth)
            if dth <= dtheta and abs(rho - cl["rho"]) <= drho:
                cl["members"].append((votes, rho, theta))
                # 以票数加权更新聚类中心
                w = np.array([m[0] for m in cl["members"]], dtype=float)
                cl["rho"] = float(np.average([m[1] for m in cl["members"]], weights=w))
                cl["theta"] = float(np.average([m[2] for m in cl["members"]], weights=w))
                cl["votes"] = int(w.sum())
                placed = True
                break
        if not placed:
            clusters.append(
                {"rho": float(rho), "theta": float(theta), "votes": int(votes), "members": [(votes, rho, theta)]}
            )
    clusters.sort(key=lambda c: -c["votes"])
    return clusters


# ==========================================================================
# 主流程
# ==========================================================================
def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    MET.mkdir(parents=True, exist_ok=True)
    viz.setup_font()

    gt_lines, gt_cfg = gtmod.load_gt(GT_JSON)
    print(f"[GT] 载入 {len(gt_lines)} 条真值直线")
    for ln in gt_lines:
        (xa, ya), (xb, yb) = ln.endpoints()
        print(f"  {ln.id}: {ln.equation_str()}   斜率角={ln.slope_angle_deg:.2f}°  "
              f"端点=({xa},{ya:.1f})-({xb},{yb:.1f})")

    # ---------------- 步骤 1：灰度化 ----------------
    rgb_clean = read_image(CLEAN)
    rgb_noisy = read_image(NOISY)
    A = imops.to_gray(rgb_noisy)          # 含噪灰度图 A
    A_ref = imops.to_gray(rgb_clean)      # 干净参考图（仅用于展示噪声）
    print(f"[步骤1] 灰度图 A: {A.shape}, 均值={A.mean():.1f}, 标准差={A.std():.1f}")

    # ---------------- 步骤 2：中值滤波 ----------------
    B3 = imops.median_filter(A, 3)
    B5 = imops.median_filter(A, 5)
    B = B3                                # 主对比使用 3x3（保边更好）
    print(f"[步骤2] 中值滤波 B(3x3): 均值={B.mean():.1f}, 标准差={B.std():.1f}")

    # ---------------- 步骤 3：四种算子边缘检测 ----------------
    def edge_maps(img: np.ndarray) -> dict:
        resp = {
            "Robert": imops.robert(img, mode="l1"),
            "Sobel": imops.sobel(img, mode="l2"),
            "Laplacian": imops.laplacian(img, neighbors=4),
            "Canny": None,   # Canny 直接输出二值图
        }
        binaries = {
            "Robert": binary_by_otsu(resp["Robert"], percentile_cap=99.5),
            "Sobel": binary_by_otsu(resp["Sobel"], percentile_cap=99.5),
            "Laplacian": binary_by_otsu(resp["Laplacian"], percentile_cap=99.5),
            "Canny": imops.canny(img, sigma=1.4, strong_frac=0.03, weak_frac=0.15).astype(bool),
        }
        return resp, binaries

    respA, binA = edge_maps(A)
    respB, binB = edge_maps(B)
    for k in binA:
        print(f"[步骤3] {k:9s}  A 边缘像素={int(binA[k].sum()):6d}   B 边缘像素={int(binB[k].sum()):6d}")

    # ---------------- 边缘检测定量评价（限制在电力线 ROI 内）----------------
    tol = float(gt_cfg.get("tolerance_px", 2.0))
    roi_w = float(gt_cfg.get("roi_halfwidth_px", 6.0))
    gt_band = gtmod.band_mask(gt_lines, shape=A.shape, halfwidth=tol)
    roi = gtmod.band_mask(gt_lines, shape=A.shape, halfwidth=roi_w)

    rows = []
    for name in ["Robert", "Sobel", "Laplacian", "Canny"]:
        for tag, bm in (("A(含噪)", binA), ("B(中值滤波)", binB)):
            pred = bm[name] & roi
            m = ev.pixel_pr(pred, gt_band, tol=tol)
            rows.append({"operator": name, "image": tag, **m})
    print("[步骤3] 电力线 ROI 内的边缘检测指标（tol=%.1fpx, ROI=±%.0fpx）" % (tol, roi_w))
    for r in rows:
        print("  %-9s %-11s P=%.3f R=%.3f F1=%.3f  (|Pred|=%d, |GT|=%d)" %
              (r["operator"], r["image"], r["precision"], r["recall"], r["f1"],
               r["n_pred"], r["n_gt"]))

    viz.write_csv(MET / "edge_detection_pr.csv", rows)

    # ---------------- 步骤 4：Hough 变换提取两条电力线 ----------------
    # 采用本项目自实现的 Hough 变换；这里同时给出四种算子边缘图上的提取结果，
    # 以体现“边缘检测质量直接影响线段提取质量”。
    line_report = {}
    all_detected = {}
    for name in ["Robert", "Sobel", "Laplacian", "Canny"]:
        for tag, bm in (("A", binA), ("B", binB)):
            edge = bm[name]
            acc, rho_vals, theta_vals, ys, xs = hg.hough_accumulator(edge, rho_res=1.0, theta_res_deg=1.0)
            peaks = hg.find_peaks(acc, rho_vals, theta_vals,
                                  threshold=max(25, int(0.25 * 511)),
                                  nms_rho=6, nms_theta=4, max_peaks=80)
            clusters = group_peaks(peaks, dtheta=2.0, drho=8.0)

            # 多线索验证：支撑点数 × 细暗线对比度，并用 FWHM 排除宽结构
            cands = []
            for cl in clusters[:25]:
                rho, theta = cl["rho"], cl["theta"]
                # TLS 精化：用支撑边缘点重新拟合
                pts = supporting_points(edge, rho, theta, band=2)
                if len(pts) >= 60:
                    r2, t2, rmse = hg.line_from_points(pts)
                    if abs(r2 - rho) < 6 and abs(t2 - theta) < 3:
                        rho, theta = r2, t2
                ws = wire_score(A, rho, theta)
                pts = supporting_points(edge, rho, theta, band=2)
                span = np.ptp(pts[:, 0]) if len(pts) else 0
                score = len(pts) * max(ws["contrast"], 0.0)
                cands.append(
                    {
                        "rho": float(rho), "theta_deg": float(theta),
                        "n_support": int(len(pts)), "span_x": int(span),
                        "contrast": ws["contrast"], "fwhm": int(ws["fwhm"]),
                        "score": float(score), "votes": cl["votes"],
                    }
                )
            # 只保留“细暗线”候选（FWHM<=6 且对比度>15 且支撑点足够）
            # 再施加**相对峰值准则**：电力线是图像中最显著的两条细暗直线，其得分
            # 应远高于地物纹理产生的次峰，故只保留 score >= 30% 最高分的候选。
            wire_like = [c for c in cands if c["fwhm"] <= 6 and c["contrast"] > 15 and c["n_support"] >= 150]
            wire_like.sort(key=lambda c: -c["score"])
            if wire_like:
                smax = wire_like[0]["score"]
                wire_like = [c for c in wire_like if c["score"] >= 0.30 * smax]
            top2 = wire_like[:2]

            # ---- 诚实统计：把所有通过"细暗线"验证的候选都记为检出，未匹配真值者算 FP ----
            det_all = []
            for c in wire_like:
                res = hg.HoughResult(rho=c["rho"], theta_deg=c["theta_deg"], votes=c["votes"])
                pts, runs = hg.walk_line(edge, c["rho"], c["theta_deg"], band=2, gap=25, min_len=30)
                res.points, res.runs = pts, runs
                det_all.append(res)

            # 最终交付结果：按 score 取前 2 条
            det = []
            for c in top2:
                res = hg.HoughResult(rho=c["rho"], theta_deg=c["theta_deg"], votes=c["votes"])
                pts, runs = hg.walk_line(edge, c["rho"], c["theta_deg"], band=2, gap=25, min_len=30)
                res.points, res.runs = pts, runs
                det.append(res)
            all_detected[(name, tag)] = det

            # 与真值配对并计算指标（用 det_all 统计，避免"只留 2 条"造成的指标虚高）
            gt_dict = [{"rho": ln.rho, "theta_deg": ln.theta_deg} for ln in gt_lines]
            pairs = ev.match_lines(det_all, gt_dict, dtheta_deg=3.0, drho=15.0)
            entries = []
            for di, gi in pairs:
                d = det_all[di]
                g = gt_lines[gi]
                pred_mask = np.zeros(A.shape, dtype=bool)
                if d.points.size:
                    pred_mask[d.points[:, 1], d.points[:, 0]] = True
                gtm = gtmod.centerline_mask(g, shape=A.shape, width=1)
                m = ev.pixel_pr(pred_mask, gtm, tol=tol)
                entries.append(
                    {
                        "gt_id": g.id,
                        "rho": d.rho, "theta_deg": d.theta_deg,
                        "d_rho": d.rho - g.rho,
                        "d_theta_deg": d.theta_deg - g.theta_deg,
                        "n_points": int(len(d.points)),
                        "n_runs": len(d.runs),
                        "equation": d.equation(),
                        **m,
                    }
                )
            # 未被匹到的检测线算 FP；未被检出的真值算 FN（线级别）
            line_report[f"{name}_{tag}"] = {
                "n_candidates": len(wire_like),
                "n_detected": len(det_all),
                "n_matched": len(pairs),
                "line_recall": len(pairs) / len(gt_lines),
                "line_precision": len(pairs) / max(len(det_all), 1),
                "details": entries,
                "top2_score": [round(c["score"], 1) for c in top2],
            }

    print("[步骤4] Hough 直线提取结果（与真值配对）")
    for k, v in line_report.items():
        print("  %-12s 检测=%d 匹配=%d 线级P=%.2f 线级R=%.2f" %
              (k, v["n_detected"], v["n_matched"], v["line_precision"], v["line_recall"]))

    # ---------------- 选定最优组合作为最终结果 ----------------
    # 由于 Project #2 的核心命题是"中值滤波能否提升线段提取效果"，
    # 最终结果在**中值滤波后(B)**的组合中选取最优，含噪输入(A)的结果作为对照一并报告。
    # 选取准则：以步骤 3 中电力线 ROI 内的边缘检测 F1 为准（该评价覆盖全部真值像素，
    # 比"是否检出两条线"这种二值指标更细粒度）。
    def _rank(k: str):
        v = line_report[k]
        f1 = np.mean([e["f1"] for e in v["details"]]) if v["details"] else 0.0
        return (v["line_recall"], v["line_precision"], f1)

    f1_on_B = {r["operator"]: r["f1"] for r in rows if r["image"] == "B(中值滤波)"}
    best_op = max(f1_on_B, key=lambda o: f1_on_B[o])
    best_key = f"{best_op}_B"
    print(f"[步骤4] 电力线 ROI 内边缘检测 F1（输入 B）：" +
          ", ".join(f"{o}={f1_on_B[o]:.4f}" for o in f1_on_B))
    print(f"[步骤4] 最终选用组合：{best_key}（B=中值滤波输入）")
    print("[步骤4] 各组合按线级指标排序：", sorted(line_report, key=_rank, reverse=True))
    final_det = all_detected[(best_key.split("_")[0], best_key.split("_")[1])]
    final_edge = binB[best_key.split("_")[0]] if best_key.endswith("_B") else binA[best_key.split("_")[0]]

    # 输出线段边缘点序列
    for i, d in enumerate(final_det, 1):
        arr = d.points
        viz.write_csv(MET / f"line{i}_edge_points.csv",
                      [{"order": j, "x": int(p[0]), "y": int(p[1])} for j, p in enumerate(arr)],
                      fieldnames=["order", "x", "y"])
    print(f"[步骤4] 已导出 {len(final_det)} 条线的边缘点序列到 results/metrics/")

    # ---------------- 出图 ----------------
    viz.fig01_gray_and_hist(FIG, A, A_ref, B3, B5)
    viz.fig02_gt_annotation(FIG, A, gt_lines, gt_cfg)
    viz.fig03_noise_median(FIG, A_ref, A, B3, B5)
    viz.fig04_operator_responses(FIG, A, respA)
    viz.fig05_binary_compare(FIG, A, binA, binB)
    viz.fig06_hough_accumulator(FIG, A, final_edge)
    viz.fig07_line_extraction(FIG, A, final_det, gt_lines)
    viz.fig08_pr_bars(FIG, rows)
    viz.fig09_zoom_line(FIG, A, B3, binA, binB, final_det, gt_lines)
    viz.fig10_wire_profiles(FIG, A, final_det)

    # ---------------- 汇总 ----------------
    summary = {
        "image": str(NOISY),
        "image_size": list(A.shape),
        "tolerance_px": tol,
        "roi_halfwidth_px": roi_w,
        "gt": [{"id": l.id, "rho": l.rho, "theta_deg": l.theta_deg,
                "slope_angle_deg": l.slope_angle_deg,
                "endpoints": [[float(v) for v in l.endpoints()[0]],
                              [float(v) for v in l.endpoints()[1]]],
                "equation": l.equation_str()} for l in gt_lines],
        "edge_detection": rows,
        "line_extraction": line_report,
        "final_combination": best_key,
        "final_lines": [
            {"rho": d.rho, "theta_deg": d.theta_deg, "slope_angle_deg": d.slope_angle_deg,
             "equation": d.equation(), "n_points": int(len(d.points)), "n_runs": len(d.runs),
             "runs": d.runs}
            for d in final_det
        ],
    }
    with open(MET / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    viz.write_report(RESULTS / "REPORT_RESULTS.md", summary)
    print("[完成] 图片 -> results/figures/, 指标 -> results/metrics/, 汇总 -> results/REPORT_RESULTS.md")


if __name__ == "__main__":
    main()
