"""
Task1 —— 图像拼接（image stitching）

对 Task1/ 下 4 组同一场景的左右视图（imgN_L / imgN_R）做拼接：
    1) SIFT 特征点检测与描述
    2) 暴力匹配 + Lowe 比值检验（ratio test）
    3) **自实现 RANSAC + DLT 单应矩阵估计**（含 Hartley 归一化）
    4) 以左图为参考帧做透视变换，画布自动union，双边羽化（feather）融合
    5) 输出拼接结果、匹配可视化与定量指标

运行：
    python3 src/stitch.py
输出：
    results/stitched/pairN.png        拼接结果
    results/figures/*.png             各组的匹配图与总览图
    results/metrics/stitch_metrics.csv 定量指标
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
TASK1_DIR = PROJ.parent / "Task1"                       # 选做实验/Task1
RES = PROJ / "results"
FIG = RES / "figures"
OUT = RES / "stitched"
MET = RES / "metrics"

RATIO = 0.75            # Lowe 比值检验阈值
RANSAC_THRESH = 3.0     # 内点重投影阈值（像素）
RANSAC_ITERS = 3000
RANSAC_CONF = 0.999


# ==========================================================================
# 几何：Hartley 归一化 + DLT + 自实现 RANSAC
# ==========================================================================
def _normalize(pts: np.ndarray):
    """Hartley 归一化：平移到质心、缩放到平均距离 sqrt(2)。

    归一化能显著改善 DLT 的数值条件数（原始像素坐标可到 1e3 量级，
    直接求解会使 A^T A 病态）。
    """
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]],
                  [0, s, -s * c[1]],
                  [0, 0, 1.0]])
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ T.T
    return ph[:, :2], T


def dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """直接线性变换（DLT）求单应矩阵 H，使 dst ≈ H · src。

    每对点提供 2 个方程；4 对点即可解（本实现要求 n>=4，用 SVD 取最小奇异向量）。
    """
    n = len(src)
    assert n >= 4, "DLT 至少需要 4 对点"
    sn, Ts = _normalize(src)
    dn, Td = _normalize(dst)
    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        x, y = sn[i]
        u, v = dn[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    _, _, vt = np.linalg.svd(A)
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def _symmetric_transfer_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """对称转移误差（正向 + 反向重投影误差），对 H 的估计误差更敏感。"""
    def proj(M, p):
        ph = np.hstack([p, np.ones((len(p), 1))]) @ M.T
        return ph[:, :2] / ph[:, 2:3]
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(src), np.inf)
    e1 = np.linalg.norm(proj(H, src) - dst, axis=1)
    e2 = np.linalg.norm(proj(Hi, dst) - src, axis=1)
    return np.sqrt((e1 ** 2 + e2 ** 2) / 2.0)


def ransac_homography(src: np.ndarray, dst: np.ndarray,
                      thresh: float = 3.0, iters: int = 3000,
                      conf: float = 0.999, seed: int = 0):
    """自实现 RANSAC 单应矩阵估计。

    采用**自适应迭代次数**：一旦内点率足够高，提前停止（标准做法）。
    返回 (H, inlier_mask, 实际迭代次数)。
    """
    rng = np.random.default_rng(seed)
    n = len(src)
    best_mask = np.zeros(n, dtype=bool)
    best_H = None
    best_cnt = 0
    it = 0
    while it < iters:
        idx = rng.choice(n, 4, replace=False)
        try:
            H = dlt_homography(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            it += 1
            continue
        if not np.all(np.isfinite(H)):
            it += 1
            continue
        err = _symmetric_transfer_error(H, src, dst)
        mask = err < thresh
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt, best_mask, best_H = cnt, mask, H
            # 自适应迭代次数
            w = cnt / n
            if w <= 0 or w >= 1:
                need = iters
            else:
                denom = np.log(max(1e-12, 1 - w ** 4))
                need = 0 if denom == 0 else np.log(1 - conf) / denom
            iters = min(iters, int(np.ceil(need)))
        it += 1
    if best_H is None:
        return None, best_mask, it
    return best_H, best_mask, it


# ==========================================================================
# 拼接：画布计算 + 羽化融合
# ==========================================================================
def compute_canvas(H: np.ndarray, shape_l, shape_r):
    """计算能容纳左图与变换后右图的输出画布（尺寸 + 平移矩阵）。"""
    hl, wl = shape_l
    hr, wr = shape_r
    corners_r = np.array([[0, 0], [wr - 1, 0], [wr - 1, hr - 1], [0, hr - 1]], dtype=np.float64)
    ph = np.hstack([corners_r, np.ones((4, 1))]) @ H.T
    proj = ph[:, :2] / ph[:, 2:3]
    all_pts = np.vstack([[[0, 0], [wl - 1, 0], [wl - 1, hl - 1], [0, hl - 1]], proj])
    xmin, ymin = np.floor(all_pts.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(all_pts.max(axis=0)).astype(int)
    W, Hh = int(xmax - xmin + 1), int(ymax - ymin + 1)
    T = np.array([[1.0, 0, -xmin], [0, 1.0, -ymin], [0, 0, 1.0]])
    return (Hh, W), T, (xmin, ymin)


def stitch_pair(img_l: np.ndarray, img_r: np.ndarray):
    """拼接一对图像，返回 (结果图, 诊断信息 dict)。"""
    t0 = time.perf_counter()
    g_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    g_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # ---- 1) SIFT 特征 ----
    sift = cv2.SIFT_create(nfeatures=4000)
    kp_l, des_l = sift.detectAndCompute(g_l, None)
    kp_r, des_r = sift.detectAndCompute(g_r, None)

    # ---- 2) 匹配 + 比值检验 ----
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des_r, des_l, k=2)          # 右 -> 左（右图投影到左图坐标）
    good = [m for m, n in knn if m.distance < RATIO * n.distance]

    src = np.float32([kp_r[m.queryIdx].pt for m in good])
    dst = np.float32([kp_l[m.trainIdx].pt for m in good])

    # ---- 3) RANSAC 单应 ----
    H, mask, iters_used = ransac_homography(src, dst, RANSAC_THRESH, RANSAC_ITERS, RANSAC_CONF)
    if H is None:
        raise RuntimeError("RANSAC 未能估计出单应矩阵")
    inl = mask.sum()
    err = _symmetric_transfer_error(H, src[mask], dst[mask])
    rmse = float(np.sqrt((err ** 2).mean()))

    # ---- 4) 画布与透视变换 ----
    (ch, cw), T, origin = compute_canvas(H, g_l.shape, g_r.shape)
    H_full = T @ H
    warped_r = cv2.warpPerspective(img_r, H_full, (cw, ch),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_l = cv2.warpPerspective(img_l, T, (cw, ch),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # 有效区掩膜（warp 后非零区域；用 ones 变换得到，避免暗像素被误判为空洞）
    ones_l = cv2.warpPerspective(np.ones_like(g_l, np.uint8) * 255, T, (cw, ch))
    ones_r = cv2.warpPerspective(np.ones_like(g_r, np.uint8) * 255, H_full, (cw, ch))

    # ---- 5) 羽化融合 ----
    valid_l = (ones_l > 0).astype(np.uint8)
    valid_r = (ones_r > 0).astype(np.uint8)
    # 用距离变换得到每个像素"到本图边界的距离"，作为融合权重：
    # 越靠图像中心权重越大，重叠区按双方距离加权平均，从而消除拼接缝
    w_l = cv2.distanceTransform(valid_l, cv2.DIST_L2, 5)
    w_r = cv2.distanceTransform(valid_r, cv2.DIST_L2, 5)
    w_l = w_l / max(w_l.max(), 1e-9)
    w_r = w_r / max(w_r.max(), 1e-9)

    both = (valid_l > 0) & (valid_r > 0)
    alpha_l = np.ones((ch, cw), np.float32)
    alpha_r = np.ones((ch, cw), np.float32)
    if both.any():
        sw = np.maximum(w_l[both] + w_r[both], 1e-9)
        alpha_l[both] = w_l[both] / sw
        alpha_r[both] = w_r[both] / sw
    alpha_l[valid_l == 0] = 0.0      # 本图无有效像素的区域权重置 0
    alpha_r[valid_r == 0] = 0.0

    out = (warped_l.astype(np.float32) * alpha_l[..., None] +
           warped_r.astype(np.float32) * alpha_r[..., None])
    out = np.clip(out, 0, 255).astype(np.uint8)

    # 裁掉四周全黑边
    filled = ((valid_l > 0) | (valid_r > 0)).astype(np.uint8)
    ys, xs = np.nonzero(filled)
    if len(xs):
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        out = out[y0:y1 + 1, x0:x1 + 1]

    dt = time.perf_counter() - t0
    info = {
        "kp_L": len(kp_l), "kp_R": len(kp_r),
        "good_matches": len(good),
        "inliers": int(inl),
        "inlier_ratio": float(inl / max(len(good), 1)),
        "ransac_iters": iters_used,
        "rmse_px": rmse,
        "out_h": out.shape[0], "out_w": out.shape[1],
        "seconds": dt,
        "H": H,
        "src": src, "dst": dst, "mask": mask,
        "kp_l": kp_l, "kp_r": kp_r, "good": good,
        "warped": warped_r, "alpha_overlap": both,
    }
    return out, info


# ==========================================================================
def draw_matches(img_l, img_r, kp_l, kp_r, good, mask, path):
    """绘制内点匹配（绿）与外点匹配（红）。"""
    inlier = [good[i] for i in range(len(good)) if mask[i]]
    outlier = [good[i] for i in range(len(good)) if not mask[i]]
    vis = cv2.drawMatches(img_r, kp_r, img_l, kp_l, inlier, None,
                          matchColor=(0, 220, 0), singlePointColor=None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if outlier:
        vis = cv2.drawMatches(img_r, kp_r, img_l, kp_l, outlier, vis,
                              matchColor=(0, 0, 220), singlePointColor=None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # 缩放到合理尺寸
    scale = min(1.0, 1800 / vis.shape[1])
    if scale < 1:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), vis)


def main():
    for d in (FIG, OUT, MET):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    panels = []
    for i in range(1, 5):
        pl = TASK1_DIR / f"img{i}_L.jpg"
        pr = TASK1_DIR / f"img{i}_R.jpg"
        if not pl.exists() or not pr.exists():
            print(f"[跳过] {pl} / {pr} 不存在")
            continue
        img_l = cv2.imread(str(pl))
        img_r = cv2.imread(str(pr))
        out, info = stitch_pair(img_l, img_r)
        cv2.imwrite(str(OUT / f"pair{i}.png"), out)
        draw_matches(img_l, img_r, info["kp_l"], info["kp_r"],
                     info["good"], info["mask"], FIG / f"task1_pair{i}_matches.png")
        print(f"[pair{i}] {img_l.shape[1]}x{img_l.shape[0]} + {img_r.shape[1]}x{img_r.shape[0]}"
              f"  ->  {out.shape[1]}x{out.shape[0]}")
        print(f"    SIFT: L={info['kp_L']}, R={info['kp_R']}; 匹配={info['good_matches']}; "
              f"内点={info['inliers']} ({info['inlier_ratio']*100:.1f}%); "
              f"RMSE={info['rmse_px']:.2f}px; RANSAC迭代={info['ransac_iters']}; "
              f"{info['seconds']*1000:.0f}ms")
        rows.append({
            "pair": f"img{i}",
            "size_L": f"{img_l.shape[1]}x{img_l.shape[0]}",
            "size_R": f"{img_r.shape[1]}x{img_r.shape[0]}",
            "kp_L": info["kp_L"], "kp_R": info["kp_R"],
            "good_matches": info["good_matches"],
            "inliers": info["inliers"],
            "inlier_ratio": round(info["inlier_ratio"], 4),
            "ransac_iters": info["ransac_iters"],
            "rmse_px": round(info["rmse_px"], 3),
            "out_size": f"{info['out_w']}x{info['out_h']}",
            "ms": round(info["seconds"] * 1000),
        })
        panels.append((f"img{i}_L", img_l, f"img{i}_R", img_r, out))

    with open(MET / "stitch_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 总览图：每组一行（L / R / 拼接）
    if panels:
        maxw = 700
        tiles = []
        for name_l, il, name_r, ir, out in panels:
            def fit(img, label):
                s = maxw / img.shape[1]
                t = cv2.resize(img, (maxw, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
                bar = np.zeros((28, maxw, 3), np.uint8)
                cv2.putText(bar, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                return np.vstack([bar, t])
            row = [fit(il, name_l), fit(ir, name_r), fit(out, "stitched")]
            hmax = max(r.shape[0] for r in row)
            row = [np.vstack([r, np.zeros((hmax - r.shape[0], r.shape[1], 3), np.uint8)]) for r in row]
            sep = np.full((hmax, 6, 3), 60, np.uint8)
            tiles.append(np.hstack([row[0], sep, row[1], sep, row[2]]))
        W = max(t.shape[1] for t in tiles)
        tiles = [np.hstack([t, np.zeros((t.shape[0], W - t.shape[1], 3), np.uint8)]) for t in tiles]
        cv2.imwrite(str(FIG / "task1_overview.png"), np.vstack(tiles))

    print(f"\n完成：拼接结果 -> {OUT}")
    print(f"      匹配图/总览 -> {FIG}")
    print(f"      指标 -> {MET/'stitch_metrics.csv'}")


if __name__ == "__main__":
    main()
