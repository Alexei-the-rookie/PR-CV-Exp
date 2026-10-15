"""
evaluate.py —— 边缘检测与线段提取的定量评价（Precision / Recall / F1）

评价采用“带容差的双向匹配”，这是边缘/边界评价中的标准做法
（如 BSDS500 的边界基准、以及结构光/线检测文献中的常用定义）：

    TP = 预测边缘中，与真值边缘的距离 <= tol 的像素数
    FP = 预测边缘中，与真值边缘的距离 >  tol 的像素数
    FN = 真值边缘中，与预测边缘的距离 >  tol 的像素数

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2PR / (P + R)

仅用单向匹配（只检查预测是否落在真值附近）会高估性能：一条
“预测了整条直线、但真值只有一小段”的结果 Precision 会虚高，
因此这里同时统计 FN。
"""

from __future__ import annotations

import numpy as np

try:  # scipy 用于快速距离变换
    from scipy.ndimage import distance_transform_edt

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

__all__ = ["pixel_pr", "pr_from_distances", "match_lines", "line_pr"]


def _distance_to_mask(mask: np.ndarray) -> np.ndarray:
    """返回到最近 True 像素的欧氏距离图（True 处为 0）。"""
    if mask.sum() == 0:
        return np.full(mask.shape, np.inf, dtype=np.float64)
    if _HAS_SCIPY:
        return distance_transform_edt(~mask)
    # 无 scipy 时的兜底：对前景坐标做广播距离计算
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.empty((H, W), dtype=np.float64)
    chunk = 64  # 分块以控制内存
    for y0 in range(0, H, chunk):
        sl = slice(y0, min(y0 + chunk, H))
        dy = (yy[sl][:, :, None] - ys[None, None, :]) ** 2      # (h, W, n)
        dx = (xx[sl][:, :, None] - xs[None, None, :]) ** 2      # (h, W, n)
        d[sl] = np.sqrt(dy + dx).min(axis=-1)
    return d


def pixel_pr(pred: np.ndarray, gt: np.ndarray, tol: float = 2.0) -> dict:
    """带容差的像素级 P/R/F1（容差区域匹配，边缘检测评价的标准做法）。

    定义（tol 为匹配容差）::

        P = |{p ∈ Pred : d(p, GT)  ≤ tol}| / |Pred|
        R = |{q ∈ GT   : d(q, Pred) ≤ tol}| / |GT|
        F1 = 2PR/(P+R)

    即：**预测像素**只要落在真值的 tol 邻域内就算命中（计入 P 的分子）；
    **真值像素**只要被任一预测像素的 tol 邻域覆盖就算被检出（计入 R 的分子）。

    两点说明：
    1) 允许"多对一"匹配是有意的：真值以 5px 宽带（真值线 ±tol）给出，代表
       "电力线在该带内"，任何落在带内的预测都正确。若强行一对一匹配，
       一条完全正确的 1px 中心线反而只有 ~1/5 的 Recall。
    2) 之所以同时统计两个方向的命中数（tp_pred / tp_gt），是因为容差匹配下
       它们一般不相等；分别用于 P 与 R 才各自自洽，报告中也分别列出。
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    n_pred, n_gt = int(pred.sum()), int(gt.sum())
    d_gt = _distance_to_mask(gt)      # 每个像素到最近真值像素的距离
    d_pred = _distance_to_mask(pred)  # 每个像素到最近预测像素的距离

    tp_pred = pred & (d_gt <= tol)    # 命中的预测像素
    tp_gt = gt & (d_pred <= tol)      # 被覆盖的真值像素
    fp = pred & (d_gt > tol)
    fn = gt & (d_pred > tol)

    precision = float(tp_pred.sum()) / n_pred if n_pred else 0.0
    recall = float(tp_gt.sum()) / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_pred": int(tp_pred.sum()),
        "tp_gt": int(tp_gt.sum()),
        "fp": int(fp.sum()),
        "fn": int(fn.sum()),
        "n_pred": n_pred,
        "n_gt": n_gt,
    }


def pr_from_distances(pred, gt, tol: float = 2.0) -> dict:
    """pixel_pr 的别名（保持接口语义清晰）。"""
    return pixel_pr(pred, gt, tol=tol)


def match_lines(detected, gt_lines, dtheta_deg: float = 2.0, drho: float = 8.0):
    """把检测到的直线与真值直线按 (theta, rho) 近邻配对（贪心、一对一）。

    参数
    ----
    detected : list[HoughResult]
    gt_lines : list[dict]，含 'rho' 与 'theta_deg'
    返回     : list[(det_idx, gt_idx)]
    """
    pairs = []
    used_gt = set()
    for i, d in enumerate(detected):
        best, bestcost = None, np.inf
        for j, g in enumerate(gt_lines):
            if j in used_gt:
                continue
            dth = abs(d.theta_deg - g["theta_deg"])
            dth = min(dth, 180.0 - dth)
            dr = abs(d.rho - g["rho"])
            if dth <= dtheta_deg and dr <= drho:
                cost = dth / dtheta_deg + dr / drho
                if cost < bestcost:
                    best, bestcost = j, cost
        if best is not None:
            used_gt.add(best)
            pairs.append((i, best))
    return pairs


def line_pr(pred_points: np.ndarray, gt_points: np.ndarray, tol: float = 2.0) -> dict:
    """对“线段边缘点序列”做像素级 P/R（把点集当作 2 值掩码处理）。"""
    return pixel_pr(pred_points, gt_points, tol=tol)
