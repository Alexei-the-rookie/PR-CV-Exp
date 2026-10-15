"""
hough.py —— 自实现的 Hough 变换直线检测与线段提取

Hough 参数化：  rho = x * cos(theta) + y * sin(theta)
    - rho   : 原点到直线的垂距，范围 [-D, D]，D = sqrt(H^2 + W^2)
    - theta : 直线法线方向与 x 轴夹角，范围 [0, pi)

累加器 acc[ir, it] 统计落在该 (rho, theta) 直线上的边缘点数；
峰值即代表图像中的主力直线。随后沿直线在边缘图上回溯，得到
“线段边缘点的序列”以及线段的实际端点。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["HoughResult", "hough_accumulator", "find_peaks", "extract_segments",
           "line_from_points", "walk_line"]


@dataclass
class HoughResult:
    """单条直线（或直线上的一个线段）的检测结果。"""

    rho: float
    theta_deg: float
    votes: int
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    # 线段端点（可能多段，因线段被树木/建筑遮挡而断裂）
    runs: list[tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=list)

    @property
    def theta_rad(self) -> float:
        return np.deg2rad(self.theta_deg)

    @property
    def slope_angle_deg(self) -> float:
        """直线相对于 x 轴的方向角（度），-90~90。"""
        a = self.theta_deg - 90.0
        while a <= -90.0:
            a += 180.0
        while a > 90.0:
            a -= 180.0
        return a

    def y_at(self, x: np.ndarray | float):
        """由直线方程求给定 x 处的 y（要求 |sin(theta)| 不太小）。"""
        th = self.theta_rad
        return (self.rho - np.cos(th) * np.asarray(x, dtype=np.float64)) / np.sin(th)

    def x_at(self, y: np.ndarray | float):
        th = self.theta_rad
        return (self.rho - np.sin(th) * np.asarray(y, dtype=np.float64)) / np.cos(th)

    def equation(self) -> str:
        """返回形如 a*x + b*y = c 的直线方程字符串。"""
        a, b = np.cos(self.theta_rad), np.sin(self.theta_rad)
        return f"{a:.6f}*x + {b:.6f}*y = {self.rho:.4f}"

    def length(self) -> float:
        """所有 run 的总长度（像素）。"""
        return float(sum(np.hypot(e[0] - s[0], e[1] - s[1]) for s, e in self.runs))

    def span(self) -> float:
        """首末点之间的跨度（近似线段长度）。"""
        if self.points.size == 0:
            return 0.0
        pts = self.points
        d = np.hypot(pts[:, 0] - pts[:, 0].min(), pts[:, 1] - pts[:, 1].min())
        # 用参数方向的投影更准确
        th = self.theta_rad
        u = np.array([-np.sin(th), np.cos(th)])
        t = pts @ u
        return float(t.max() - t.min())


def hough_accumulator(
    edge: np.ndarray,
    rho_res: float = 1.0,
    theta_res_deg: float = 0.5,
):
    """构造 Hough 累加器。

    返回 (acc, rho_vals, theta_vals, ys, xs)
        acc        : (n_rho, n_theta) int32 累加器
        rho_vals   : 各行的 rho 值
        theta_vals : 各列的 theta 值（度）
        ys, xs     : 参与投票的边缘像素坐标
    """
    ys, xs = np.nonzero(edge)
    H, W = edge.shape
    D = float(np.hypot(H, W))
    n_rho = int(np.ceil(2 * D / rho_res)) + 1
    theta_vals = np.arange(0.0, 180.0, theta_res_deg)
    rho_vals = np.arange(-D, D + rho_res, rho_res)

    th = np.deg2rad(theta_vals)
    cos_t, sin_t = np.cos(th), np.sin(th)
    x = xs.astype(np.float64)[:, None]
    y = ys.astype(np.float64)[:, None]
    rho = x * cos_t + y * sin_t                     # (n_pix, n_theta)
    ir = np.round((rho + D) / rho_res).astype(np.int64)
    np.clip(ir, 0, len(rho_vals) - 1, out=ir)

    acc = np.zeros((len(rho_vals), len(theta_vals)), dtype=np.int32)
    for it in range(len(theta_vals)):
        acc[:, it] = np.bincount(ir[:, it], minlength=len(rho_vals))
    return acc, rho_vals, theta_vals, ys, xs


def find_peaks(
    acc: np.ndarray,
    rho_vals: np.ndarray,
    theta_vals: np.ndarray,
    threshold: int,
    nms_rho: int = 7,
    nms_theta: int = 7,
    max_peaks: int = 200,
):
    """在累加器中寻找局部极大值（带 rho/theta 方向非极大值抑制）。

    返回列表 [(votes, rho, theta_deg), ...]，按票数降序。
    """
    peaks: list[tuple[int, float, float]] = []
    a = acc.copy()
    order = np.argsort(a, axis=None)[::-1]
    taken = np.zeros_like(a, dtype=bool)
    for flat in order:
        if a.flat[flat] < threshold:
            break
        ir, it = np.unravel_index(flat, a.shape)
        if taken[ir, it]:
            continue
        peaks.append((int(a[ir, it]), float(rho_vals[ir]), float(theta_vals[it])))
        r0, r1 = max(0, ir - nms_rho), min(a.shape[0], ir + nms_rho + 1)
        t0, t1 = max(0, it - nms_theta), min(a.shape[1], it + nms_theta + 1)
        taken[r0:r1, t0:t1] = True
        if len(peaks) >= max_peaks:
            break
    return peaks


def line_from_points(points: np.ndarray):
    """对一组共线点做总体最小二乘（TLS/PCA）拟合，返回 (rho, theta_deg, rmse)。

    点集的主方向即直线方向，法线方向由主方向旋转 90 度得到。
    """
    pts = np.asarray(points, dtype=np.float64)
    mean = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - mean, full_matrices=False)
    direction = vt[0]                     # 主方向
    normal = np.array([-direction[1], direction[0]])
    if normal[0] < 0:                     # 规范化：保证法线朝 +x，theta 落在 [0,180)
        normal = -normal
    theta = np.arctan2(normal[1], normal[0]) % np.pi
    rho = float(mean @ normal)
    resid = (pts - mean) @ normal
    rmse = float(np.sqrt(np.mean(resid**2)))
    return rho, float(np.rad2deg(theta)), rmse


def _walk_line(
    edge: np.ndarray,
    rho: float,
    theta_deg: float,
    band: int = 2,
    gap: int = 12,
    min_len: int = 40,
):
    """沿给定直线在边缘图上回溯，收集支撑点并切分为若干连通段(run)。

    参数
    ----
    band    : 法向允许的像素偏差（容差）
    gap     : 允许的断裂长度（用于把被遮挡的线段接起来）
    min_len : 最短有效段长度
    """
    H, W = edge.shape
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    # 用沿直线的参数 t 表示：P(t) = rho*n + t*u
    n = np.array([c, s])
    u = np.array([-s, c])

    # 采样范围覆盖整幅图像（保证两端都能取到）
    corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], dtype=float)
    t_vals = (corners - rho * n) @ u
    t_lo, t_hi = t_vals.min() - 2, t_vals.max() + 2
    ts = np.arange(np.floor(t_lo), np.ceil(t_hi) + 1, 1.0)

    hits = []           # (t, x, y)
    for t in ts:
        best = None
        for d in range(-band, band + 1):
            p = rho * n + t * u + d * n
            xi, yi = int(round(p[0])), int(round(p[1]))
            if 0 <= xi < W and 0 <= yi < H and edge[yi, xi]:
                if best is None or abs(d) < abs(best[0]):
                    best = (d, xi, yi)
        if best is not None:
            hits.append((t, best[1], best[2]))
    if not hits:
        return [], []

    points = np.array([[x, y] for _, x, y in hits], dtype=int)
    t_arr = np.array([t for t, _, _ in hits])

    # 按 t 排序后切分为 runs（相邻 t 间隔超过 gap 则断开）
    runs = []
    start = 0
    for i in range(1, len(t_arr) + 1):
        if i == len(t_arr) or t_arr[i] - t_arr[i - 1] > gap:
            seg_t = t_arr[start:i]
            if seg_t.max() - seg_t.min() >= min_len:
                p0 = rho * n + seg_t.min() * u
                p1 = rho * n + seg_t.max() * u
                runs.append(
                    (
                        (int(round(p0[0])), int(round(p0[1]))),
                        (int(round(p1[0])), int(round(p1[1]))),
                    )
                )
            start = i
    return points, runs


# 对外公开的别名：沿给定直线回溯边缘点
walk_line = _walk_line


def extract_segments(
    edge: np.ndarray,
    peaks,
    band: int = 2,
    gap: int = 12,
    min_len: int = 40,
    refine: bool = True,
):
    """由 Hough 峰值出发，在边缘图上回溯得到线段与点序列。

    返回 list[HoughResult]，points 已按沿线参数排序。
    """
    results: list[HoughResult] = []
    for votes, rho, theta in peaks:
        pts, runs = _walk_line(edge, rho, theta, band=band, gap=gap, min_len=min_len)
        if pts.size == 0 or not runs:
            continue
        res = HoughResult(rho=float(rho), theta_deg=float(theta), votes=int(votes))
        res.points = pts
        res.runs = runs
        if refine and len(pts) >= 20:
            r2, t2, rmse = line_from_points(pts)
            # 仅在重拟合结果与原峰接近时采用，避免被离群点带偏
            if abs(r2 - rho) < 6 and abs(t2 - theta) < 3:
                res.rho, res.theta_deg = r2, t2
        results.append(res)
    return results
