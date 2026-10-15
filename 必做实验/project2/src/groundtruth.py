"""
groundtruth.py —— 电力线真值的读取、栅格化与 ROI 生成

真值以直线参数 (rho, theta) 形式保存在 data/ground_truth/power_lines.json 中，
本模块负责把线参数转换为：
    - 线段边缘点序列（沿线的有序像素坐标）
    - 中心线掩码（1px 宽）
    - 电力线“本体质”带（±W 像素，用于边缘检测评价的真值）
    - ROI 掩码（±R 像素，评价时把预测边缘限制在该带内，避免被全图其它纹理淹没）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["load_gt", "LineGT", "centerline_points", "centerline_mask", "band_mask"]


class LineGT:
    """单条真值直线。"""

    def __init__(self, d: dict):
        self.id: str = d["id"]
        self.rho: float = float(d["rho"])
        self.theta_deg: float = float(d["theta_deg"])
        self.x_start: int = int(d.get("x_start", 0))
        self.x_end: int = int(d.get("x_end", 511))
        self.note: str = d.get("note", "")

    # ---- 基本几何 ----
    @property
    def theta_rad(self) -> float:
        return np.deg2rad(self.theta_deg)

    @property
    def slope_angle_deg(self) -> float:
        return self.theta_deg - 90.0

    def y_at(self, x):
        th = self.theta_rad
        return (self.rho - np.cos(th) * np.asarray(x, dtype=np.float64)) / np.sin(th)

    def x_at(self, y):
        th = self.theta_rad
        return (self.rho - np.sin(th) * np.asarray(y, dtype=np.float64)) / np.cos(th)

    def equation_str(self) -> str:
        a, b = np.cos(self.theta_rad), np.sin(self.theta_rad)
        return f"{a:.6f}*x + {b:.6f}*y = {self.rho:.4f}"

    def endpoints(self):
        """返回裁剪到图像内的线段两端点 (x0,y0),(x1,y1)。"""
        x0, x1 = self.x_start, self.x_end
        return (x0, float(self.y_at(x0))), (x1, float(self.y_at(x1)))

    def centroid(self):
        (xa, ya), (xb, yb) = self.endpoints()
        return ((xa + xb) / 2.0, (ya + yb) / 2.0)


def load_gt(path: str | Path) -> tuple[list[LineGT], dict]:
    """读取真值 JSON，返回 (直线列表, 元信息)。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    lines = [LineGT(d) for d in cfg["lines"]]
    return lines, cfg


def centerline_points(line: LineGT, shape=(512, 512)) -> np.ndarray:
    """把直线栅格化为有序的像素点序列（按 x 递增）。

    返回 (N, 2) int 数组，列为 (x, y)。
    """
    H, W = shape
    xs = np.arange(line.x_start, line.x_end + 1)
    ys = np.round(line.y_at(xs)).astype(int)
    ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    return np.stack([xs[ok], ys[ok]], axis=1)


def centerline_mask(line: LineGT, shape=(512, 512), width: int = 1) -> np.ndarray:
    """中心线掩码；width=1 表示 1 像素宽，width=3 表示 3 像素宽。"""
    H, W = shape
    m = np.zeros((H, W), dtype=bool)
    pts = centerline_points(line, shape)
    r = width // 2
    for x, y in pts:
        m[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1] = True
    return m


def band_mask(
    lines: list[LineGT],
    shape=(512, 512),
    halfwidth: float = 2.0,
) -> np.ndarray:
    """到任意一条真值中心线的距离 <= halfwidth 的像素集合。

    用逐像素法向距离计算（比形态学膨胀更精确，避免棋盘距离误差）。
    """
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.zeros((H, W), dtype=bool)
    for ln in lines:
        th = ln.theta_rad
        c, s = np.cos(th), np.sin(th)
        d = np.abs(xx * c + yy * s - ln.rho)          # 到该直线的法向距离
        xs = xx * (-s) + yy * c                        # 沿线方向的投影（判端点）
        ok_x = (xx >= ln.x_start) & (xx <= ln.x_end)
        out |= (d <= halfwidth) & ok_x
        del xs, d
    return out
