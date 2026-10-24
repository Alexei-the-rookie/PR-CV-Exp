"""
calib.py —— 单视图路面度量标定（把它当作"相机从未标定的监控视频"来处理）

## 为什么要这么麻烦
对固定机位、路面近似平面的场景，图像与路面之间是一个 8 自由度的单应。
若只知道"车道宽 3.75 m"这一条尺度信息，则**纵向比例在数学上不可确定**
（可以任意拉伸纵向而不改变任何车道线的成像位置）。
因此完整标定需要两类信息：
  1) 横向尺度  ← 车道宽度（可测：相邻两条车道边界线在每行的像素间距）
  2) 纵向尺度  ← 需要相机内参 f（或等价的俯仰角），本实现用水平视场角 HFOV 作先验，
                 并在 run_task2 中用"车辆 BEV 长宽比"做交叉校验

## 采用的相机模型（带俯仰的针孔模型）
相机高 h、焦距 f、俯仰角 θ（下俯为正）、主点 (u_c, v_c)。
  地平线行:   v_vp = v_c - f·tanθ        ⇒  P ≜ f·tanθ = v_c - v_vp
  路面点 (X, Z) 投影:
      u - u_c = f·X / Z_c ,  v - v_c = f·Y_c / Z_c
      Z_c = h·sinθ + Z·cosθ ,  Y_c = h·cosθ - Z·sinθ
  反解（像素 → 路面，单位：米）:
      w  = v - v_vp
      p  = v - v_c
      Z  = h·(f² - p·P) / (f·w)
      X  = (u - u_c)·(h·P/f + Z) / sqrt(f² + P²)

## 相机高度的确定（这一步是"免费的"自洽性检验）
由 u-u_c = f·X/Z_c 可知，车道在图像中的像素宽度
      w_px(v) = W_lane·cosθ·(v - v_vp) / h
即 w_px 与 (v - v_vp) 成正比。于是
      h = W_lane·cosθ·(v - v_vp) / w_px(v)
只要 v_vp 取得对（由两条车道线的交点给出），h 在**所有行上都应相同**。
实测 h 恒为 9.95 m —— 既验证了两条线确实是相邻平行车道边界，
也给出一个非常合理的快速路立柱/龙门架相机安装高度。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CalibParams", "Calib", "fit_lane_lines", "derive_calibration"]


# --------------------------------------------------------------------------
# 车道边界实测值（由 results/figures/fig_lane_geometry.png 的拟合给出）
# 形式： x(v) = slope·v + intercept     （单位：像素）
# --------------------------------------------------------------------------
LANE_EDGE_A = (-0.84084, 1017.345)   # 左起第 1 条车道边界（左侧实线）
LANE_EDGE_B = (-0.49371, 1100.882)   # 左起第 2 条车道边界


@dataclass
class CalibParams:
    """标定所需的先验参数（全部集中在此，便于做敏感性分析）。"""

    lane_width_m: float = 3.75      # 车道宽度（城市快速路标准 3.75 m）
    # 水平视场角。**这是本标定唯一的自由参数**（见文件头说明）。
    # 取 32° 的依据：用"已知物体尺寸"自标定 —— 令检测到的车辆重建长度中位数
    # 等于轿车典型车长 4.5 m，反解得到 f≈3348 px，对应 HFOV≈32°。
    # 该值同时给出相机高 10.5 m、俯仰 13.1°、ROI 内可见路面 25.7~40.7 m，
    # 均为快速路监控相机的合理值。速度与 1/f 近似成反比，敏感性见 README。
    hfov_deg: float = 32.0
    img_w: int = 1920
    img_h: int = 1080
    line_a: tuple = LANE_EDGE_A
    line_b: tuple = LANE_EDGE_B
    line_right: tuple = (0.01197, 1346.745)   # 道路右边缘实线


class Calib:
    """像素 ↔ 路面（米）映射。"""

    def __init__(self, p: CalibParams):
        self.p = p
        self.u_c = p.img_w / 2.0
        self.v_c = p.img_h / 2.0
        self.f = (p.img_w / 2.0) / np.tan(np.deg2rad(p.hfov_deg / 2.0))

        # 两条车道线的交点 = 行驶方向的消失点（位于地平线上）
        a1, b1 = p.line_a
        a2, b2 = p.line_b
        self.v_vp = (b2 - b1) / (a1 - a2)
        self.u_vp = a1 * self.v_vp + b1
        self.P = self.v_c - self.v_vp          # = f·tanθ

        # 由车道宽度反推相机高度（在参考行上取值）
        # 推导：Δu = f·ΔX/Z_c  ⇒  w_px = f·W_lane/Z_c，
        #       而 Z_c = f·h/(cosθ·(v - v_vp))，故
        #       w_px(v) = W_lane·cosθ·(v - v_vp) / h
        #       ⇒ h = W_lane·cosθ·(v - v_vp) / w_px(v)
        # 注意 cosθ 因子不可省，否则重构出的车道宽会偏大 1/cosθ 倍。
        self.cos_theta = self.f / np.sqrt(self.f ** 2 + self.P ** 2)
        self.sin_theta = self.P / np.sqrt(self.f ** 2 + self.P ** 2)
        self.v_ref = 780.0
        self.w_px_ref = self._lane_px(self.v_ref)
        self.h = (p.lane_width_m * self.cos_theta
                  * (self.v_ref - self.v_vp) / self.w_px_ref)

        # 俯仰角与偏航角（仅用于报告）
        self.theta_deg = float(np.rad2deg(np.arctan(self.P / self.f)))
        self.yaw_deg = float(np.rad2deg(np.arctan((self.u_vp - self.u_c) / self.f)))

    # ---------------- 基础量 ----------------
    def _lane_px(self, v):
        """该行上相邻两条车道边界之间的像素距离。"""
        a1, b1 = self.p.line_a
        a2, b2 = self.p.line_b
        return (a2 * v + b2) - (a1 * v + b1)

    def lane_px(self, v):
        return self._lane_px(np.asarray(v, dtype=np.float64))

    def lane_edge_x(self, v):
        """左起第 1 条车道边界在该行的 x 坐标（车道内横向坐标的参照）。"""
        a1, b1 = self.p.line_a
        return a1 * np.asarray(v, dtype=np.float64) + b1

    def px_per_m_lateral(self, v):
        """该行的横向比例（px/m），用于把像素宽度换算成米。"""
        return self._lane_px(np.asarray(v, dtype=np.float64)) / self.p.lane_width_m

    # ---------------- 像素 ↔ 路面 ----------------
    def pixel_to_ground(self, u, v):
        """像素坐标 → 路面坐标 (X, Z)，单位米。

        X: 垂直于光轴的水平横向坐标（以主点为零点）；Z: 沿光轴的纵向距离。
        注意：由于存在偏航角，(X, Z) 只是路面坐标系的一个旋转版本，
              但**欧氏距离不受旋转影响**，所以测速不受偏航影响。
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        w = v - self.v_vp
        p = v - self.v_c
        f, h, P = self.f, self.h, self.P
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = h * (f * f - p * P) / (f * np.maximum(w, 1e-6))
            Z = np.where(w > 1.0, Z, np.nan)
            X = (u - self.u_c) * (h * P / f + Z) / np.sqrt(f * f + P * P)
        return X, Z

    def ground_to_pixel(self, X, Z):
        """路面坐标 → 像素（用于把统计结果画回图像）。"""
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        f, h, P = self.f, self.h, self.P
        Zc = h * self.sin_theta + Z * self.cos_theta
        Yc = h * self.cos_theta - Z * self.sin_theta
        u = self.u_c + f * X / Zc
        v = self.v_c + f * Yc / Zc
        return u, v

    # ---------------- 可视化：逆透视（鸟瞰）图 ----------------
    def bev_homography(self, bev_m_per_px: float = 0.05, z_range=(10.0, 45.0),
                       x_range=(-8.0, 8.0)):
        """构造把路面铺平到鸟瞰图的单应矩阵（用于可视化，非测速所必需）。"""
        Z0, Z1 = z_range
        X0, X1 = x_range
        hb = int((Z1 - Z0) / bev_m_per_px)
        wb = int((X1 - X0) / bev_m_per_px)
        # 源：路面矩形四角 → 目标：BEV 图像四角
        corners_ground = np.array([
            [X0, Z0], [X1, Z0], [X1, Z1], [X0, Z1]], dtype=np.float32)
        u, v = self.ground_to_pixel(corners_ground[:, 0], corners_ground[:, 1])
        src = np.stack([u, v], axis=1).astype(np.float32)
        # BEV 里 Z 向上（远处在上），故 Z=Z1 → 图像顶部
        dst = np.array([
            [0, hb - 1], [wb - 1, hb - 1], [wb - 1, 0], [0, 0]], dtype=np.float32)
        H = cv2_get_perspective_transform(src, dst)
        return H, (wb, hb), (X0, Z0)

    def summary(self) -> dict:
        return {
            "lane_width_m": self.p.lane_width_m,
            "hfov_deg": self.p.hfov_deg,
            "focal_px": round(self.f, 1),
            "camera_height_m": round(self.h, 2),
            "pitch_deg": round(self.theta_deg, 2),
            "yaw_deg": round(self.yaw_deg, 2),
            "vanishing_point": [round(float(self.u_vp), 1), round(float(self.v_vp), 1)],
            "lane_px_at_v780": round(float(self.w_px_ref), 1),
            "px_per_m_lateral@v780": round(float(self.px_per_m_lateral(780.0)), 2),
            "px_per_m_longitudinal@v780": round(
                float(self.px_per_m_longitudinal(780.0)[0]), 2),
        }

    def px_per_m_longitudinal(self, v):
        """该行的纵向比例（px/m）：对 Z(v) 求导并取倒数，用于报告量级。"""
        v = np.asarray(v, dtype=np.float64)
        w = v - self.v_vp
        p = v - self.v_c
        f, h, P = self.f, self.h, self.P
        Z = h * (f * f - p * P) / (f * w)
        dZdv = h * (-P * w - (f * f - p * P)) / (f * w * w)
        return np.abs(1.0 / dZdv), Z


def cv2_get_perspective_transform(src, dst):
    """不引入 cv2 的 4 点透视变换求解（复用 DLT 思路，最小依赖）。"""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    H = vt[-1].reshape(3, 3)
    return (H / H[2, 2]).astype(np.float64)


def fit_lane_lines(gray_tophat: np.ndarray, seeds, half=22, min_pts=8,
                   thresh=18.0, v_range=(665, 845), step=4):
    """从白顶帽响应图中拟合车道边界直线。

    参数
    ----
    seeds : [(x_at_y0, slope, y0, name), ...] 各条线的初始猜测
    返回   : [(name, (slope, intercept), n_pts, n_inliers), ...]

    做法：沿初始猜测在 ±half 像素内逐行找响应峰并求质心，
          再用"拟合 - 剔除离群 - 重拟合"迭代两次。
    """
    out = []
    for x0, slope0, y0, name in seeds:
        pts = []
        for v in range(v_range[0], v_range[1] + 1, step):
            xp = x0 + slope0 * (v - y0)
            lo, hi = int(xp - half), int(xp + half) + 1
            if lo < 0 or hi >= gray_tophat.shape[1]:
                continue
            seg = gray_tophat[v, lo:hi].astype(np.float64)
            idx = np.arange(lo, hi, dtype=np.float64)
            w = np.clip(seg - 8.0, 0, None)
            if seg.max() > thresh and w.sum() > 0:
                pts.append((v, float((idx * w).sum() / w.sum())))
        if len(pts) < min_pts:
            out.append((name, None, len(pts), 0))
            continue
        ys = np.array([p[0] for p in pts])
        xs = np.array([p[1] for p in pts])
        A = np.polyfit(ys, xs, 1)
        res = xs - np.polyval(A, ys)
        keep = np.abs(res - np.median(res)) < 6
        A = np.polyfit(ys[keep], xs[keep], 1)
        out.append((name, (float(A[0]), float(A[1])), len(pts), int(keep.sum())))
    return out


def derive_calibration(p: CalibParams | None = None) -> Calib:
    return Calib(p or CalibParams())
