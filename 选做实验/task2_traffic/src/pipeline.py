"""
pipeline.py —— 车辆检测 + 多目标跟踪（卡尔曼 + 匈牙利匹配）

设计要点
--------
1. **检测**：固定机位 ⇒ 用 MOG2 混合高斯背景建模。阴影被标记为 127 后剔除，
   避免"车+影"连成一大块。检测只在"着色区域(tint area)"内进行。
2. **跟踪在世界坐标（路面米制）里做**，而不是在像素里做：
   路面上的匀速运动映射到 (X, Z) 后仍是匀速直线，因此卡尔曼的匀速模型是**精确成立**的；
   若在像素域做，车辆越远移动越慢，匀速模型会系统性失配。
3. **关联**用匈牙利算法做全局最优匹配（而非贪心），代价为世界坐标下的欧氏距离，
   并附加"尺寸相容性"门限，抑制检测框跳变导致的错配。
4. **测速**不用卡尔曼的速度状态（受噪声与初始化影响大），而是对整条轨迹的
   (t, X, Z) 做**稳健线性回归**，斜率即速度分量 —— 利用了全部观测，方差最小。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["DetectorConfig", "Detector", "Track", "MultiObjectTracker"]


# ==========================================================================
# 检测
# ==========================================================================
@dataclass
class DetectorConfig:
    history: int = 250
    var_threshold: float = 30.0
    detect_shadows: bool = True
    open_ksize: int = 5
    close_ksize: int = 11
    min_area: int = 3500
    min_w: int = 28
    min_h: int = 26
    max_w_lane_ratio: float = 2.6    # 宽度超过 2.6 个车道宽 → 判为粘连，做拆分
    warmup: int = 30                 # 背景模型预热帧数（此间不输出检测）


class Detector:
    """MOG2 背景建模 + 形态学 + 连通域 + 粘连拆分。"""

    def __init__(self, cfg: DetectorConfig, roi_mask: np.ndarray, calib):
        self.cfg = cfg
        self.roi = roi_mask
        self.calib = calib
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg.history,
            varThreshold=cfg.var_threshold,
            detectShadows=cfg.detect_shadows,
        )
        self.frame_idx = 0

    def _split_by_area(self, mask, x, y, w, h):
        """对一个过宽的粘连连通域，沿列方向的前景投影找极小值处切分。

        车与车之间的间隙在列投影上表现为低谷，据此可把并排/首尾相接的两个目标分开。
        """
        sub = mask[y:y + h, x:x + w]
        col = sub.sum(axis=0).astype(np.float64)
        if len(col) < 12:
            return [(x, y, w, h)]
        k = max(3, len(col) // 12)
        col_s = np.convolve(col, np.ones(k) / k, mode="same")
        # 只在中间一半范围内找低谷，避免把边缘削掉
        lo, hi = len(col_s) // 4, 3 * len(col_s) // 4
        if hi <= lo:
            return [(x, y, w, h)]
        j = int(np.argmin(col_s[lo:hi])) + lo
        if col_s[j] > 0.55 * col_s.mean():
            return [(x, y, w, h)]          # 没有明显低谷，不拆
        return [(x, y, j, h), (x + j, y, w - j, h)]

    def detect(self, frame: np.ndarray):
        """返回 [(x, y, w, h, area), ...]（图像坐标）。"""
        self.frame_idx += 1
        fg = self.bg.apply(frame)
        fg[fg == 127] = 0                              # 剔除阴影
        fg = cv2.bitwise_and(fg, self.roi)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                              np.ones((self.cfg.open_ksize,) * 2, np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                              np.ones((self.cfg.close_ksize,) * 2, np.uint8))
        if self.frame_idx <= self.cfg.warmup:
            return []

        n, _, st, _ = cv2.connectedComponentsWithStats(fg, 8)
        out = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < self.cfg.min_area or w < self.cfg.min_w or h < self.cfg.min_h:
                continue
            lane_w = float(self.calib.lane_px(y + h / 2.0))
            if w > self.cfg.max_w_lane_ratio * lane_w:
                out.extend(self._split_by_area(fg, x, y, w, h))
            else:
                out.append((x, y, w, h, a))
        # 拆分后的子块重新做最小尺寸校验
        out = [o for o in out if o[2] >= self.cfg.min_w and o[3] >= self.cfg.min_h]
        return out


# ==========================================================================
# 单条航迹：卡尔曼滤波（世界坐标下的匀速模型）
# ==========================================================================
class Track:
    """一条航迹。状态向量 x = [X, Z, vX, vZ]（路面米制）。"""

    _next_id = 1

    def __init__(self, X: float, Z: float, bbox, frame_idx: int):
        self.id = Track._next_id
        Track._next_id += 1
        self.x = np.array([X, Z, 0.0, 0.0], dtype=np.float64)
        # 观测噪声：X 方向由横向比例决定，Z 方向因透视压缩而不确定度更大
        self.P = np.diag([1.0, 4.0, 25.0, 100.0])
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.R = np.diag([0.35 ** 2, 1.2 ** 2])   # 观测噪声协方差
        self.Q = np.diag([0.02, 0.08, 0.6, 2.0])  # 过程噪声

        self.bbox = bbox
        self.hits = 1
        self.miss = 0
        self.age = 0
        # 观测序列：(帧号, X, Z, u底边, v底边, 是否有效)
        #   - 底边坐标用于越线计数（车辆尾部通过计数线是一个几何上明确的事件）
        #   - 有效性用于测速：当检测框贴住图像上/下边界时，车辆被裁切，
        #     框的几何中心与底边都会"钉"在边界上，此时算出的 Z 不随时间变化，
        #     会把速度严重低估，必须剔除。
        self.track: list[tuple[int, float, float, float, float, bool]] = [
            (frame_idx, X, Z, bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3], True)]
        # 卡尔曼滤波后的状态序列（测速用）：(帧号, X, Z, 是否有效)
        self.filt: list[tuple[int, float, float, bool]] = [(frame_idx, X, Z, True)]
        self.history_bbox: list = [bbox]
        self.confirmed = False

    # ---------------- 滤波 ----------------
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        return self.x[0], self.x[1]

    def update(self, X: float, Z: float, bbox):
        z = np.array([X, Z], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.bbox = bbox
        self.history_bbox.append(bbox)
        self.hits += 1
        self.miss = 0

    def add_observation(self, frame_idx: int, X: float, Z: float, u: float, v: float,
                        valid: bool = True):
        """记录一帧观测。同时保存世界坐标 (X,Z) 与图像坐标 (u,v)：
        世界坐标用于测速，图像坐标用于判定是否穿越计数线。"""
        self.track.append((frame_idx, X, Z, u, v, valid))
        # 同时记录卡尔曼滤波后的位置（测速用，噪声更小）
        self.filt.append((frame_idx, float(self.x[0]), float(self.x[1]), valid))

    # ---------------- 属性 ----------------
    @property
    def pos(self):
        return self.x[0], self.x[1]

    def speed_kmh(self, fps: float, min_samples: int = 10):
        """对整条轨迹做稳健线性回归，返回 (速度 km/h, 有效样本数, 拟合 R²)。

        使用**卡尔曼滤波后的**轨迹而非原始检测点：路面坐标 Z 对图像行 v 极其敏感
        （在 ROI 内约 0.1 m/px 以上），检测框底边的抖动会被放大成显著的速度噪声。
        回归时同时剔除离群样本（残差 > 2.5σ），避免个别错误关联污染测速。
        """
        if len(self.filt) < min_samples:
            return None
        arr = np.array([f for f in self.filt if f[3]], dtype=np.float64)
        if len(arr) < min_samples:
            return None
        t = (arr[:, 0] - arr[0, 0]) / fps
        X, Z = arr[:, 1], arr[:, 2]

        def robust_slope(t, y):
            A = np.vstack([t, np.ones_like(t)]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            for _ in range(3):
                r = y - A @ coef
                s = np.std(r) + 1e-9
                keep = np.abs(r) < 2.5 * s
                if keep.sum() < 3:
                    break
                coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
            pred = A @ coef
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
            return float(coef[0]), 1.0 - ss_res / ss_tot

        vx, r2x = robust_slope(t, X)
        vz, r2z = robust_slope(t, Z)
        v = float(np.hypot(vx, vz)) * 3.6     # m/s -> km/h
        return v, len(arr), min(r2x, r2z)


# ==========================================================================
# 多目标跟踪
# ==========================================================================
@dataclass
class TrackerConfig:
    gate_m: float = 4.0          # 关联门限（米，世界坐标）
    max_miss: int = 18           # 允许连续丢失的帧数
    min_hits: int = 5            # 连续命中多少帧后确认
    max_speed_kmh: float = 220.0 # 运动合理性约束（用于剔除误关联）


class MultiObjectTracker:
    def __init__(self, cfg: TrackerConfig, calib, fps: float):
        self.cfg = cfg
        self.calib = calib
        self.fps = fps
        self.tracks: list[Track] = []
        self.confirmed_history: list[Track] = []
        self.frame_idx = -1

    def _to_ground(self, bbox, img_h: int):
        """把检测框换算成世界坐标。

        返回 (X, Z, u_bottom, v_bottom, valid)：
          - (X, Z) 取自检测框**几何中心**：中心对框的高度变化不敏感；
            若用底边，车辆从画面底部进入时底边会被图像边界\"钉住\"，
            导致 Z 恒定不变、速度被严重低估。
          - (u_bottom, v_bottom) 取自底边中点，用于越线计数（尾部通过是明确的几何事件）。
          - valid=False 表示检测框贴住了图像上/下边界，该帧的几何量不可信。
        """
        x, y, w, h = bbox[:4]
        u_c = x + w / 2.0
        v_c = y + h / 2.0
        X, Z = self.calib.pixel_to_ground(u_c, v_c)
        u_b = x + w / 2.0
        v_b = y + h
        valid = (y > 2) and (y + h < img_h - 2)
        return float(X), float(Z), float(u_b), float(v_b), bool(valid)

    def step(self, detections, frame_idx: int, img_h: int = 1080):
        self.frame_idx = frame_idx
        # 1) 预测
        for tr in self.tracks:
            tr.predict()

        # 2) 构造代价矩阵
        dets_g = [self._to_ground(d, img_h) for d in detections]
        cost = np.full((len(self.tracks), len(detections)), 1e6)
        for i, tr in enumerate(self.tracks):
            px, pz = tr.pos
            for j, (X, Z, _, _, _) in enumerate(dets_g):
                if not np.isfinite(X) or not np.isfinite(Z):
                    continue
                d = np.hypot(X - px, Z - pz)
                if d <= self.cfg.gate_m:
                    cost[i, j] = d
        # 3) 匈牙利匹配
        matches, unmatched_tr, unmatched_det = [], [], []
        if cost.size and len(detections):
            rows, cols = linear_sum_assignment(cost)
            assigned_r, assigned_c = set(), set()
            for r, c in zip(rows, cols):
                if cost[r, c] < self.cfg.gate_m:
                    matches.append((r, c))
                    assigned_r.add(r)
                    assigned_c.add(c)
            unmatched_tr = [i for i in range(len(self.tracks)) if i not in assigned_r]
            unmatched_det = [j for j in range(len(detections)) if j not in assigned_c]
        else:
            unmatched_tr = list(range(len(self.tracks)))
            unmatched_det = list(range(len(detections)))

        # 4) 更新已匹配
        for r, c in matches:
            tr = self.tracks[r]
            X, Z, u_b, v_b, valid = dets_g[c]
            tr.update(X, Z, detections[c])
            tr.add_observation(frame_idx, X, Z, u_b, v_b, valid)

        # 5) 未匹配航迹：丢失计数
        for r in unmatched_tr:
            self.tracks[r].miss += 1
        # 6) 未匹配检测：新建航迹
        for c in unmatched_det:
            X, Z, u_b, v_b, valid = dets_g[c]
            if np.isfinite(X) and np.isfinite(Z):
                self.tracks.append(Track(X, Z, detections[c], frame_idx))

        # 7) 航迹管理
        alive = []
        for tr in self.tracks:
            if tr.miss > self.cfg.max_miss:
                if tr.confirmed:
                    self.confirmed_history.append(tr)
            else:
                if not tr.confirmed and tr.hits >= self.cfg.min_hits:
                    tr.confirmed = True
                alive.append(tr)
        self.tracks = alive
        return self.tracks

    def finalize(self):
        self.confirmed_history.extend([t for t in self.tracks if t.confirmed])
        self.confirmed_history.sort(key=lambda t: t.id)
        return self.confirmed_history
