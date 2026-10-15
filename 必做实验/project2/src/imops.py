"""
imops.py —— 本项目自实现的图像处理基础算子（不依赖 OpenCV 的高层算子实现）

包含：
    to_gray            : 彩色转灰度
    conv2d             : 2D 卷积（reflect 边界填充）
    median_filter      : 中值滤波（滑窗，自实现）
    gaussian_blur      : 可分离高斯平滑
    robert             : Robert 交叉差分算子
    sobel              : Sobel 算子
    laplacian          : Laplacian 算子（4 邻域 / 8 邻域）
    canny              : Canny 算子（平滑-梯度-非极大值抑制-双阈值滞后）
    otsu_threshold     : 大津法自适应阈值
    normalize_u8       : 归一化到 uint8

说明：所有算子均由 numpy 直接实现，便于在报告中说明原理与复杂度。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "to_gray",
    "conv2d",
    "median_filter",
    "gaussian_kernel",
    "gaussian_blur",
    "robert",
    "sobel",
    "laplacian",
    "canny",
    "otsu_threshold",
    "normalize_u8",
]


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    """彩色图像转灰度图像。

    采用 ITU-R BT.601 亮度权重：Y = 0.299R + 0.587G + 0.114B
    输入若已是单通道则原样返回（并保证为 uint8）。
    """
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:  # BGRA
        img = img[:, :, :3]
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def normalize_u8(x: np.ndarray) -> np.ndarray:
    """把任意浮点响应线性拉伸到 [0, 255]，便于显示与二值化。"""
    x = x.astype(np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    return np.clip((x - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _pad_reflect(img: np.ndarray, pad: int) -> np.ndarray:
    """镜像（reflect）边界填充，避免引入零值造成假边缘。"""
    return np.pad(img, pad, mode="reflect")


def conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """二维相关/卷积运算（核不翻转，即数学意义上的相关，边缘检测中常用）。

    参数
    ----
    img    : (H, W) 浮点或整型数组
    kernel : (kh, kw) 卷积核，kh、kw 均为奇数

    返回
    ----
    out : (H, W) float32，与输入同尺寸（边界用 reflect 填充）
    """
    img = img.astype(np.float32)
    kernel = kernel.astype(np.float32)
    kh, kw = kernel.shape
    assert kh % 2 == 1 and kw % 2 == 1, "卷积核尺寸必须为奇数"
    ph, pw = kh // 2, kw // 2
    padded = _pad_reflect(img, max(ph, pw))
    padded = padded[..., :]  # reflect 填充量一致，下面按需取窗口
    H, W = img.shape
    out = np.zeros((H, W), dtype=np.float32)
    # 直接滑窗累加：核尺寸很小（<=5），朴素实现足够快且逻辑清晰
    for i in range(kh):
        for j in range(kw):
            w = kernel[i, j]
            if w == 0:
                continue
            out += w * padded[i : i + H, j : j + W]
    return out


# --------------------------------------------------------------------------
# 中值滤波
# --------------------------------------------------------------------------
def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """中值滤波：用窗口内灰度的中位数替代中心像素。

    对椒盐/脉冲噪声几乎可以完全滤除，同时对阶跃边缘的模糊比均值滤波小。

    参数
    ----
    img   : (H, W) 灰度图
    ksize : 窗口边长（奇数），典型 3 或 5

    返回
    ----
    (H, W) uint8
    """
    assert ksize % 2 == 1, "窗口尺寸必须为奇数"
    img = img.astype(np.uint8)
    pad = ksize // 2
    padded = _pad_reflect(img.astype(np.float32), pad)
    H, W = img.shape
    # 用滑动窗口视图一次性取出所有 (H, W, k, k) 窗口，再对后两维取中值
    win = np.lib.stride_tricks.sliding_window_view(padded, (ksize, ksize))
    win = win[:H, :W]
    out = np.median(win, axis=(-1, -2))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# 高斯平滑
# --------------------------------------------------------------------------
def gaussian_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    """生成归一化的一维高斯核（可分离卷积用）。"""
    if radius is None:
        radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


def gaussian_blur(img: np.ndarray, sigma: float = 1.4) -> np.ndarray:
    """可分离高斯平滑：先横向再纵向，复杂度由 O(k^2) 降为 O(2k)。"""
    img = img.astype(np.float32)
    k = gaussian_kernel(sigma)
    r = len(k) // 2
    padded = _pad_reflect(img, r)
    H, W = img.shape
    tmp = np.zeros((H, W), dtype=np.float32)
    for i, w in enumerate(k):
        tmp += w * padded[r : r + H, i : i + W]
    padded = _pad_reflect(tmp, r)
    out = np.zeros((H, W), dtype=np.float32)
    for i, w in enumerate(k):
        out += w * padded[i : i + H, r : r + W]
    return out


# --------------------------------------------------------------------------
# 一阶微分算子：Robert / Sobel
# --------------------------------------------------------------------------
def robert(img: np.ndarray, mode: str = "l1") -> np.ndarray:
    """Robert 交叉差分算子。

    两个 2x2 模板：
        Gx = [[ 1,  0],      Gy = [[ 0,  1],
              [ 0, -1]]            [-1,  0]]
    定位精度高（对角差分），但对噪声敏感，且输出比原图小 1 像素（已补零对齐）。

    参数
    ----
    mode : 'l1' 用 |Gx|+|Gy|（运算快）；'l2' 用 sqrt(Gx^2+Gy^2)
    """
    img = img.astype(np.float32)
    kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    H, W = img.shape
    # 2x2 模板：对 (H-1, W-1) 做内积，再补一行一列
    gx = np.zeros((H, W), dtype=np.float32)
    gy = np.zeros((H, W), dtype=np.float32)
    a = img[:-1, :-1]
    b = img[:-1, 1:]
    c = img[1:, :-1]
    d = img[1:, 1:]
    gx[:-1, :-1] = a - d
    gy[:-1, :-1] = b - c
    if mode == "l1":
        return np.abs(gx) + np.abs(gy)
    return np.sqrt(gx**2 + gy**2)


def sobel(img: np.ndarray, mode: str = "l2") -> np.ndarray:
    """Sobel 算子：先平滑再差分，抗噪能力优于 Robert。

        Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
              [-2, 0, 2],            [ 0,  0,  0],
              [-1, 0, 1]]            [ 1,  2,  1]]
    """
    img = img.astype(np.float32)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = conv2d(img, kx)
    gy = conv2d(img, ky)
    if mode == "l1":
        return np.abs(gx) + np.abs(gy)
    return np.sqrt(gx**2 + gy**2)


# --------------------------------------------------------------------------
# 二阶微分算子：Laplacian
# --------------------------------------------------------------------------
def laplacian(img: np.ndarray, neighbors: int = 4, use_abs: bool = True) -> np.ndarray:
    """Laplacian 算子（二阶微分）。

        neighbors=4 : [[ 0,  1,  0],
                       [ 1, -4,  1],
                       [ 0,  1,  0]]
        neighbors=8 : [[ 1,  1,  1],
                       [ 1, -8,  1],
                       [ 1,  1,  1]]
    Laplacian 是各向同性算子但会放大噪声，通常需先平滑。
    """
    if neighbors == 4:
        k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    else:
        k = np.ones((3, 3), dtype=np.float32)
        k[1, 1] = -8.0
    r = conv2d(img.astype(np.float32), k)
    return np.abs(r) if use_abs else r


# --------------------------------------------------------------------------
# 阈值：大津法
# --------------------------------------------------------------------------
def otsu_threshold(x: np.ndarray, nbins: int = 256) -> float:
    """大津法（Otsu）自动阈值：最大化类间方差。"""
    v = x.astype(np.float32).ravel()
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return lo
    hist, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    m0 = np.cumsum(p * centers)
    mT = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = np.where(w0 > 0, m0 / np.maximum(w0, 1e-12), 0.0)
        mu1 = np.where(w1 > 0, (mT - m0) / np.maximum(w1, 1e-12), 0.0)
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    k = int(np.nanargmax(sigma_b))
    return float(centers[k])


# --------------------------------------------------------------------------
# Canny 算子
# --------------------------------------------------------------------------
def _nms(mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """沿梯度方向做非极大值抑制，细化边缘到单像素宽。"""
    H, W = mag.shape
    out = np.zeros((H, W), dtype=np.float32)
    # 量化到 4 个方向：0/45/90/135 度
    a = (np.rad2deg(ang) + 180.0) % 180.0
    q = np.zeros_like(a, dtype=np.int8)
    q[(a < 22.5) | (a >= 157.5)] = 0
    q[(a >= 22.5) & (a < 67.5)] = 1
    q[(a >= 67.5) & (a < 112.5)] = 2
    q[(a >= 112.5) & (a < 157.5)] = 3

    m = mag
    # 方向 0：比较左右
    n1 = np.zeros_like(m)
    n2 = np.zeros_like(m)
    n1[:, 1:], n2[:, :-1] = m[:, :-1], m[:, 1:]          # q==0 左右
    sel = q == 0
    ok = sel & (m >= n1) & (m >= n2)
    out[ok] = m[ok]
    # 方向 1（45°）：比较左上-右下
    n1 = np.zeros_like(m)
    n2 = np.zeros_like(m)
    n1[1:, 1:], n2[:-1, :-1] = m[:-1, :-1], m[1:, 1:]
    sel = q == 1
    ok = sel & (m >= n1) & (m >= n2)
    out[ok] = m[ok]
    # 方向 2（90°）：比较上下
    n1 = np.zeros_like(m)
    n2 = np.zeros_like(m)
    n1[1:, :], n2[:-1, :] = m[:-1, :], m[1:, :]
    sel = q == 2
    ok = sel & (m >= n1) & (m >= n2)
    out[ok] = m[ok]
    # 方向 3（135°）：比较右上-左下
    n1 = np.zeros_like(m)
    n2 = np.zeros_like(m)
    n1[1:, :-1], n2[:-1, 1:] = m[:-1, 1:], m[1:, :-1]
    sel = q == 3
    ok = sel & (m >= n1) & (m >= n2)
    out[ok] = m[ok]
    return out


def _hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """双阈值滞后连接：强边缘全部保留，弱边缘仅在连通到强边缘时保留。"""
    from collections import deque

    H, W = strong.shape
    out = np.zeros((H, W), dtype=bool)
    dq = deque()
    ys, xs = np.nonzero(strong)
    for y, x in zip(ys, xs):
        out[y, x] = True
        dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and weak[ny, nx] and not out[ny, nx]:
                    out[ny, nx] = True
                    dq.append((ny, nx))
    return out


def canny(
    img: np.ndarray,
    sigma: float = 1.4,
    strong_frac: float = 0.03,
    weak_frac: float = 0.15,
    thresholds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Canny 边缘检测（完整自实现）。

    流程：高斯平滑 -> Sobel 梯度 -> 幅值/方向 -> 非极大值抑制
          -> 双阈值 + 滞后连接 -> 二值边缘图

    参数
    ----
    sigma       : 高斯平滑标准差
    strong_frac : 强边缘像素占比（按非极大值抑制后幅值的分位数确定高阈值），
                  经验值 0.02~0.05。
    weak_frac   : 弱边缘像素占比（决定低阈值），经验值 0.10~0.20。
    thresholds  : 若给定 (low, high) 则直接使用固定阈值，忽略上面的占比参数。

    说明：直接用固定阈值的 Canny 对噪声图很敏感（低阈值取低了会把整幅
    噪点连成一片）；这里改用"分位数阈值"，使高/低阈值自适应图像内容，
    也与大津法二值化在思路上保持一致，便于公平比较。
    """
    sm = gaussian_blur(img, sigma)
    gx = conv2d(sm, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32))
    gy = conv2d(sm, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32))
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)
    thin = _nms(mag, ang)

    if thresholds is not None:
        lo, hi = float(thresholds[0]), float(thresholds[1])
    else:
        hi = float(np.percentile(thin, (1.0 - strong_frac) * 100.0))
        lo = float(np.percentile(thin, (1.0 - weak_frac) * 100.0))
    strong = thin >= hi
    weak = (thin >= lo) & ~strong
    return _hysteresis(strong, weak).astype(np.uint8)


def gradient_angle(img: np.ndarray, sigma: float = 1.4) -> np.ndarray:
    """返回 Sobel 梯度方向（弧度），供分析边缘走向使用。"""
    sm = gaussian_blur(img, sigma)
    gx = conv2d(sm, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32))
    gy = conv2d(sm, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32))
    return np.arctan2(gy, gx)
