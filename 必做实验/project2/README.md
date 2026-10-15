# Project #2：边缘检测与线段提取（Edge detection and line segment extraction）

南京航空航天大学 自动化学院《模式识别与图像处理》课内实验 **必做实验 Project #2** 的完整实现。

## 1. 实验要求（摘自《实验指导书》）

> **Project #2**（Project #1 或者 #2 必须选做一个）
> 该 Project 主要是为了掌握和实践图像处理中的图像增强和边缘检测的知识。
>
> 1. 将 `Image1_withnoise.jpg` 转换为灰度图像 A。
> 2. 利用中值滤波的方法，编制程序对图像 A 进行中值滤波，得到滤波后的图像 B。
> 3. 利用锐化方法，编制程序，对图像 A 和 B 分别使用罗伯特算子、索伯尔算子、canny 算子、
>    拉普拉斯算子，分别进行边缘检测，比较哪种边缘检测的方法好，以及进行中值滤波能否提高
>    线段边缘检测的精度和召回率（Precision、Recall 指标）。
> 4. 在边缘检测基础上将两条电力线完整的提取出来：求解直线方程参数（建议尝试 Hough 变换的
>    方法），给出线段边缘点的序列，并评价检测效果（Precision、Recall 指标）。

## 2. 目录结构

```
project2/
├── README.md                     本文件
├── requirements.txt              Python 依赖
├── 实验报告_Project2.md          实验报告（Markdown，含封面占位）
├── data/
│   └── ground_truth/
│       └── power_lines.json      两条电力线的真值参数（人工核定）
├── src/
│   ├── imops.py                  自实现基础算子（灰度/中值/卷积/Robert/Sobel/Laplacian/Canny/Otsu）
│   ├── hough.py                  自实现 Hough 变换、峰值聚类、线段回溯与 TLS 精化
│   ├── evaluate.py               Precision / Recall / F1（带容差的双向匹配）
│   ├── groundtruth.py            真值读取、栅格化、ROI 生成
│   ├── viz.py                    全部实验插图与结果汇总输出
│   └── run_experiment.py         主实验流程（一键复现全部结果）
└── results/
    ├── REPORT_RESULTS.md         指标汇总（自动生成，供报告直接引用）
    ├── figures/                  10 张实验插图
    └── metrics/                  CSV / JSON 指标与线段边缘点序列
```

## 3. 运行方法

```bash
cd project2
pip install -r requirements.txt
python3 src/run_experiment.py
```

全部结果（插图、指标、汇总）写入 `results/`。在 512×512 图像上整条流水线约 **5 秒**。

> 说明：所有算法（灰度化、中值滤波、卷积、四种边缘检测算子、大津法阈值、Hough 变换）
> 均由 NumPy 直接实现，**未调用 OpenCV/Scikit-image 等库的高层算子**；
> 仅使用 Pillow 读图、SciPy 做欧氏距离变换、Matplotlib 出图。

## 4. 方法要点

| 步骤 | 方法 | 实现位置 |
|---|---|---|
| 灰度化 | $Y=0.299R+0.587G+0.114B$ | `imops.to_gray` |
| 中值滤波 | reflect 边界 + 滑窗取中位数，窗口 3×3 / 5×5 | `imops.median_filter` |
| Robert 算子 | 2×2 交叉差分，$\|G_x\|+\|G_y\|$ | `imops.robert` |
| Sobel 算子 | 3×3 平滑差分，$\sqrt{G_x^2+G_y^2}$ | `imops.sobel` |
| Laplacian 算子 | 4/8 邻域二阶差分，取绝对值 | `imops.laplacian` |
| Canny 算子 | 高斯平滑→Sobel 梯度→非极大值抑制→分位数双阈值+滞后连接 | `imops.canny` |
| 二值化 | 大津法（最大化类间方差），响应先做 99.5% 分位裁剪 | `imops.otsu_threshold` |
| 直线检测 | $\rho=x\cos\theta+y\sin\theta$ 累加器 + 峰值 NMS + 聚类 | `hough.hough_accumulator` / `find_peaks` |
| 线段提取 | 沿直线在 ±2 px 法向带内回溯边缘点 → 切分连通段 → TLS 精化 | `hough.walk_line` / `line_from_points` |
| 细暗线验证 | 法向灰度剖面的对比度与半高全宽（FWHM），排除宽浅地形结构 | `run_experiment.wire_score` |

### 两条电力线的真值（人工核定）

采用 Hough 参数化 $\rho = x\cos\theta + y\sin\theta$：

| 线 | $\theta$ | $\rho$ | 直线方程 | 端点 | 法向剖面 |
|---|---|---|---|---|---|
| L1 | 76.70° | 206.00 | $0.230050x + 0.973179y = 206.0000$ | (0, 211.7) → (511, 90.9) | 背景≈132，谷底≈92，宽 3 px |
| L2 | 81.35° | 420.75 | $0.150398x + 0.988626y = 420.7500$ | (0, 425.6) → (511, 347.9) | 背景≈134，谷底≈102，宽 2 px |

获取流程：大尺度黑帽变换 → $(\theta,\rho)$ 全参数空间扫描统计线上平均暗对比度 →
候选线法向剖面检验（真线为**窄而深**的暗谷，地物为**宽而浅**）→ 局部参数优化 → 叠加原图人工复核。

### 评价指标定义

在电力线 ROI（真值线 ±6 px 波段）内，采用带容差的双向匹配（$tol = 2$ px）：

$$
TP = \#\{\text{预测边缘像素}: d(p, GT) \le tol\},\quad
FP = \#\{\text{预测边缘像素}: d(p, GT) > tol\}
$$
$$
FN = \#\{\text{真值像素}: d(q, Pred) > tol\},\quad
P = \frac{TP}{TP+FP},\quad R = \frac{TP}{TP+FN},\quad
F_1 = \frac{2PR}{P+R}
$$

同时统计 $FN$ 是必要的：若只检查"预测是否落在真值附近"，一条预测了整幅图
边缘的方法 Precision 会被严重高估。

## 5. 主要结论（详见报告）

- 中值滤波显著提升 Robert / Sobel / Laplacian 的效果：
  Laplacian 的 $F_1$ 从 0.659 → 0.940（+0.281），Sobel 从 0.930 → 0.955。
- Canny 因自身已含高斯平滑与自适应阈值，中值滤波对其增益为负
  （$F_1$ 0.941 → 0.923），说明**预处理与算子自身平滑能力存在冗余**。
- 在电力线 ROI 内，中值滤波后 **Sobel** 表现最好（$P=0.953, R=0.957, F_1=0.955$）。
- 采用"Sobel(中值滤波后) + Hough + 细暗线验证 + 相对峰值准则"，
  两条电力线均被完整提取，线级 $P=R=1.0$，直线参数误差 $|\Delta\rho| < 0.9$ px、$|\Delta\theta| < 0.03°$。

## 6. 转 Word 时的格式要求

《实验指导书》规定（Markdown 无法表达，转 Word 时需手动设置）：

- 一级标题：四号黑体；二级标题：小四黑体
- 正文：小四宋体，1.2 倍行距
- 公式：居中并编号（编号靠右）
- 图、表及其标题：居中
- 必须使用教师提供的封面（本报告的封面已按模板给出占位）
