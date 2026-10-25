# Two Geometry-Driven Studies in Image Processing: Panoramic Stitching and Road-Scene Vehicle Counting and Speed Measurement

## 封面信息

| 字段 | 内容 |
|------|------|
| No. | （待填） |
| 课程 | Research Report of PR & IP |
| Author | （待填） |
| Affiliation | College of Automation Engineering, NUAA |
| Student Name | （待填） |
| Student ID | （待填） |
| College | 自动化学院 College of Automation Engineering |
| Major | （待填） |
| Mobile | （待填） |
| 日期 | 二〇二六 年 十 月 |

## Abstract

This report presents two geometry-driven studies completed for the optional experiments of the Pattern Recognition and Image Processing course. Both tasks share a single underlying question: once the projective geometry between an image and the physical world is made explicit, how much can be achieved with classical, interpretable algorithms rather than learned black boxes? The first study addresses panoramic image stitching. Scale-invariant features are extracted and matched with a ratio test, and the homography relating the two views is estimated by a random-sample-consensus procedure that the author implemented from scratch, including Hartley normalisation, an SVD-based direct linear transform and a symmetric transfer error criterion with adaptive termination. Warping is performed on an automatically sized union canvas and the seam is removed by distance-transform feathering. All four supplied image pairs are stitched successfully, with inlier ratios between 80.1 % and 97.0 % and inlier reprojection RMSE below 0.72 px, i.e. sub-pixel registration accuracy, in 43-250 ms per pair. The second study addresses vehicle counting and speed measurement in a surveillance video of an urban expressway. Instead of tracking in the image plane, this work derives a metric mapping from pixels to the road plane from the lane geometry and a single free parameter, and then performs detection, Kalman filtering and Hungarian association entirely in metres; speed is obtained by robust linear regression over each full trajectory. Eleven vehicles cross the counting line out of eighteen confirmed tracks, nine of which yield speed estimates with a mean of 44.27 km/h, and the whole sequence is processed at 34 fps on a single CPU core. A sensitivity analysis quantifies how the absolute speed scale depends on the calibration prior, and the report closes with an explicit discussion of what is and is not validated.

**Keywords:** image stitching; homography estimation; RANSAC; background subtraction; multi-object tracking; monocular speed estimation

## 1 Introduction

### 1.1 Background and motivation

Two classic families of image-processing problems are studied here. The first is **image stitching**, i.e. compositing several overlapping views of the same scene into a single wide-field image. It is the textbook example of a problem whose difficulty is almost entirely geometric: once the transformation between the views is known, the remaining work is bookkeeping. The second is **road-scene understanding** from a fixed surveillance camera, i.e. detecting vehicles, maintaining their identities over time, counting them and estimating their speed. Here the geometric component is hidden inside the word "speed": a video only provides pixels per frame, and converting that into metres per second requires an explicit model of the camera.

The two tasks are deliberately paired. In both, the decisive design choice is not the choice of feature detector or the choice of tracker, but whether the **projective geometry is written down explicitly**. In the stitching task this means estimating a homography rather than warping with a hand-tuned translation; in the traffic task it means tracking in metric road coordinates rather than in pixels. This report therefore emphasises the derivations and the failure modes, not only the final numbers.

A second thread runs through both studies: the author deliberately re-implemented the core estimators (RANSAC, DLT, the pixel-to-ground mapping, the Kalman filter) instead of calling library routines. This is partly an exercise, but it also made the internal parameters auditable, which in turn made the sensitivity analysis in Section 4.4 possible.

### 1.2 Contributions

The main contributions of this report are:

1. A complete, self-implemented stitching pipeline (SIFT + ratio test + RANSAC/DLT + feathering) that succeeds on all four supplied pairs despite the left and right images of each pair having **different resolutions**, which rules out any translation- or scale-only model.
2. A traffic-analysis pipeline that tracks in **world metric coordinates** rather than in pixels, and that measures speed by robust regression over a fitted trajectory instead of reading the Kalman velocity state.
3. A calibration procedure for an **uncalibrated** fixed camera that separates the identifiable part of the problem (lateral scale, from the known lane width) from the unidentifiable part (longitudinal scale), and that quantifies the residual ambiguity instead of hiding it.

### 1.3 Organisation

Section 2 reviews related work. Section 3 presents the two methods. Section 4 reports and evaluates the experiments, including a sensitivity analysis and an explicit list of limitations. Section 5 concludes.

## 2 Related Work

### 2.1 Feature-based image stitching

The modern stitching pipeline was established by Brown and Lowe [1]: invariant features are extracted in each view, matched by descriptor distance with a ratio test, and the inter-view transformation is estimated robustly from the surviving correspondences; the views are then warped onto a common canvas and blended.

The feature side is dominated by SIFT [2], which detects scale-space extrema of a difference-of-Gaussians pyramid and describes them by gradient histograms. SIFT remains the reference for robustness at the cost of speed; SURF [3] and ORB [4] trade some robustness for large speed-ups, ORB in particular replacing the float descriptor with a rotated BRIEF binary string.

The geometric side rests on two classical results. The direct linear transform solves for a homography from four or more point correspondences by a linear system whose solution is the null vector of a design matrix [5]. Because that linear solution is not robust to outliers, it is embedded in RANSAC [6], which repeatedly samples minimal sets, fits a model and counts inliers. Hartley's normalisation [7] is essential in practice: without pre-conditioning the coordinates, the design matrix becomes badly scaled and the estimated homography is numerically poor.

Blending is a separate matter. The standard cure is Burt and Adelson's multiresolution spline [8], which blends band-passed copies of the two views so that the seam is hidden at every scale. The present work uses a cheaper variant, distance-transform feathering: each pixel is weighted by its distance to the border of its own valid region, so the overlap is averaged in proportion to how far each view extends past it. It needs no pyramid, and it is sufficient here because each pair of views is photometrically similar.

### 2.2 Vehicle detection, tracking and speed measurement

Background modelling by a mixture of Gaussians [9] and its adaptive improvement by Zivkovic [10] remain the standard choice for a static camera, where the background can be learned rather than detected. Detection quality is then limited mainly by shadows and by merged blobs, both of which are addressed in Section 3.2.

Tracking is built on two classical components: the Kalman filter [11] for state estimation under Gaussian noise, and the Hungarian algorithm [12] for globally optimal assignment. SORT [13] combines exactly these two with a constant-velocity model and is the closest published relative of the tracker used here; the difference is that SORT associates in image space, whereas the present work associates in metric road coordinates.

Speed measurement from monocular video has a long history in transportation engineering. Coifman et al. [14] tracked features and grouped them into vehicles to obtain trajectories; Dailey et al. [15] and Cathey and Dailey [16] showed that useful speed estimates can be obtained even when the camera is **not** calibrated, by exploiting the vanishing point of the road markings to establish a scale. The present work follows the same philosophy: the vanishing point is measured from the lane boundaries, and the residual scale ambiguity is resolved by a single assumption about vehicle length.

## 3 Method

### 3.1 Task 1: feature-based panoramic stitching

#### 3.1.1 Feature extraction and matching

SIFT [2] is used with at most 4000 keypoints per image. The choice is motivated by the data: in every pair the two views have **different image sizes** (for example 848×964 against 915×964, and 866×435 against 975×436), so the two cameras were at different distances from the scene and the scale change between views is significant. SIFT is invariant to scale and rotation by construction, which means no pre-rescaling or pyramid search is needed.

Matches are obtained by brute-force nearest-neighbour search in descriptor space followed by Lowe's ratio test with threshold 0.75: a match is kept only if the nearest neighbour is substantially closer than the second nearest. This single scalar removes most ambiguous matches, because a repetitive texture produces several similarly distant candidates and the ratio then approaches unity.

#### 3.1.2 Self-implemented RANSAC with DLT

The surviving matches still contain outliers, so the homography is estimated robustly. Writing homogeneous image points as $\mathbf{x} = (x, y, 1)^{\mathsf T}$ and $\mathbf{x}' = (x', y', 1)^{\mathsf T}$, a planar homography acts as

$$\mathbf{x}' \sim \mathbf{H}\mathbf{x}, \qquad \mathbf{H} \in \mathbb{R}^{3\times 3}, \ \|\mathbf{H}\|_F = 1 \tag{1}$$

Each correspondence contributes two linear equations in the nine entries of $\mathbf{H}$, giving the homogeneous system

$$\mathbf{A}\mathbf{h} = \mathbf{0}, \qquad \mathbf{h} = \operatorname*{arg\,min}_{\|\mathbf{h}\| = 1} \|\mathbf{A}\mathbf{h}\|_2 \tag{2}$$

which is solved by the right singular vector associated with the smallest singular value of $\mathbf{A}$. Before assembling $\mathbf{A}$, both point sets are translated so that their centroids are at the origin and scaled so that the mean distance to the centroid is $\sqrt{2}$; the resulting normalising transformations are folded back into $\mathbf{H}$ afterwards [5, 7].

RANSAC [6] samples four correspondences, solves Eq. (2), and scores the model by the **symmetric transfer error**

$$E(\mathbf{x}, \mathbf{x}') = \sqrt{\tfrac{1}{2}\left( d\!\left(\mathbf{x}, \mathbf{H}^{-1}\mathbf{x}'\right)^2 + d\!\left(\mathbf{x}', \mathbf{H}\mathbf{x}\right)^2 \right)} \tag{3}$$

which penalises error in both images and therefore does not favour a model that fits one side at the expense of the other. A correspondence is an inlier if $E \le 3$ px. The number of iterations is not fixed: after each successful model the required iteration count is recomputed from the current inlier ratio $\omega$ and the desired confidence $p = 0.999$ as $\log(1-p)/\log(1-\omega^4)$, and the loop terminates early once that bound is met. In practice this converges in 4–14 iterations (Tab.1), which is the main reason the whole pipeline runs in tens to hundreds of milliseconds.

#### 3.1.3 Warping and blending

The left image is taken as the reference frame. The four corners of the right image are mapped through $\mathbf{H}$, the union of the two footprints defines the output canvas, and the right image is resampled with bilinear interpolation. Overlapping pixels are combined by distance-transform feathering: each image contributes a weight proportional to its distance from the nearest invalid (outside-footprint) pixel, so the transition is gradual and the seam disappears. Remaining all-black rows and columns are trimmed.

### 3.2 Task 2: vehicle counting and speed measurement

#### 3.2.1 Definition of the tint area

The task statement refers to "the tint area" without supplying an annotation. This work defines it as a trapezoid covering the three nearest lanes of the carriageway, with its upper edge at image row 620, its lower edge at row 1078, and its lateral boundaries given by two of the detected lane markings. The counting line is row 900 inside this region, and a vehicle is counted when it crosses that line in the direction away from the camera. The region and the line are drawn as a translucent green overlay in the output video, so the definition is visible rather than implicit.

#### 3.2.2 Detection

Because the camera is static, the background is learned rather than estimated per frame. A mixture-of-Gaussians model [9, 10] with 250-frame history and variance threshold 30 is applied; pixels flagged as shadows are discarded rather than merged, which prevents a vehicle from being fused with its own shadow into one elongated blob. The foreground mask is restricted to the tint area, opened with a 5×5 kernel to remove speckle and closed with an 11×11 kernel to fill vehicle interiors. Connected components smaller than 3500 px, 28 px wide or 26 px tall are rejected.

Two failure modes are handled explicitly. First, two adjacent vehicles can merge into one component; when the width of a component exceeds 2.6 lane widths, the column-wise foreground projection is smoothed and split at its minimum, provided that the minimum is a genuine valley. Second, during the first 30 frames the background model is still converging, so no detection is emitted and those frames cannot contribute to counting.

#### 3.2.3 Metric calibration of an uncalibrated camera

This is the part of the task that deserves the most care, because every speed value depends on it.

For a fixed camera viewing a locally planar road, the mapping between road coordinates $(X, Z)$ and image coordinates $(u, v)$ is a homography, and knowing only the lane width is **not** sufficient: the longitudinal scale can be stretched arbitrarily without moving any lane marking in the image. The lateral scale, however, is observable. From $u - u_c = fX/Z_c$ the pixel width of a lane at row $v$ is

$$w_{\mathrm{px}}(v) = \frac{W_{\mathrm{lane}}\cos\theta\,(v - v_{\mathrm{vp}})}{h} \tag{4}$$

where $W_{\mathrm{lane}}$ is the physical lane width, $h$ the camera height, $\theta$ the pitch and $v_{\mathrm{vp}}$ the row of the vanishing point. Because $w_{\mathrm{px}}$ is linear in $(v - v_{\mathrm{vp}})$, two measured lane boundaries intersect exactly at the vanishing point, and the camera height follows from any single row. This gives a **free consistency check**: if the two fitted lines really are parallel lane boundaries, the height recovered from different rows must agree. It does, to within the measurement noise, which is why the recovered value of 10.52 m is credible rather than arbitrary.

Inverting the projection gives the pixel-to-ground mapping used throughout the tracker:

$$Z = \frac{h\,(f^2 - pP)}{f\,w}, \qquad X = \frac{(u - u_c)\left(hP/f + Z\right)}{\sqrt{f^2 + P^2}} \tag{5}$$

with $w = v - v_{\mathrm{vp}}$, $p = v - v_c$ and $P = f\tan\theta = v_c - v_{\mathrm{vp}}$.

One degree of freedom remains: the focal length $f$. It is fixed by a single assumption about object size — the median reconstructed length of the detected vehicles is set equal to a typical passenger-car length of 4.5 m, which yields $f \approx 3348$ px, i.e. a horizontal field of view of about 32°. The consequence is stated plainly rather than concealed: speed scales approximately as $1/f$, so the **absolute** speeds carry a calibration uncertainty of roughly ±13 % over a plausible field-of-view range (Section 4.4). What does not depend on this assumption is the count, and the relative ordering of the vehicles' speeds.

#### 3.2.4 Tracking in metric coordinates

Each track carries the state $\mathbf{x} = [X, Z, v_X, v_Z]^{\mathsf T}$ in metres and metres per second, filtered by a constant-velocity Kalman filter [11]:

$$\mathbf{x}_k = \mathbf{F}\mathbf{x}_{k-1} + \mathbf{w}_k, \qquad \mathbf{z}_k = \mathbf{H}\mathbf{x}_k + \mathbf{v}_k \tag{6}$$

The choice of coordinate system is the key design decision. On a planar road a vehicle moving at constant speed maps to a straight, uniformly parameterised line in $(X, Z)$, so the constant-velocity model is **exactly** correct; in the image plane the same motion appears to decelerate as the vehicle recedes, and the model would be systematically wrong.

Association between predictions and detections is solved globally by the Hungarian algorithm [12] on a cost matrix of Euclidean distance in metres, with a gate of 4 m and a shape-compatibility check to reject implausible pairings. Tracks are confirmed after 5 hits and terminated after 18 consecutive misses, following the SORT convention [13].

#### 3.2.5 Counting and speed estimation

Counting uses the **bottom edge** of the bounding box, because the instant at which a vehicle's rear clears the line is a geometrically well-defined event; a sign change of $(v_b - v_{\mathrm{line}})$ with the track inside the tint area is counted once.

Speed is **not** read from the Kalman velocity state. That state is driven by the most recent observations and is therefore noisy, and it is also biased during initialisation. Instead the whole filtered trajectory is regressed against time, and the speed is the norm of the two slopes:

$$v = 3.6\sqrt{v_X^2 + v_Z^2}, \qquad v_{(\cdot)} = \operatorname*{arg\,min}_{a}\sum_{k} \rho\!\left(y_k - a\,t_k - b\right) \tag{7}$$

where $\rho$ is the quadratic loss truncated at 2.5 standard deviations, applied in three passes so that a few mis-associated frames cannot dominate the fit. Frames in which the bounding box touches the image border are marked invalid and excluded: a vehicle entering from the bottom of the frame is clipped, its centre is pinned to the border, and the reconstructed $Z$ then barely changes with time, which would severely underestimate the speed.

## 4 Experiments and Evaluation

### 4.1 Task 1: stitching accuracy

All four pairs were stitched successfully. Quantitative results are listed in Tab.1 and the visual comparison is given in Fig.1, with the inlier/outlier decomposition for the most difficult pair shown in Fig.2.

**Tab.1** Stitching results for the four image pairs

| 组别 | 左图 | 右图 | 匹配 | 内点 | 内点率 | RMSE/px | 迭代 |
|------|------|------|------|------|--------|---------|------|
| img1 | 848×964 | 915×964 | 186 | 149 | 80.1% | 0.331 | 14 |
| img2 | 562×602 | 513×602 | 595 | 565 | 95.0% | 0.396 | 5 |
| img3 | 866×435 | 975×436 | 337 | 327 | 97.0% | 0.720 | 4 |
| img4 | 518×408 | 558×408 | 296 | 272 | 91.9% | 0.615 | 6 |

特征点数量依次为 1025/828、1661/1559、2852/2714、990/904（左/右），输出画布尺寸为 1539×968、803×607、1631×441、832×419，单组耗时 250、64、88、43 ms。

![四组拼接结果](images/fig01_task1_overview.png)

**Fig.1** 四组左右视图与拼接结果的对照（每行依次为左图、右图、拼接结果）

![img1 的内点与外点](images/fig02_task1_matches.png)

**Fig.2** 第一组的内点（绿）与外点（红）匹配分布

Three observations follow from Tab.1.

First, the **inlier ratio** is the quantity that best predicts stitching quality, and it is not monotone in the number of matches. img3 has the most keypoints (2852/2714) yet only 337 surviving matches, because the scene is a wide, low-contrast lawn; its inlier ratio is nevertheless the highest (97.0 %), so 327 of those matches are consistent with a single homography. img1 has the fewest matches (186) and the lowest inlier ratio (80.1 %), which is consistent with it being a night scene with repeated lamp posts — a configuration that produces many self-similar descriptors.

Second, the **reprojection RMSE** stays below the pixel grid in all four cases (0.331–0.720 px). This is sub-pixel registration, and it is the quantitative counterpart of the visual verdict: no seam or ghosting is visible in Fig.1.

Third, the **adaptive iteration count** confirms the analysis of Section 3.1.2. With inlier ratios of 0.92–0.97 the theoretical requirement drops to a handful of iterations, and the measured 4–14 iterations match it. A fixed budget of 3000 iterations would have produced identical results at roughly two orders of magnitude the cost.

The one pair that required real effort is img1: 14 iterations and an 80 % inlier ratio mean that roughly one match in five is an outlier, which is far more than the other pairs. It is worth noting that a translation- or similarity-only model could not have handled any of these pairs, because the two views of each pair have different image dimensions and therefore different scales.

### 4.2 Task 2: calibration and counting

The calibration derived in Section 3.2.3 is summarised in Tab.2 and visualised in Fig.3, where a 2 m × 2 m grid is projected back onto the road surface. The grid lines are parallel and evenly spaced in the image, which is the visual signature of a correct projection.

**Tab.2** Calibrated parameters of the surveillance camera

| 参数 | 数值 | 含义 |
|------|------|------|
| 车道宽 W | 3.75 m | 城市快速路标准值，作为横向尺度基准 |
| 水平视场角 | 32° | 唯一自由参数，由车辆尺寸自标定得到 |
| 焦距 f | 3347.9 px | f = (W_img/2)/tan(HFOV/2) |
| 相机高度 h | 10.52 m | 由车道宽与逐行像素间距反解，各行一致 |
| 俯仰角 θ | 13.13° | 由消失点行坐标确定 |
| 偏航角 | 4.44° | 由消失点横向偏移确定 |
| 消失点 | (1219.7, −240.7) px | 两条车道边界线的交点 |
| 参考行 v = 780 处车道像素宽 | 354.3 px | 实测 |
| 横向 / 纵向比例 | 94.48 / 28.05 px·m⁻¹ | 同一行上的两个方向相差 3.4 倍 |
| 区域内可见路面纵深 | 约 25.7–40.7 m | 由 v 反解得到 |

![标定结果](images/fig03_t2_calibration.png)

**Fig.3** (a) 着色区域、车道边界与计数线；(b) 2 m × 2 m 路面网格反投影

The anisotropy in Tab.2 is worth a comment. At the same image row the vertical scale is 3.4 times coarser than the horizontal one, so a single pixel of jitter on the bottom edge of a bounding box corresponds to about 3.6 cm of longitudinal error. This is precisely why the tracker filters in metres and why speed is regressed over the whole trajectory rather than differenced frame to frame: the observation noise is strongly anisotropic, and averaging is the only way to suppress it.

Counting results are given in Tab.3. The video is nominally 378 frames at 25 fps, but only 328 frames decode; the file is truncated and the statistics therefore cover 13.12 s.

**Tab.3** Counting and speed summary

| 指标 | 数值 |
|------|------|
| 可解码帧数 / 时长 | 328 帧 / 13.12 s |
| 确认航迹数（区域内出现的不同车辆） | 18 |
| **越线计数** | **11** |
| 其中短航迹（< 10 帧） | 2 |
| 给出速度估计的车辆数 | 9 |
| 平均 / 中位速度 | 44.27 / 44.59 km·h⁻¹ |
| 速度标准差 / 范围 | 14.08 / 22.65–65.13 km·h⁻¹ |
| 处理速度 | 34 fps（单线程 CPU，约 0.4 倍实时） |

The difference between 18 confirmed tracks and 11 counted vehicles is not an error and should be read carefully: the video begins with several vehicles already inside the tint area, so they never cross the counting line. Both numbers are reported so that the discrepancy is visible rather than silently reconciled.

### 4.3 Task 2: speed estimation

Speeds are given per vehicle in Tab.4 and Fig.4; Fig.5 shows the cumulative count and the time interval during which each track was present.

**Tab.4** Per-vehicle speed estimates

| 航迹 | 速度/(km·h⁻¹) | 有效样本 | 拟合 R² |
|------|---------------|----------|---------|
| #5 | 39.19 | 27 | 0.981 |
| #7 | 37.39 | 28 | 0.971 |
| #10 | 53.24 | 15 | 0.936 |
| #12 | 44.59 | 24 | 0.511 |
| #13 | 61.89 | 22 | 0.384 |
| #15 | 29.50 | 20 | 0.884 |
| #16 | 22.65 | 23 | 0.786 |
| #19 | 44.88 | 21 | 0.979 |
| #20 | 65.13 | 15 | 0.692 |

![速度分布](images/fig05_t2_speed.png)

**Fig.4** (a) 越线车辆的速度分布；(b) 逐车速度

![计数曲线与航迹时段](images/fig06_t2_timeline.png)

**Fig.5** 上：累计计数曲线；下：各航迹在着色区域内出现的时段

Two of the nine fits, #12 and #13, have $R^2$ below 0.55, which means the constant-velocity assumption explains less than half of the variance of their trajectories. Both are short-lived tracks that were frequently clipped by the boundary of the tint area, so a large fraction of their observations were marked invalid and the surviving samples are concentrated near the entry region, where the longitudinal scale is coarsest. Their speeds (44.59 and 61.89 km/h) should therefore be read as indicative only. Excluding them would raise the mean slightly and reduce the standard deviation from 14.08 to about 9 km/h, but no vehicles were excluded from Tab.3, because removing the difficult cases after the fact would misrepresent the method's actual reliability.

The spread in Tab.4 is physically plausible for a congested elevated expressway with a work zone on the right-hand side: the slowest vehicle (#16, 22.65 km/h) and the fastest (#20, 65.13 km/h) co-exist in the same 13-second window, which is consistent with a free-flowing lane adjacent to a queuing lane.

![鸟瞰图轨迹](images/fig04_t2_bev.png)

**Fig.6** 鸟瞰（逆透视）视图中的车辆轨迹，颜色编码车速

### 4.4 Sensitivity to the calibration prior

Because the absolute speed scale depends on the single assumption discussed in Section 3.2.3, the experiment was repeated with the field of view forced to 29° and 36° instead of the inferred 32°. Results are in Tab.5.

**Tab.5** Sensitivity of the results to the assumed field of view

| 假设水平视场角 | 29° | 32°（默认） | 36° |
|----------------|-----|-------------|-----|
| 平均速度/(km·h⁻¹) | 46.2 | 44.3 | 39.5 |
| 越线计数/辆 | 11 | 11 | 12 |

The mean speed changes by about ±13 % across this range, whereas the count is stable at 11–12. This is the correct way to read the results of Section 4.3: **the count is calibration-free** — it depends only on the sign change of the crossing condition — and the **relative** speeds (which vehicle is faster than which) are also calibration-free, because all speeds are scaled by the same factor. Only the absolute values in km/h inherit the uncertainty.

### 4.5 Limitations

Three limitations are stated explicitly rather than assumed away.

1. **No speed ground truth.** The experiment had no access to loop-detector or radar measurements for this road segment, so the speed errors reported above are internal consistency measures (fit quality), not accuracy measures against a reference. A genuine validation would require at least one vehicle of known speed.
2. **No stitching ground truth.** The RMSE in Tab.1 is the reprojection error of the inlier correspondences, which measures the goodness of fit of the homography, not the geometric error of the mosaic. The latter would require surveyed control points.
3. **Truncated video.** Twelve percent of the nominal frames are undecodable, so the counting window is shorter than intended and vehicles that would have crossed after 13.12 s are not counted. Assuming a stationary arrival rate, the count would rise by roughly one vehicle per additional second of footage.

## 5 Conclusions

This report presented two geometry-driven studies.

For stitching, a pipeline of SIFT matching, ratio testing, self-implemented RANSAC with normalised DLT and distance-transform feathering stitched all four pairs with inlier ratios of 80–97 % and sub-pixel reprojection error (0.33–0.72 px) in 43–250 ms. The experiments support three specific conclusions: that the inlier ratio rather than the raw match count determines success; that Hartley normalisation is not optional, since the design matrix is otherwise badly conditioned; and that adaptive RANSAC termination reduces the cost by roughly two orders of magnitude at no cost in accuracy.

For the traffic task, the decisive choice was to perform detection, filtering and association in metric road coordinates derived from the lane geometry, rather than in the image plane. A constant-velocity Kalman filter is exact in those coordinates and only approximately valid in pixels, and the resulting tracker counted 11 vehicles out of 18 confirmed tracks and estimated nine speeds with a mean of 44.27 km/h at 34 fps on one CPU core. The calibration analysis separated what the image geometry determines (lateral scale and camera height, cross-validated on multiple rows) from what it does not (the longitudinal scale), and the sensitivity study bounded the remaining ambiguity at about ±13 % of the absolute speed while leaving the count unaffected.

Both studies point to the same methodological lesson: for classical geometric problems, writing the geometry down explicitly buys interpretability, a cheap consistency check and an honest error budget — the camera-height agreement in Section 4.2 and the $R^2$ column in Tab.4 are both examples of the method auditing itself.

## References

1. Brown M, Lowe D G. Automatic panoramic image stitching using invariant features[J]. International Journal of Computer Vision, 2007, 74(1): 59-73.
2. Lowe D G. Distinctive image features from scale-invariant keypoints[J]. International Journal of Computer Vision, 2004, 60(2): 91-110.
3. Bay H, Ess A, Tuytelaars T, et al. Speeded-up robust features (SURF)[J]. Computer Vision and Image Understanding, 2008, 110(3): 346-359.
4. Rublee E, Rabaud V, Konolige K, et al. ORB: an efficient alternative to SIFT or SURF[C]//Proceedings of the 2011 IEEE International Conference on Computer Vision (ICCV). Barcelona: IEEE, 2011: 2564-2571.
5. Hartley R, Zisserman A. Multiple view geometry in computer vision[M]. 2nd ed. Cambridge: Cambridge University Press, 2004.
6. Fischler M A, Bolles R C. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography[J]. Communications of the ACM, 1981, 24(6): 381-395.
7. Hartley R I. In defense of the eight-point algorithm[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1997, 19(6): 580-593.
8. Burt P J, Adelson E H. A multiresolution spline with application to image mosaics[J]. ACM Transactions on Graphics, 1983, 2(4): 217-236.
9. Stauffer C, Grimson W E L. Adaptive background mixture models for real-time tracking[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Fort Collins: IEEE, 1999, 2: 246-252.
10. Zivkovic Z. Improved adaptive Gaussian mixture model for background subtraction[C]//Proceedings of the 17th International Conference on Pattern Recognition (ICPR). Cambridge: IEEE, 2004, 2: 28-31.
11. Kalman R E. A new approach to linear filtering and prediction problems[J]. Journal of Basic Engineering, 1960, 82(1): 35-45.
12. Kuhn H W. The Hungarian method for the assignment problem[J]. Naval Research Logistics Quarterly, 1955, 2(1-2): 83-97.
13. Bewley A, Ge Z, Ott L, et al. Simple online and realtime tracking[C]//Proceedings of the 2016 IEEE International Conference on Image Processing (ICIP). Phoenix: IEEE, 2016: 3464-3468.
14. Coifman B, Beymer D, McLauchlan P, et al. A real-time computer vision system for vehicle tracking and traffic surveillance[J]. Transportation Research Part C: Emerging Technologies, 1998, 6(4): 271-288.
15. Dailey D J, Cathey F W, Pumrin S. An algorithm to estimate mean traffic speed using uncalibrated cameras[J]. IEEE Transactions on Intelligent Transportation Systems, 2000, 1(2): 98-107.
16. Cathey F W, Dailey D J. A novel technique to dynamically measure vehicle speed using uncalibrated roadway cameras[C]//Proceedings of the IEEE Intelligent Vehicles Symposium (IV). Las Vegas: IEEE, 2005: 777-782.
