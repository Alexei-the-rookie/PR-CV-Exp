"""
run_task2.py —— Task2：视频中指定区域内的车辆计数与测速

流程
----
  1. 由车道几何构造"着色区域 (tint area)" 与 计数线
  2. MOG2 背景建模检测车辆 → 形态学 → 连通域 → 粘连拆分
  3. 卡尔曼 + 匈牙利匹配多目标跟踪（在世界米制坐标中跟踪）
  4. 越线计数 + 轨迹稳健回归测速
  5. 输出：标注视频、轨迹图（鸟瞰）、速度分布、逐车明细、汇总指标

运行：
    python3 src/run_task2.py                 # 全流程（含输出视频）
    python3 src/run_task2.py --no-video      # 跳过视频（更快）

输出：
    results/video/annotated.mp4
    results/figures/*.png
    results/metrics/tracks.csv, summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calib as calib_mod          # noqa: E402
from pipeline import (Detector, DetectorConfig, MultiObjectTracker,   # noqa: E402
                      TrackerConfig)

PROJ = Path(__file__).resolve().parents[1]
VIDEO = PROJ.parent / "Task2" / "video1.mp4"
RES = PROJ / "results"
FIG = RES / "figures"
MET = RES / "metrics"
VIDOUT = RES / "video"

# ---- 区域与计数线设置 ----
ROI_TOP = 620          # 着色区域上边界（图像行）
ROI_BOT = 1078         # 下边界
COUNT_ROW = 900        # 计数线所在行

COLOR_TINT = (120, 200, 0)      # BGR
COLOR_BOX = (0, 235, 0)
COLOR_ID = (0, 255, 255)
COLOR_LINE = (0, 140, 255)


# ==========================================================================
def build_roi(cal: calib_mod.Calib, shape=(1080, 1920)):
    """由左右车道边界线构造着色区域多边形与掩膜。"""
    a1, b1 = cal.p.line_a
    ar, br = cal.p.line_right
    poly = np.array([
        [int(a1 * ROI_TOP + b1), ROI_TOP],
        [int(ar * ROI_TOP + br), ROI_TOP],
        [int(ar * ROI_BOT + br), ROI_BOT],
        [int(a1 * ROI_BOT + b1), ROI_BOT],
    ], dtype=np.int32)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return poly, mask


def draw_tint(frame, poly, alpha=0.18):
    """在画面上叠加半透明的"着色区域"。"""
    ov = frame.copy()
    cv2.fillPoly(ov, [poly], COLOR_TINT)
    return cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0)


# ==========================================================================
def analyze(cal, write_video: bool, verbose: bool = True):
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    poly, roi_mask = build_roi(cal, (H, W))
    det = Detector(DetectorConfig(), roi_mask, cal)
    trk = MultiObjectTracker(TrackerConfig(), cal, fps)

    writer = None
    if write_video:
        VIDOUT.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(VIDOUT / "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # 计数状态
    counted_ids: set[int] = set()
    count_events: list[dict] = []
    frame_idx = 0
    t0 = time.perf_counter()
    snapshot_rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = det.detect(frame)
        tracks = trk.step(dets, frame_idx, img_h=H)
        confirmed = [t for t in tracks if t.confirmed]

        # ---- 越线判定 ----
        for t in confirmed:
            if t.id in counted_ids or len(t.track) < 2:
                continue
            vs = np.array([o[4] for o in t.track])
            fs = np.array([o[0] for o in t.track])
            idx = np.where((vs[:-1] > COUNT_ROW) & (vs[1:] <= COUNT_ROW))[0]
            if len(idx):
                k = int(idx[0])
                # 线性插值求穿越时刻
                v0, v1 = vs[k], vs[k + 1]
                frac = (COUNT_ROW - v0) / (v1 - v0) if abs(v1 - v0) > 1e-9 else 0.0
                t_cross = (fs[k] + frac) / fps
                counted_ids.add(t.id)
                sp = t.speed_kmh(fps)
                count_events.append({
                    "track_id": t.id,
                    "t_cross_s": round(float(t_cross), 3),
                    "frame_cross": int(round(fs[k] + frac)),
                    "speed_kmh_at_cross": round(sp[0], 2) if sp else None,
                })

        # ---- 视频标注 ----
        if writer is not None:
            vis = draw_tint(frame, poly)
            a1, b1 = cal.p.line_a
            ar, br = cal.p.line_right
            cv2.line(vis, (int(a1 * COUNT_ROW + b1), COUNT_ROW),
                     (int(ar * COUNT_ROW + br), COUNT_ROW), COLOR_LINE, 3)
            cv2.putText(vis, f"COUNT LINE  M={len(counted_ids)}",
                        (int(a1 * COUNT_ROW + b1) + 10, COUNT_ROW - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_LINE, 2)
            for t in confirmed:
                x, y, w, h = t.bbox[:4]
                cv2.rectangle(vis, (x, y), (x + w, y + h), COLOR_BOX, 2)
                sp = t.speed_kmh(fps)
                label = f"#{t.id}"
                if sp is not None:
                    label += f"  {sp[0]:.0f} km/h"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
                cv2.rectangle(vis, (x, y - th - 8), (x + tw + 6, y), COLOR_ID, -1)
                cv2.putText(vis, label, (x + 3, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
            cv2.putText(vis, f"frame {frame_idx:03d}   t={frame_idx/fps:5.2f}s   "
                             f"counted={len(counted_ids)}",
                        (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            writer.write(vis)

        if frame_idx % 40 == 0 and len(confirmed):
            snapshot_rows.append((frame_idx, [(t.id, t.bbox) for t in confirmed]))
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    all_tracks = trk.finalize()
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"[分析] {frame_idx} 帧, {fps:.0f} fps, 用时 {elapsed:.1f}s "
              f"({frame_idx/elapsed:.1f} fps 处理速度)")
        print(f"[分析] 确认航迹 {len(all_tracks)} 条, 越线计数 {len(counted_ids)}")

    return {
        "fps": fps, "W": W, "H": H, "n_frames": frame_idx,
        "poly": poly, "tracks": all_tracks, "count_events": count_events,
        "elapsed": elapsed,
    }


# ==========================================================================
def summarize(res, cal):
    fps = res["fps"]
    events = sorted(res["count_events"], key=lambda e: e["t_cross_s"])
    counted_ids = {e["track_id"] for e in events}
    rows = []
    for t in res["tracks"]:
        sp = t.speed_kmh(fps)
        v_all = np.array([o[4] for o in t.track])
        n_valid = sum(1 for o in t.track if o[5])
        r2 = sp[2] if sp else None
        rows.append({
            "track_id": t.id,
            "frames": len(t.track),
            "valid_frames": n_valid,
            "t_start_s": round(t.track[0][0] / fps, 2),
            "t_end_s": round(t.track[-1][0] / fps, 2),
            "v_start": int(round(v_all[0])),
            "v_end": int(round(v_all[-1])),
            "X_start_m": round(t.track[0][1], 2),
            "Z_start_m": round(t.track[0][2], 2),
            "X_end_m": round(t.track[-1][1], 2),
            "Z_end_m": round(t.track[-1][2], 2),
            "speed_kmh": round(sp[0], 2) if sp else "",
            "speed_samples": sp[1] if sp else 0,
            "fit_r2": round(sp[2], 4) if sp else "",
            # 速度是否可靠：只看有效样本数。
            # 不把 R² 当硬门限：匝道下坡段车辆本来就在加速/减速，匀速模型拟合的 R²
            # 会自然偏低（如 0.4），但最小二乘斜率仍然是无偏的"区间平均速度"。
            # R² 作为质量指标一并列出，供人工判断。
            "speed_reliable": bool(sp and sp[1] >= 12),
            "crossed_count_line": t.id in counted_ids,
        })

    counted = [r for r in rows if r["crossed_count_line"]]
    with_speed = [r for r in counted if r["speed_kmh"] != ""]
    reliable = [r for r in counted if r["speed_reliable"]]
    speeds = np.array([r["speed_kmh"] for r in reliable], dtype=float)

    summary = {
        "video": str(VIDEO),
        "fps": fps,
        "n_frames": res["n_frames"],
        "duration_s": round(res["n_frames"] / fps, 2),
        "note_frames": "元数据标称 378 帧(15.12s)，但实际仅 328 帧可解码(13.12s)，"
                       "文件尾部有损坏/截断，全部统计基于可解码的 328 帧。",
        "roi_rows": [ROI_TOP, ROI_BOT],
        "count_line_row": COUNT_ROW,
        "calibration": cal.summary(),
        "n_tracks_confirmed": len(res["tracks"]),
        "n_vehicles_counted": len(counted),
        "n_counted_short_track": sum(1 for r in counted if r["frames"] < 10),
        "n_counted_with_speed": len(with_speed),
        "n_counted_speed_reliable": len(reliable),
        "count_events": events,
        "speed_stats_kmh": {
            "basis": "仅统计越线且有效样本>=12 的车辆（R² 作为质量指标另列）",
            "n": int(len(speeds)),
            "mean": round(float(speeds.mean()), 2) if len(speeds) else None,
            "median": round(float(np.median(speeds)), 2) if len(speeds) else None,
            "std": round(float(speeds.std(ddof=1)), 2) if len(speeds) > 1 else None,
            "min": round(float(speeds.min()), 2) if len(speeds) else None,
            "max": round(float(speeds.max()), 2) if len(speeds) else None,
        },
        "processing_seconds": round(res["elapsed"], 1),
        "processing_fps": round(res["n_frames"] / res["elapsed"], 1),
    }
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true", help="不输出标注视频")
    ap.add_argument("--hfov", type=float, default=None, help="覆盖视场角(度)做敏感性分析")
    args = ap.parse_args()

    for d in (FIG, MET, VIDOUT):
        d.mkdir(parents=True, exist_ok=True)

    p = calib_mod.CalibParams()
    if args.hfov is not None:
        p.hfov_deg = args.hfov
    cal = calib_mod.Calib(p)
    print("[标定]", json.dumps(cal.summary(), ensure_ascii=False))

    res = analyze(cal, write_video=not args.no_video)
    rows, summary = summarize(res, cal)

    with open(MET / "tracks.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(MET / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    s = summary["speed_stats_kmh"]
    print(f"\n=== 结果 ===")
    print(f"  确认航迹数                : {summary['n_tracks_confirmed']}")
    print(f"  越线车辆数（区域计数）    : {summary['n_vehicles_counted']}"
          f"  （其中短航迹<10帧 {summary['n_counted_short_track']} 条）")
    print(f"  给出速度估计的车辆数      : {summary['n_counted_with_speed']}")
    print(f"  速度可靠的车辆数          : {summary['n_counted_speed_reliable']}")
    if s["n"]:
        print(f"  车速(km/h, n={s['n']})     : 均值 {s['mean']:.1f} / 中位 {s['median']:.1f} "
              f"/ 标准差 {s['std']:.1f} / 范围 {s['min']:.1f}-{s['max']:.1f}")
    print(f"  处理速度                  : {summary['processing_fps']} fps")
    print(f"  输出 -> {RES}")

    try:
        import plots
        plots.make_all(res, rows, summary, cal, FIG)
        print(f"  插图 -> {FIG}")
    except Exception as e:  # 出图失败不应影响主结果
        print(f"  [警告] 出图失败: {e}")


if __name__ == "__main__":
    main()
