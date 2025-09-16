#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import tifffile
import roifile
from math import sqrt
from typing import List, Tuple

# 可选：全局最优匹配（匈牙利算法），没有 scipy 会自动回退贪心
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

BIG = 1e6  # 大惩罚（禁止配对）

# ---------- 基础IO ----------

def read_rois(zip_path: str):
    """读取 ImageJ ROI zip，返回 [{'t','y','x','roi','name','idx'} ...]，t 为 0-based 帧号"""
    rois = roifile.ImagejRoi.fromfile(zip_path)
    out = []
    for i, r in enumerate(rois):
        # t 优先用 position / t_position / z_position（ImageJ的1-based）
        t = max(int(r.position or 0), int(r.t_position or 0), int(r.z_position or 0)) - 1
        if t < 0:
            # 退路：从 name 中解析 TTTT-YYYY-XXXX-...
            t = 0
            try:
                parts = (r.name or "").split("-")
                if len(parts) >= 1 and parts[0].isdigit():
                    t = int(parts[0]) - 1
            except Exception:
                pass
        y = int(r.top)
        x = int(r.left)
        out.append({"t": t, "y": y, "x": x, "roi": r, "name": r.name or "", "idx": i})
    return out

def load_probamap(tif_path: str):
    """读取概率图，输出 (T,Y,X)；若是 (Y,X) 则补一帧"""
    arr = tifffile.imread(tif_path)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape {arr.shape}, expected (T,Y,X) or (Y,X)")
    return arr

def sample_prob(prob, t, y, x) -> float:
    """取中心像素概率，超出边界裁剪"""
    t = int(np.clip(t, 0, prob.shape[0] - 1))
    y = int(np.clip(y, 0, prob.shape[1] - 1))
    x = int(np.clip(x, 0, prob.shape[2] - 1))
    return float(prob[t, y, x])

# ---------- 距离/约束 ----------

def spatial_dist(a, b) -> float:
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return sqrt(dx * dx + dy * dy)

def within_threshold(a, b, dxy: float, dt: int) -> bool:
    return (abs(a["t"] - b["t"]) <= dt) and (spatial_dist(a, b) <= dxy)

# ---------- 匹配策略 ----------

def build_cost_matrix(preds, gts, dxy: float, dt: int) -> np.ndarray:
    """代价 = dxy + |dt|；超阈值则置为 BIG 禁止匹配"""
    C = np.full((len(preds), len(gts)), BIG, dtype=float)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            dt_ = abs(p["t"] - g["t"])
            if dt_ <= dt:
                dxy_ = spatial_dist(p, g)
                if dxy_ <= dxy:
                    C[i, j] = sqrt((dxy_/dxy)**2 + (dt_/dt)**2)
    return C

def match_hungarian(preds, gts, dxy: float, dt: int):
    """匈牙利算法全局最优一对一匹配，返回：匹配对列表、未匹配pred索引、未匹配gt索引"""
    C = build_cost_matrix(preds, gts, dxy, dt)
    if C.size == 0:
        return [], list(range(len(preds))), list(range(len(gts)))
    row_ind, col_ind = linear_sum_assignment(C)
    pairs = []
    used_pred = set()
    used_gt = set()
    for r, c in zip(row_ind, col_ind):
        if C[r, c] < BIG:
            pairs.append((r, c))
            used_pred.add(r)
            used_gt.add(c)
    unmatched_pred = [i for i in range(len(preds)) if i not in used_pred]
    unmatched_gt = [j for j in range(len(gts)) if j not in used_gt]
    return pairs, unmatched_pred, unmatched_gt

def match_greedy(preds, gts, dxy: float, dt: int):
    """贪心：按最小代价逐个占位，非全局最优（与旧实现风格相近）"""
    C = build_cost_matrix(preds, gts, dxy, dt)
    pairs = []
    used_pred = set()
    used_gt = set()
    # 展平候选 (i,j,cost) 并排序
    cand = [(i, j, C[i, j]) for i in range(C.shape[0]) for j in range(C.shape[1]) if C[i, j] < BIG]
    cand.sort(key=lambda x: x[2])
    for i, j, _ in cand:
        if i not in used_pred and j not in used_gt:
            pairs.append((i, j))
            used_pred.add(i)
            used_gt.add(j)
    unmatched_pred = [i for i in range(len(preds)) if i not in used_pred]
    unmatched_gt = [j for j in range(len(gts)) if j not in used_gt]
    return pairs, unmatched_pred, unmatched_gt

# ---------- 导出 ----------

def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

import roifile

def roi_with_name(r, name):
    # 优先用 frombytes/tobytes 克隆（roifile 支持）
    try:
        r2 = roifile.ImagejRoi.frombytes(r.tobytes())
    except Exception:
        # 兜底：手动拷字段（字段可能随 roifile 版本略有差异）
        r2 = roifile.ImagejRoi()
        r2.version = getattr(r, "version", 227)
        r2.roitype = r.roitype
        r2.n_coordinates = getattr(r, "n_coordinates", 1)
        r2.left = r.left
        r2.top = r.top
        r2.position = getattr(r, "position", 0)
        r2.t_position = getattr(r, "t_position", 0)
        r2.z_position = getattr(r, "z_position", 0)
        r2.c_position = getattr(r, "c_position", 0)
        r2.integer_coordinates = getattr(r, "integer_coordinates", None)
        r2.stroke_width = getattr(r, "stroke_width", 1)
        r2.stroke_color = getattr(r, "stroke_color", None)
    r2.name = name
    return r2


def save_zip(path: str, rois_list):
    if rois_list:
        roifile.roiwrite(path, rois_list, mode="w")

def write_csv(path: str, header: List[str], rows: List[List]):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(description="Audit matching between predicted and GT ROIs with global optimal (Hungarian) or greedy fallback.")
    ap.add_argument("--pred", required=False, help="Predicted ROI zip path.")
    ap.add_argument("--gt", required=False, help="Ground-truth ROI zip path.")
    ap.add_argument("--proba", required=False, help="(Optional) probability map TIFF (T,Y,X or Y,X).")
    ap.add_argument("--outdir", required=False, default="./audit_out", help="Output directory.")
    ap.add_argument("--dxy", type=float, default=10.0, help="Max XY distance in pixels. [default: 10]")
    ap.add_argument("--dt", type=int, default=4, help="Max time distance in frames. [default: 4]")
    ap.add_argument("--strategy", choices=["auto", "hungarian", "greedy"], default="auto",
                    help="'auto' uses Hungarian if available, else greedy. [default: auto]")
    ap.add_argument("--annotate", action="store_true", help="Append details (dist/prob) to ROI names in exported zips.")
    # 允许无参数时在代码里硬编码
    args = ap.parse_args()

    # 如果没传参数，这里可以手动填（方便你“直接在代码里面输入”）
    if not args.pred or not args.gt:
        # ←←← 在这里把路径填上即可：
        args.pred = "/Users/yuyangsmacbook/Desktop/results/retrain468_pure_0/img_0001_0050_center_2048x1024_cell_death.zip"
        args.gt   = "/Users/yuyangsmacbook/Desktop/img_0001_0050_center_2048x1024_cell_death.zip"
        args.proba = "/Users/yuyangsmacbook/Desktop/results/retrain468_pure_0/img_0001_0050_center_2048x1024_cell_death.tif"
        args.outdir = "/Users/yuyangsmacbook/Desktop/check_by_myself/model468_pure_0"
        # ↑↑↑ 改成你的绝对路径；若没概率图可置 None 或删掉此行

    ensure_dir(args.outdir)

    preds = read_rois(args.pred)
    gts = read_rois(args.gt)

    prob = None
    if args.proba and os.path.exists(args.proba):
        prob = load_probamap(args.proba)

    # 采样概率
    if prob is not None:
        for r in preds:
            r["prob"] = sample_prob(prob, r["t"], r["y"], r["x"])
    else:
        for r in preds:
            r["prob"] = np.nan

    # 选择策略
    use_hungarian = (args.strategy == "hungarian") or (args.strategy == "auto" and linear_sum_assignment is not None)
    if use_hungarian:
        pairs, unp, ung = match_hungarian(preds, gts, args.dxy, args.dt)
        mode = "hungarian"
    else:
        pairs, unp, ung = match_greedy(preds, gts, args.dxy, args.dt)
        mode = "greedy"

    TP = len(pairs); FP = len(unp); FN = len(ung)
    precision = TP/(TP+FP) if (TP+FP)>0 else 0.0
    recall    = TP/(TP+FN) if (TP+FN)>0 else 0.0

    print(f"Strategy: {mode}")
    print(f"TP={TP}, FP={FP}, FN={FN}, Precision={precision:.4f}, Recall={recall:.4f}")

    # --- 导出 CSV ---
    match_rows = []
    for i, j in pairs:
        p = preds[i]; g = gts[j]
        dt = abs(p["t"] - g["t"])
        dxy = spatial_dist(p, g)
        cost = dxy + dt
        match_rows.append([
            i, p["name"], p["t"], p["y"], p["x"], (None if np.isnan(p["prob"]) else int(round(p["prob"]))),
            j, g["name"], g["t"], g["y"], g["x"], f"{dxy:.3f}", dt, f"{cost:.3f}"
        ])
    write_csv(
        os.path.join(args.outdir, "matches.csv"),
        ["pred_idx","pred_name","pred_t","pred_y","pred_x","pred_prob",
         "gt_idx","gt_name","gt_t","gt_y","gt_x","dist_xy","dist_t","cost"],
        match_rows
    )

    fp_rows = []
    fp_rois = []
    for i in unp:
        p = preds[i]
        fp_rows.append([i, p["name"], p["t"], p["y"], p["x"], (None if np.isnan(p["prob"]) else int(round(p["prob"])))])
        newname = p["name"]
        if args.annotate:
            pref = f"{p['t']+1:04d}-{p['y']:04d}-{p['x']:04d}"
            newname = f"{pref}-FP-{'' if np.isnan(p['prob']) else int(round(p['prob']))}"
        fp_rois.append(roi_with_name(p["roi"], newname))
    write_csv(
        os.path.join(args.outdir, "false_pos.csv"),
        ["pred_idx","pred_name","pred_t","pred_y","pred_x","pred_prob"],
        fp_rows
    )

    fn_rows = []
    fn_rois = []
    for j in ung:
        g = gts[j]
        fn_rows.append([j, g["name"], g["t"], g["y"], g["x"]])
        newname = g["name"]
        if args.annotate:
            pref = f"{g['t']+1:04d}-{g['y']:04d}-{g['x']:04d}"
            newname = f"{pref}-FN"
        fn_rois.append(roi_with_name(g["roi"], newname))
    write_csv(
        os.path.join(args.outdir, "false_neg.csv"),
        ["gt_idx","gt_name","gt_t","gt_y","gt_x"],
        fn_rows
    )

    # --- 导出 ROI zip（方便在 Fiji 里复核） ---
    # 1) 匹配到的对：各导出一份（pred与gt）
    matched_pred_rois = []
    matched_gt_rois = []
    for i, j in pairs:
        p = preds[i]; g = gts[j]
        pn = p["name"]; gn = g["name"]
        if args.annotate:
            dt = abs(p["t"] - g["t"]); dxy = spatial_dist(p, g)
            prob_s = "" if np.isnan(p["prob"]) else f"-P{int(round(p['prob']))}"
            pn = f"{p['t']+1:04d}-{p['y']:04d}-{p['x']:04d}-MATCH-d{int(round(dxy))}-t{dt}{prob_s}"
            gn = f"{g['t']+1:04d}-{g['y']:04d}-{g['x']:04d}-MATCH"
        matched_pred_rois.append(roi_with_name(p["roi"], pn))
        matched_gt_rois.append(roi_with_name(g["roi"], gn))

    save_zip(os.path.join(args.outdir, "matched_pred.zip"), matched_pred_rois)
    save_zip(os.path.join(args.outdir, "matched_gt.zip"),   matched_gt_rois)
    save_zip(os.path.join(args.outdir, "false_pos.zip"),    fp_rois)
    save_zip(os.path.join(args.outdir, "false_neg.zip"),    fn_rois)

    # --- 保存汇总 ---
    summary_path = os.path.join(args.outdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Strategy: {mode}\n")
        f.write(f"Pred: {args.pred}\nGT:   {args.gt}\nProba: {args.proba or 'None'}\n")
        f.write(f"dxy={args.dxy}, dt={args.dt}\n")
        f.write(f"TP={TP}, FP={FP}, FN={FN}\nPrecision={precision:.6f}\nRecall={recall:.6f}\n")
    print(f"Outputs saved to: {args.outdir}")

if __name__ == "__main__":
    main()