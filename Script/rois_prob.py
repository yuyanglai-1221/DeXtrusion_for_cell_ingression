#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import tifffile
import roifile
from math import sqrt

def read_rois(zip_path):
    rois = roifile.ImagejRoi.fromfile(zip_path)
    out = []
    for r in rois:
        t = max(int(r.position or 0), int(r.t_position or 0), int(r.z_position or 0)) - 1
        if t < 0:
            try:
                parts = (r.name or "").split("-")
                t = int(parts[0]) - 1
            except Exception:
                t = 0
        y = int(r.top)
        x = int(r.left)
        out.append({"t": t, "y": y, "x": x, "roi": r, "name": r.name or ""})
    return out

def load_probamap(tif_path):
    arr = tifffile.imread(tif_path)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]  # (1, Y, X)
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape {arr.shape}, expected (T,Y,X) or (Y,X)")
    return arr

def sample_prob(prob, t, y, x):
    t = int(np.clip(t, 0, prob.shape[0] - 1))
    y = int(np.clip(y, 0, prob.shape[1] - 1))
    x = int(np.clip(x, 0, prob.shape[2] - 1))
    return float(prob[t, y, x])

def spatial_dist(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return (dx * dx + dy * dy) ** 0.5

def is_conflict(a, b, dxy, dt):
    return (abs(a["t"] - b["t"]) <= dt) and (spatial_dist(a, b) <= dxy)

def greedy_nms(rois, dxy, dt):
    rois_sorted = sorted(rois, key=lambda r: r["prob"], reverse=True)
    kept = []
    for r in rois_sorted:
        if not any(is_conflict(r, k, dxy, dt) for k in kept):
            kept.append(r)
    return kept

def main():
    ap = argparse.ArgumentParser(
        description="NMS dedup of DeXtrusion ROIs using probability map (keep higher center value)."
    )
    ap.add_argument("--roi",   required=True, help="Input ROI zip (predicted events).")
    ap.add_argument("--proba", required=True, help="Probability map TIFF (e.g. *_cell_death_proba.tif).")
    ap.add_argument("--out",   required=True, help="Output ROI zip after NMS.")
    ap.add_argument("--dxy", type=float, default=10, help="Max XY distance (pixels) to consider same event. [default: 10]")
    ap.add_argument("--dt",  type=int,   default=4,  help="Max time distance (frames) to consider same event. [default: 4]")
    ap.add_argument("--annotate_prob", action="store_true",
                    help="Append probability value to ROI name.")
    args = ap.parse_args()

    # 小检查：必须是存在的绝对路径
    for label, p in (("ROI", args.roi), ("PROBA", args.proba)):
        if not os.path.isabs(p):
            raise SystemExit(f"❌ {label} path is not absolute: {p}")
        if not os.path.exists(p):
            raise SystemExit(f"❌ {label} file not found: {p}")
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"ROI  : {args.roi}")
    print(f"PROBA: {args.proba}")
    print(f"OUT  : {args.out}")

    rois = read_rois(args.roi)
    prob = load_probamap(args.proba)

    for r in rois:
        r["prob"] = sample_prob(prob, r["t"], r["y"], r["x"])
    kept = greedy_nms(rois, dxy=args.dxy, dt=args.dt)

    print(f"Total ROIs: {len(rois)}")
    print(f"Kept after NMS: {len(kept)}")
    print(f"Removed as duplicates: {len(rois) - len(kept)}")

    if args.annotate_prob:
        for r in kept:
            try:
                parts = (r["name"] or "").split("-")
                if len(parts) >= 3 and parts[0].isdigit():
                    prefix = "-".join(parts[:3])
                else:
                    prefix = f"{r['t']+1:04d}-{r['y']:04d}-{r['x']:04d}"
            except Exception:
                prefix = f"{r['t']+1:04d}-{r['y']:04d}-{r['x']:04d}"
            r["roi"].name = f"{prefix}-0-{int(round(r['prob']))}"

    roifile.roiwrite(args.out, [r["roi"] for r in kept], mode="w")
    print(f"✅ Saved: {args.out}")

if __name__ == "__main__":
    main()