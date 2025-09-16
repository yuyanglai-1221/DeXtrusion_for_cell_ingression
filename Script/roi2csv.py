import os, re, zipfile
import numpy as np
import pandas as pd

# ========== 输入输出 ==========
src_zip = "/Users/yuyangsmacbook/Desktop/results/retrain466_mix4/img_0001_0050_center_2048x1024_cell_death.zip"
out_csv = "/Users/yuyangsmacbook/Desktop/results/retrain466_mix4/img_0001_0050_center_2048x1024_cell_death.csv"

# ========== 解析 ROI 文件名：'TTTT-YYYY-XXXX.roi' ==========
def parse_name_triplet_from_filename(fname: str):
    bn = os.path.basename(fname)
    if bn.lower().endswith(".roi"):
        bn = bn[:-4]
    m = re.match(r"^\s*(\d+)-(\d+)-(\d+)", bn)
    if not m:
        return None
    T = int(m.group(1)) - 1  # 1-based -> 0-based (napari 用 0-based 更方便)
    y = int(m.group(2))
    x = int(m.group(3))
    return (T, y, x)

rows = []
with zipfile.ZipFile(src_zip, "r") as zf:
    names = [n for n in zf.namelist() if n.lower().endswith(".roi")]
    names.sort()
    for i, n in enumerate(names):
        trip = parse_name_triplet_from_filename(n)
        if trip is None:
            continue
        t0, y, x = trip
        rows.append([i, float(t0), float(y), float(x)])

df = pd.DataFrame(rows, columns=["index","axis-0","axis-1","axis-2"])
df.to_csv(out_csv, index=False)
print(f"✅ CSV written: {out_csv}  (points: {len(df)})")

# ========== （可选）直接在 napari 里查看，并用 MATLAB 默认配色 ==========
# MATLAB default color order (R2014b+):
MATLAB_COLORS = np.array([
    [0.0000, 0.4470, 0.7410],  # blue
    [0.8500, 0.3250, 0.0980],  # orange
    [0.9290, 0.6940, 0.1250],  # yellow
    [0.4940, 0.1840, 0.5560],  # purple
    [0.4660, 0.6740, 0.1880],  # green
    [0.3010, 0.7450, 0.9330],  # cyan
    [0.6350, 0.0780, 0.1840],  # maroon
], dtype=float)

try:
    import napari

    # napari 需要 NxD 的坐标，按 (T, Y, X)
    coords = df[["axis-0","axis-1","axis-2"]].to_numpy()

    # 颜色：按 T % 7 取 MATLAB 色表，并加上 alpha 通道
    idx = (df["axis-0"].astype(int) % len(MATLAB_COLORS)).to_numpy()
    face_rgb = MATLAB_COLORS[idx]
    face_rgba = np.concatenate([face_rgb, np.ones((len(face_rgb), 1))], axis=1)  # alpha=1

    viewer = napari.Viewer()
    layer = viewer.add_points(
        coords,
        name="events",
        face_color=MATLAB_COLORS[4],
        size=25,                  # 根据图像尺度可调大：如 25/50/100
        symbol='disc',            # 更醒目
        blending='translucent',
    )

    # 如果你想统一一种颜色（例如 MATLAB 蓝），改成：
    # layer.face_color = np.concatenate([np.tile(MATLAB_COLORS[0], (len(coords),1)),
    #                                    np.ones((len(coords),1))], axis=1)

    print("👀 Opening napari…")
    napari.run()

except Exception as e:
    print("ℹ️ Skipped napari preview:", e)
