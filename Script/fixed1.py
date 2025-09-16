# test_fixed_crop_from_names.py
import os, re
import numpy as np
import tifffile
from zipfile import ZipFile
from flow_gen import MovieGeneratorFromROI   # 只用 get_roi_img

# ======== 配置 ========
data_path = "/Users/yuyangsmacbook/project/dextrusion-main/Script"
movie_tif = os.path.join(data_path, "center.tif")
roi_zip   = os.path.join(data_path, "FN.zip")

# 固定裁剪窗口
frame_shape  = (5, 25)   # (过去帧数, 未来帧数)
win_halfsize = (25, 25)  # (半高, 半宽)

# 文件名解析：z-y-x 或 z_y_x
NAME_RE = re.compile(r".*?(\d+)[-_](\d+)[-_](\d+)\.roi$", re.IGNORECASE)
Z_IS_1_BASED = True   # 文件名里的 z 是否从 1 开始计数（常见）。如不需要减 1，改为 False

# ======== 工具：归一化到 uint8 ========
def to_uint8(a):
    a = a.astype(np.float32)
    a -= a.min()
    m = a.max()
    if m > 0:
        a /= m
    return (a * 255).astype(np.uint8)

# ======== 读取影像 ========
img = tifffile.imread(movie_tif)  # (T,H,W)
print("movie shape:", img.shape)

# ======== 列出 zip 里的 ROI 名称 ========
with ZipFile(roi_zip, "r") as zf:
    roi_names = [n for n in zf.namelist() if n.lower().endswith(".roi")]
roi_names = sorted(roi_names, key=lambda s: os.path.basename(s))
print("num rois (by names):", len(roi_names))

# ======== 输出文件夹 ========
fixed_dir = os.path.join(data_path, "FN")
os.makedirs(fixed_dir, exist_ok=True)

# ======== 生成器（仅用于 get_roi_img 逻辑） ========
base_gen = MovieGeneratorFromROI(
    data_path=data_path,
    batch_size=1,
    frame_shape=frame_shape,
    win_halfsize=win_halfsize,
    val_ratio=0.2,
    balance=True,
    ncat=4,
    cat_names=["", "_cell_death.zip", "_cell_sop.zip", "_cell_division.zip"],
    naug=1,
    augment_withnoise=False,
    verbose=False
)

# ======== 主循环：按文件名解析坐标并裁剪 ========
saved = skipped = badname = 0

def try_crop(z_raw, y, x):
    """先按配置的 1-based/0-based 处理 z；若越界失败，再回退试另一种 z。"""
    z0 = z_raw - 1 if Z_IS_1_BASED else z_raw
    crop = MovieGeneratorFromROI.get_roi_img(base_gen, z0, y, x, img)
    if crop is None:
        # 回退路径：尝试不减 1 的 z
        z1 = z_raw if Z_IS_1_BASED else (z_raw - 1)
        crop = MovieGeneratorFromROI.get_roi_img(base_gen, z1, y, x, img)
        if crop is None:
            return None, (z0, z1)
        else:
            return crop, (z1,)
    else:
        return crop, (z0,)

for i, name in enumerate(roi_names):
    stem = os.path.basename(name)
    m = NAME_RE.match(stem)
    if not m:
        print(f"[skip bad name] roi#{i} '{stem}' 不符合 <z>-<y>-<x>.roi 或 <z>_<y>_<x>.roi")
        badname += 1
        continue
    z_raw, y, x = map(int, m.groups())

    crop, used_zs = try_crop(z_raw, y, x)
    if crop is None:
        print(f"[skip] roi#{i} '{stem}' (z_raw={z_raw}, y={y}, x={x}) 窗口越界（时间或空间不足）")
        skipped += 1
        continue

    # 用原始 roi 文件名作为输出名（仅改后缀）
    out_name = os.path.splitext(stem)[0] + ".tif"
    tifffile.imwrite(os.path.join(fixed_dir, out_name), to_uint8(crop), imagej=True)
    saved += 1

print(f"Done. saved={saved}, skipped={skipped}, badname={badname}")
print("Fixed crops dir:", fixed_dir)