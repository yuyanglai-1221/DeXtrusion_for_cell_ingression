# test_fixed_crop.py
import os
import numpy as np
import tifffile
from zipfile import ZipFile
from dextrusion.RoiUtils import read_rois
from flow_gen import MovieGeneratorFromROI   # 复用你已有的类，只用其中的 get_roi_img

# ======== 路径 ========
data_path = "/Users/yuyangsmacbook/project/dextrusion-main/Script"  # 放 movie.tif & zip 的文件夹
movie_tif = os.path.join(data_path, "center.tif")
roi_zip   = os.path.join(data_path, "FP.zip")   # 换成你要的类别 zip

# 固定裁剪的时间/空间窗口
frame_shape = (5, 25)   # (过去帧数, 未来帧数) —— 总帧数 = 10+10 = 20
win_halfsize = (50, 50)  # 空间半窗（上下/左右各 25）→ 窗口尺寸 51x51

# ======== 读取影像与 ROI ========
img  = tifffile.imread(movie_tif)        # (T, H, W)
rois = read_rois(roi_zip)                # [(z,y,x), ...]

print("movie shape:", img.shape)
print("num rois:", len(rois))

# （可选）尝试从 zip 中读取各 ROI 文件名，用于更友好的命名
roi_names = None
try:
    with ZipFile(roi_zip, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".roi")]
        # 去掉目录前缀，只保留文件名，并做稳定排序
        roi_names = [os.path.basename(n) for n in sorted(names)]
        if len(roi_names) != len(rois):
            print(f"[warn] zip 中 .roi 数({len(roi_names)}) != 读取到的 ROI 数({len(rois)})，将改用索引命名。")
            roi_names = None
except Exception as e:
    print("[info] 读取 zip 名称失败，使用索引命名；原因：", e)
    roi_names = None

# ======== 输出文件夹 ========
fixed_dir = os.path.join(data_path, "FP")
os.makedirs(fixed_dir, exist_ok=True)

# ======== 工具：归一化到 uint8 ========
def to_uint8(a):
    a = a.astype(np.float32)
    a -= a.min()
    m = a.max()
    if m > 0:
        a /= m
    return (a * 255).astype(np.uint8)

# ======== 只做固定裁剪 ========
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

saved = skipped = 0
for i, (z, y, x) in enumerate(rois):
    crop = MovieGeneratorFromROI.get_roi_img(base_gen, z, y, x, img)
    if crop is None:
        print(f"[skip] roi#{i} (t={z}, y={y}, x={x}) 窗口越界（时间或空间不足）")
        skipped += 1
        continue

    fname = f"z{z+1:04d}_y{y:04d}_x{x:04d}.tif"

    tifffile.imwrite(os.path.join(fixed_dir, fname), to_uint8(crop), imagej=True)
    saved += 1

print(f"Done. saved={saved}, skipped={skipped}")
print("Fixed crops dir:", fixed_dir)
