import sys, os
from zipfile import ZipFile
from dextrusion.RoiUtils import read_rois

def inspect_zip(zip_path):
    if not os.path.isfile(zip_path):
        print(f"[ERR] not found: {zip_path}")
        return
    try:
        rois = read_rois(zip_path)  # list of (t, y, x)
    except Exception as e:
        print(f"[ERR] failed to read {zip_path}: {e}")
        return

    total = len(rois)
    neg = [(i, r) for i, r in enumerate(rois) if r[0] < 0]  # t < 0
    print(f"\n== {os.path.basename(zip_path)} ==")
    print(f"Total ROIs: {total}")
    print(f"t<0 count : {len(neg)}")

    if neg:
        print("Examples of t<0 (up to 10):")
        for i, (z, y, x) in neg[:10]:
            print(f"  #{i:4d}: t={z}, y={y}, x={x}")

    # 额外：显示 zip 内 .roi 文件数量，便于对比
    try:
        with ZipFile(zip_path, "r") as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".roi")]
        if len(names) != total:
            print(f"[warn] .roi files in zip = {len(names)}, read_rois returned = {total}")
    except Exception:
        pass

if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python check_roi_zip.py /path/to/xxx.zip [/path/to/yyy.zip ...]")
        sys.exit(0)
    for p in paths:
        inspect_zip(p)
