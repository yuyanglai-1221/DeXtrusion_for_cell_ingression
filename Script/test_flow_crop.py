# test_flow_crop.py
import os
import numpy as np
import tifffile
from dextrusion.RoiUtils import read_rois
from flow_gen import (
    MovieGeneratorFromROI,
    FlowFollowingMovieGeneratorFromROI,
)

def load_velocity_and_centers_from_mat(mat_path,
                                       u_name='u_new', v_name='v_new',
                                       x_name='x_new', y_name='y_new',
                                       fillna=0.0):
    """
    Read MATLAB .mat files: supports v7 (scipy.io) and v7.3 (mat73 or hdf5storage).
    u/v/x/y can be (T,H,W) numpy arrays or per-frame cell/list.

    Returns:
      velocity: (T,H,W,2), [...,0]=vy, [...,1]=vx
      centers : (T,2) per-frame global center (y,x)
    """
    def _load_mat_any(path):
        # Try scipy (v7). For v7.3 it raises NotImplementedError.
        try:
            import scipy.io as sio
            return sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            pass
        # v7.3: try mat73 first
        try:
            import mat73
            return mat73.loadmat(path)
        except Exception:
            pass
        # Fallback: hdf5storage
        try:
            import hdf5storage
            return hdf5storage.loadmat(path)
        except Exception as e:
            raise RuntimeError(
                "Failed to read .mat (v7.3). Please install `mat73` or `hdf5storage`."
            ) from e

    mat = _load_mat_any(mat_path)

    def _asarray(x):
        return np.asarray(x)

    def stack_cells(obj):
        """
        Normalize (T,H,W) arrays or lists/cells of T 2D matrices or T scalars
        into numpy arrays:
          - returns (T,H,W) or (T,) as float32
        """
        # list/tuple (common for v7.3 cell arrays)
        if isinstance(obj, (list, tuple)):
            T = len(obj)
            first = _asarray(obj[0])
            if first.ndim == 0:
                out = np.empty((T,), dtype=np.float32)
                for t in range(T):
                    out[t] = float(_asarray(obj[t]))
                return out
            elif first.ndim == 2:
                H, W = first.shape
                out = np.empty((T, H, W), dtype=np.float32)
                for t in range(T):
                    out[t] = _asarray(obj[t]).astype(np.float32)
                return out
            else:
                raise ValueError("Each cell must be a scalar or a 2D matrix.")

        # numpy array
        arr = np.asarray(obj)
        if arr.dtype == object:
            # expand object (cell) arrays then recurse
            seq = [np.asarray(a) for a in arr.ravel()]
            return stack_cells(seq)
        else:
            arr = arr.astype(np.float32)
            if arr.ndim in (1, 3):
                return arr
            raise ValueError(f"Unsupported array dims: {arr.shape}")

    # Extract variables (change names if your .mat uses different keys)
    try:
        u = stack_cells(mat[u_name])  # (T,H,W) or (T,)
        v = stack_cells(mat[v_name])
    except KeyError as e:
        raise KeyError(
            f"Key {e.args[0]} not found in {mat_path}. Print mat.keys() to verify names."
        )

    # Assemble velocity (vy, vx)
    velocity = np.empty(u.shape + (2,), dtype=np.float32)
    velocity[..., 0] = v  # vy
    velocity[..., 1] = u  # vx
    if np.isnan(velocity).any():
        velocity = np.where(np.isnan(velocity), fillna, velocity)

    # centers: from x/y (either per-frame scalar centers or per-frame HxW grids)
    try:
        xarr = stack_cells(mat[x_name])
        yarr = stack_cells(mat[y_name])
    except KeyError:
        # If x/y missing: fall back to zeros (not recommended, but allows running)
        T = velocity.shape[0]
        centers = np.zeros((T, 2), dtype=np.float32)
        return velocity, centers

    if xarr.ndim == 1 and yarr.ndim == 1:
        centers = np.stack([yarr, xarr], axis=-1).astype(np.float32)  # (T,2)
    elif xarr.ndim == 3 and yarr.ndim == 3:
        T = xarr.shape[0]
        centers = np.empty((T, 2), dtype=np.float32)
        for t in range(T):
            xg = xarr[t]; yg = yarr[t]
            xc = np.nanmedian(xg) if np.isfinite(xg).any() else (xg.shape[1]-1)/2.0
            yc = np.nanmedian(yg) if np.isfinite(yg).any() else (yg.shape[0]-1)/2.0
            centers[t, 0] = yc
            centers[t, 1] = xc
    else:
        raise ValueError("x_new/y_new dims mismatch: both must be (T,) or both (T,H,W).")

    return velocity, centers


# ======== Paths ========
data_path = "/Users/yuyangsmacbook/project/dextrusion-main/Script"  # folder containing movie.tif & zip
movie_tif = os.path.join(data_path, "049_100.tif")
roi_zip   = os.path.join(data_path, "_nothing.zip")   # change to other category zip if needed
mat_path  = "/Users/yuyangsmacbook/project/dextrusion-main/Script/PIV_049_100.mat"  # your .mat

frame_shape = (5,5)
win_halfsize = (25,25)   # spatial half-window
vel_scale = 1.0          # if u/v are already in pixel/frame, keep 1.0

# ======== Read image, ROIs, velocity ========
img = tifffile.imread(movie_tif)  # (T, H, W)
rois = read_rois(roi_zip)         # [(z,y,x), ...]
velocity, centers = load_velocity_and_centers_from_mat(mat_path, fillna=0.0)  # (T,H,W,2), (T,2)

print("movie shape:", img.shape)
print("velocity shape:", velocity.shape, "centers shape:", centers.shape)
print("num rois:", len(rois))

# ======== Output folders ========
fixed_dir = os.path.join(data_path, "fixed_crops3")
flow_dir  = os.path.join(data_path, "flow_crops3")
os.makedirs(fixed_dir, exist_ok=True)
os.makedirs(flow_dir,  exist_ok=True)

# ======== Helper: normalize to uint8 ========
def to_uint8(a):
    a = a.astype(np.float32)
    a -= a.min()
    m = a.max()
    if m > 0:
        a /= m
    return (a * 255).astype(np.uint8)

# ======== Generators (fixed-crop and flow-following) ========
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

flow_gen = FlowFollowingMovieGeneratorFromROI(
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
    verbose=False,
    velocity=velocity,
    centers=centers,
    vel_scale=vel_scale,
    interp_order=1,
    border_mode='reflect'
)

# ======== Batch export (iterate all ROIs; customize indices if needed) ========
roi_indices = range(len(rois))  # e.g. [190, 191, 192]

n_ok = n_skip = 0
for i in roi_indices:
    z, y, x = rois[i]

    # fixed crop
    fixed = MovieGeneratorFromROI.get_roi_img(base_gen, z, y, x, img)
    if fixed is None:
        print(f"[skip fixed] roi#{i} at (t={z}, y={y}, x={x}) out of bounds")
        n_skip += 1
        continue

    # flow-following crop
    flow = flow_gen.get_roi_img(z, y, x, img)
    if flow is None:
        print(f"[skip flow ] roi#{i} at (t={z}, y={y}, x={x}) out of bounds / velocity time insufficient")
        n_skip += 1
        continue

    # unified filename (use 1-based t for Fiji friendliness)
    fname = f"roi{i:04d}_t{z+1:04d}_y{y}_x{x}.tif"
    tifffile.imwrite(os.path.join(fixed_dir, fname), to_uint8(fixed), imagej=True)
    tifffile.imwrite(os.path.join(flow_dir,  fname), to_uint8(flow),  imagej=True)
    n_ok += 1

print(f"Done. saved={n_ok}, skipped={n_skip}")
print("Fixed crops dir:", fixed_dir)
print("Flow  crops dir:", flow_dir)
