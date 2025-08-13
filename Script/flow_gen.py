# -*- coding: utf-8 -*-
from tensorflow.keras.utils import Sequence, to_categorical
import os, math, random
import tifffile
from glob import glob
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import exposure
from dextrusion.RoiUtils import read_rois

"""
BSD 3-Clause License (original notice preserved)
"""

# =========================
#  Velocity / centers loader
# =========================
def load_velocity_and_centers_from_mat(mat_path,
                                       u_name='u_new', v_name='v_new',
                                       x_name='x_new', y_name='y_new',
                                       fillna=0.0):
    """
    Read a MATLAB .mat file (each frame as a cell is supported):
      - u_new, v_new: shape (T, H, W) provided either as a cell array or a stacked ndarray
      - x_new, y_new: either per-frame scalars (center) or per-frame HxW grids (global coords)

    Returns:
      velocity: (T, H, W, 2) with [...,0]=vy and [...,1]=vx
      centers : (T, 2) per-frame global center coordinates (y, x)
    """
    import scipy.io as sio
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    def stack_cells(obj_arr):
        # Allow MATLAB cell arrays or already-stacked ndarrays
        arr = np.asarray(obj_arr, dtype=object) if isinstance(obj_arr, np.ndarray) and obj_arr.dtype == object else obj_arr
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            T = arr.shape[0]
            first = np.asarray(arr[0])
            if first.ndim == 0:
                out = np.empty((T,), dtype=np.float32)
                for t in range(T):
                    out[t] = float(np.asarray(arr[t]))
                return out
            elif first.ndim == 2:
                H, W = first.shape
                out = np.empty((T, H, W), dtype=np.float32)
                for t in range(T):
                    out[t] = np.asarray(arr[t], dtype=np.float32)
                return out
            else:
                raise ValueError("Each cell must be a scalar or a 2D array.")
        else:
            arr = np.asarray(obj_arr)
            if arr.ndim == 3:
                return arr.astype(np.float32)
            if arr.ndim == 1:  # 1D sequence of scalars
                return arr.astype(np.float32)
            raise ValueError("Unrecognized structure for x/y/u/v.")

    u = stack_cells(mat[u_name])  # (T,H,W)
    v = stack_cells(mat[v_name])  # (T,H,W)

    velocity = np.empty(u.shape + (2,), dtype=np.float32)
    velocity[..., 0] = v  # vy
    velocity[..., 1] = u  # vx
    if np.isnan(velocity).any():
        velocity = np.where(np.isnan(velocity), fillna, velocity)

    xarr = stack_cells(mat[x_name])
    yarr = stack_cells(mat[y_name])

    if xarr.ndim == 1 and yarr.ndim == 1:
        # per-frame scalar center
        centers = np.stack([yarr, xarr], axis=-1).astype(np.float32)
    elif xarr.ndim == 3 and yarr.ndim == 3:
        # per-frame HxW grids → use median as a robust global center
        T = xarr.shape[0]
        centers = np.empty((T, 2), dtype=np.float32)
        for t in range(T):
            xg = xarr[t]; yg = yarr[t]
            xc = np.nanmedian(xg) if np.isfinite(xg).any() else (xg.shape[1] - 1) / 2.0
            yc = np.nanmedian(yg) if np.isfinite(yg).any() else (yg.shape[0] - 1) / 2.0
            centers[t, 0] = yc
            centers[t, 1] = xc
    else:
        raise ValueError("x_new/y_new dims mismatch: both must be (T,) or both (T,H,W).")

    return velocity, centers  # velocity: (T,H,W,2), centers: (T,2) as (y,x)


# ===============================
#  Original MovieGeneratorFromROI
# ===============================
class MovieGeneratorFromROI(Sequence):
    """Generate small, short windows around ROI positions.

    The generator creates and writes the windows into a temporary folder, which are read
    during training. This avoids loading all data upfront. The temporary folder is removed
    at the end of training.
    """

    def __init__(self,
                 data_path,
                 batch_size=1,
                 frame_shape=(5,5),
                 win_halfsize=(25,25),
                 val_ratio=0.2,
                 balance=True,
                 ncat=3,
                 cat_names=None,
                 naug=1,
                 augment_withnoise=True,
                 add_nothing_windows=10,
                 tmp_directory_path=None,
                 verbose=True,
                 **kwargs):

        self.data_path = data_path
        if data_path is not None:
            if not os.path.isdir(data_path):
                raise Exception("Folder "+data_path+" not found")
            self.dir_list = glob(os.path.join(data_path, '*.tif'))

        if tmp_directory_path is None:
            randid = random.randint(1,1000000)
            self.tempdir = os.path.join(data_path, f"tmp_data_from_roi_{randid}")
        else:
            self.tempdir = tmp_directory_path

        self.frame_shape = frame_shape
        self.half_size = win_halfsize
        self.shape = (int(np.sum(frame_shape)),) + (2*win_halfsize[0]+1, 2*win_halfsize[1]+1)
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.ncat = ncat
        self.augment = naug
        self.augment_noise = augment_withnoise
        self.nothing = 1.2  # slight shift; favor more negatives
        self.add_nothing = add_nothing_windows
        self.balance = balance  # honor the passed-in flag
        self.trainindex = []
        self.valindex = []
        self.verbose = verbose

        if cat_names is None:
            self.catnames = ["", "_cell_death.zip", "_cell_sop.zip", "_cell_division.zip"]
        else:
            self.catnames = cat_names

        _validation = kwargs.get("_validation", None)
        if _validation is None:
            self.initialize()
        else:
            self.catlist = kwargs.get("_cat", None)
            self.filelist = kwargs.get("_files", None)
            self.donoise = kwargs.get("_donoise", None)
            self.trainindex = np.arange(len(self.filelist))

    def initialize(self):
        self.catlist = []
        self.filelist = []
        self.donoise = []
        self.create_lists()
        self.split_val()

    def contain_roi(self, roi, z, y, x):
        """Check if window centered at (z,y,x) contains the given ROI point."""
        if roi[0] < z - self.frame_shape[0] or roi[0] > (z + self.frame_shape[1]):
            return False
        if roi[1] < (y - self.half_size[0]) or roi[1] > (y + self.half_size[0] + 1):
            return False
        return roi[2] > (x - self.half_size[1]) and roi[2] < (x + self.half_size[1] + 1)

    def not_contain_rois(self, rois, z, y, x):
        """Return True if window centered at (z,y,x) contains none of the ROIs in the list."""
        for roi in rois:
            if self.contain_roi(roi, z, y, x):
                return False
        return True

    def split_val(self):
        """Randomly split into training and validation sets (non-overlapping file-level samples)."""
        npts = len(self.catlist)
        indexes = np.arange(npts)
        np.random.shuffle(indexes)

        nbval = int(self.val_ratio * npts)
        val = np.random.permutation(indexes)[:nbval]
        indexes = np.array([i for i in indexes if i not in val])
        self.valindex = val
        self.trainindex = indexes

    def create_lists(self):
        """Scan the folder and list all training windows."""
        if self.verbose:
            print("Found files: ")

        if not os.path.exists(self.tempdir):
            os.makedirs(self.tempdir)

        num = 0
        for imgname in self.dir_list:
            if self.verbose:
                print(imgname)
            img = tifffile.imread(imgname)
            nz, ny, nx = img.shape
            purename = os.path.basename(imgname)
            basename = imgname[:len(imgname)-4]

            # count ROIs per category
            crois = []
            nrois = np.zeros(self.ncat)
            for cat in range(1, self.ncat):
                roifilename = basename + self.catnames[cat]
                if os.path.isfile(roifilename):
                    rois = read_rois(roifilename)
                    crois = crois + rois
                    nrois[cat] = len(rois)

            if self.verbose:
                print("Nrois " + str(max(nrois)))

            # per-category ROIs (+ balancing/augmentation)
            for cat in range(1, self.ncat):
                roifilename = basename + self.catnames[cat]
                if os.path.isfile(roifilename):
                    rois = read_rois(roifilename)
                    if nrois[cat] < 0.75 * max(nrois) and self.balance:
                        nrois[cat] = 0.85 * max(nrois)
                    num = self.add_and_augment_rois(rois, nrois, cat, num, img, purename)

            # Random "Nothing" ROIs
            lim = max(nrois) * self.nothing * self.augment * 3000
            it = 0
            ndata = 0
            target = max(nrois) * self.nothing * self.augment
            while ndata < target and (it < lim):
                it += 1
                x = self.half_size[1] + np.random.randint(nx - 2*self.half_size[1])
                y = self.half_size[0] + np.random.randint(ny - 2*self.half_size[0])
                z = self.frame_shape[0] + np.random.randint(nz - self.frame_shape[1])
                if self.not_contain_rois(crois, z, y, x):
                    rimg = self.get_roi_img(z, y, x, img)
                    if rimg is not None:
                        self.add_point(rimg, 0, num, purename, np.random.uniform(0,1.0) < 0.5)
                        ndata += 1
                        num += 1

            # Explicit "nothing" ROIs from _nothing.zip
            if self.add_nothing > 1:
                num = self.add_nothing_rois(basename, num, img, purename)

    def add_nothing_rois(self, basename, num, img, purename):
        roifilename = basename + "_nothing.zip"
        if os.path.isfile(roifilename):
            rois = read_rois(roifilename)
            for iroi in range(len(rois)):
                z, y, x = self.get_roi_pos(rois, iroi)
                rimg = self.get_roi_img(z, y, x, img)
                if rimg is not None:
                    self.add_point(rimg, 0, num, purename, True)
                    num += 1

            # augmentation
            if self.augment > 1:
                ntarget = len(rois) * self.add_nothing
                ndone = 0
                lim = ntarget * 1000
                it = 0
                while (ndone < ntarget) and (it < lim):
                    it += 1
                    ind = random.randrange(len(rois))
                    z, y, x = self.get_roi_pos(rois, ind)
                    rimg = self.get_roi_img(z, y, x, img)
                    if rimg is not None:
                        self.add_point(rimg, 0, num, purename, False)
                        num += 1
                        ndone += 1
        return num

    def add_and_augment_rois(self, rois, nrois, cat, num, img, purename):
        ncat = 0
        for iroi in range(len(rois)):
            z, y, x = self.get_roi_pos(rois, iroi)
            rimg = self.get_roi_img(z, y, x, img)
            if rimg is not None:
                self.add_point(rimg, cat, num, purename, True)
                num += 1
                ncat += 1

        # augment if necessary
        lim = nrois[cat] * self.augment * 2000
        it = 0
        while (ncat < nrois[cat] * self.augment) and (it < lim):
            it += 1
            ind = random.randrange(len(rois))
            z, y, x = self.get_roi_pos(rois, ind)
            rimg = self.get_roi_img(z, y, x, img)
            if rimg is not None:
                self.add_point(rimg, cat, num, purename, False)
                num += 1
                ncat += 1
        return num

    def get_roi_pos(self, rois, ind):
        """Return a jittered position around the ROI center."""
        roi = rois[ind]
        xroi = roi[2]; yroi = roi[1]; frame = roi[0]
        y = math.floor(yroi + np.random.uniform(-1.0,1.0) * self.half_size[0] * 0.2)
        x = math.floor(xroi + np.random.uniform(-1.0,1.0) * self.half_size[1] * 0.2)
        z = math.floor(frame + np.random.uniform(-1.0,1.0) * 2)
        return (z, y, x)

    def add_point(self, im, cat, num, name, orig):
        outname = os.path.join(self.tempdir, f"cat{cat}_{name}_roi{num}.tif")
        im = np.uint8(self._min_max_scaling(im) * 255)
        tifffile.imwrite(outname, im, imagej=True)
        self.filelist.append(outname)
        self.catlist.append(cat)
        self.donoise.append(False if orig else True)

    def write_img(self, im, ind, num, folder="conv"):
        resdir = os.path.join(self.data_path, f"results_{folder}")
        if not os.path.exists(resdir):
            os.makedirs(resdir)
        fname = os.path.basename(self.get_batch_names(ind)[num])
        outname = os.path.join(resdir, fname)
        im = np.uint8(self._min_max_scaling(im) * 255)
        tifffile.imwrite(outname, im, imagej=True)

    def get_roi_img(self, z, y, x, img):
        """Return the image window around (z,y,x); None if it doesn't fit."""
        if (x-self.half_size[1]) < 0 or (x+self.half_size[1]+1) > img.shape[2]:
            return None
        if (y-self.half_size[0]) < 0 or (y+self.half_size[0]+1) > img.shape[1]:
            return None
        if (z-self.frame_shape[0]) < 0 or (z+self.frame_shape[1]) > img.shape[0]:
            return None
        return img[z-self.frame_shape[0]:z+self.frame_shape[1],
                   y-self.half_size[0]:y+self.half_size[0]+1,
                   x-self.half_size[1]:x+self.half_size[1]+1]

    def get_validation_generator(self):
        return self.__class__(
            data_path=self.data_path,
            batch_size=self.batch_size,
            frame_shape=self.frame_shape,
            win_halfsize=self.half_size,
            val_ratio=self.val_ratio,
            balance=self.balance,
            naug=self.augment,
            augment_withnoise=self.augment_noise,
            verbose=self.verbose,
            ncat=self.ncat,
            tmp_directory_path=self.tempdir,
            _validation=True,
            _cat=[self.catlist[i] for i in self.valindex],
            _files=[self.filelist[i] for i in self.valindex],
            _donoise=[self.donoise[i] for i in self.valindex]  # fixed index
        )

    def __len__(self):
        nfiles = len(self.trainindex)
        return math.floor(nfiles / self.batch_size) if self.batch_size > 0 else 0

    def __getitem__(self, idx):
        img_batch = np.empty((self.batch_size,) + self.shape + (1,), dtype=np.float32)
        cat_batch = [0]*self.batch_size

        for batch in range(self.batch_size):
            index = self.trainindex[(batch + idx*self.batch_size) % len(self.trainindex)]
            img = tifffile.imread(self.filelist[index]).astype(np.float32)
            noising = self.donoise[index]

            # random flips
            if np.random.uniform(0,1.0) < 0.3:
                img = np.flip(img, axis=1)
            if np.random.uniform(0,1.0) < 0.3:
                img = np.flip(img, axis=2)

            if noising and self.augment > 1 and self.augment_noise:
                # global gamma
                if np.random.uniform(0,1.0) < 0.25:
                    gamma = 0.6 + 0.8*np.random.uniform(0,1.0)
                    img = exposure.adjust_gamma(img, gamma)
                # per-frame gamma
                if np.random.uniform(0,1.0) < 0.3:
                    gamma = 1.5*np.random.uniform(0,1.0)
                    frame = np.random.randint(0, int(np.sum(self.frame_shape)))
                    img[frame,] = exposure.adjust_gamma(img[frame,], gamma)
                # blur
                if np.random.uniform(0,1.0) < 0.25:
                    sig = 1.5*np.random.uniform(0,1.0)
                    img = gaussian_filter(img, sigma=sig)
                # noise
                if np.random.uniform(0,1.0) < 0.4:
                    noise = np.random.normal(0,1,img.shape)
                    sig = 1.5*np.random.uniform(0,1.0)
                    noise = gaussian_filter(noise, sigma=sig)
                    img = img + noise
                # blackout square
                if np.random.uniform(0,1.0) < 0.15:
                    sqsize = np.random.randint(1,12)
                    mask = np.full((self.half_size[0]*2+1, self.half_size[1]*2+1), 1, dtype=np.float32)
                    indx = np.random.randint(0, self.half_size[0]*2 - sqsize)
                    indy = np.random.randint(0, self.half_size[1]*2 - sqsize)
                    mask[indx:(indx+sqsize), indy:(indy+sqsize)] = 0
                    mask = np.repeat(mask[np.newaxis, :, :], int(np.sum(self.frame_shape)), axis=0)
                    img = img * mask
                # white square
                if np.random.uniform(0,1.0) < 0.15:
                    sqsize = np.random.randint(1,12)
                    mask = np.full((self.half_size[0]*2+1, self.half_size[1]*2+1), 0, dtype=np.float32)
                    indx = np.random.randint(0, self.half_size[0]*2 - sqsize)
                    indy = np.random.randint(0, self.half_size[1]*2 - sqsize)
                    mask[indx:(indx+sqsize), indy:(indy+sqsize)] = 1
                    mask = np.repeat(mask[np.newaxis, :, :], int(np.sum(self.frame_shape)), axis=0)
                    img = img + mask

            img_batch[batch, :, :, :, 0] = img
            cat_batch[batch] = self.catlist[index]

        cat_batch = to_categorical(cat_batch, self.ncat)
        return self._min_max_scaling(img_batch), cat_batch

    def get_batch_names(self, idx):
        fnames = []
        for batch in range(self.batch_size):
            index = self.trainindex[(batch + idx*self.batch_size) % len(self.trainindex)]
            fnames.append(self.filelist[index])
        return fnames

    def clean_tempdir(self):
        """Remove the temporary folder."""
        import shutil
        if os.path.isdir(self.tempdir) and os.path.abspath(self.tempdir) != os.path.abspath(self.data_path):
            shutil.rmtree(self.tempdir, ignore_errors=True)

    def on_epoch_end(self):
        np.random.shuffle(self.trainindex)

    def _min_max_scaling(self, data):
        n = data - np.min(data)
        d = np.max(data) - np.min(data)
        if d > 0:
            return n / d
        return n


# ============================================
#  Flow-following crop: use x/y centers per frame
# ============================================
class FlowFollowingMovieGeneratorFromROI(MovieGeneratorFromROI):
    """
    Use the velocity field and per-frame global centers to map global (y,x)
    to local (i,j) on the velocity grid, sample velocities, update the ROI center
    across frames, and crop flow-aligned movies.
    """
    def __init__(self, *args,
                 velocity=None,      # (T,H,W,2), [...,0]=vy, [...,1]=vx
                 centers=None,       # (T,2) per-frame global center (y,x)
                 vel_scale=1.0,      # convert velocities to pixels/frame if needed
                 interp_order=1,     # map_coordinates order
                 border_mode='reflect',
                 **kwargs):
        super().__init__(*args, **kwargs)
        if velocity is None or centers is None:
            raise ValueError("Please provide velocity (T,H,W,2) and centers (T,2).")
        self.velocity = np.asarray(velocity, dtype=np.float32)
        self.centers  = np.asarray(centers , dtype=np.float32)
        self.vel_scale = float(vel_scale)
        self.interp_order = int(interp_order)
        self.border_mode = str(border_mode)

        if self.velocity.ndim != 4 or self.velocity.shape[-1] != 2:
            raise ValueError(f"velocity should have shape (T,H,W,2), got {self.velocity.shape}")
        if self.centers.ndim != 2 or self.centers.shape[1] != 2:
            raise ValueError(f"centers should have shape (T,2), got {self.centers.shape}")
        if self.velocity.shape[0] != self.centers.shape[0]:
            raise ValueError("velocity and centers must have the same time length T.")

        # Local grid center in index space (0-based)
        self._loc_cy = (self.velocity.shape[1] - 1) / 2.0
        self._loc_cx = (self.velocity.shape[2] - 1) / 2.0

    def get_roi_img(self, z, y, x, img):
        """
        Integrate the velocity field to update the center per frame, then sample
        the crop around the updated centers.
        Time window: half-open interval [z - past, z + future)
        """
        past, future = self.frame_shape
        T_total, H, W = img.shape
        T_vel, Hv, Wv, _ = self.velocity.shape

        t0 = z - past
        t1 = z + future
        if t0 < 0 or t1 > T_total:  # image time bounds
            return None
        if t0 < 0 or t1 > T_vel:    # velocity time bounds
            return None

        hy, hx = self.half_size
        win_h = 2*hy + 1
        win_w = 2*hx + 1
        T_win = past + future

        oy = np.arange(-hy, hy+1, dtype=np.float32)[:, None]   # (win_h,1)
        ox = np.arange(-hx, hx+1, dtype=np.float32)[None, :]   # (1,win_w)

        out = np.empty((T_win, win_h, win_w), dtype=np.float32)

        yc = float(y); xc = float(x)
        centers = np.empty((T_win, 2), dtype=np.float32)
        centers[past] = (yc, xc)

        # integrate forward
        yk, xk = yc, xc
        for k in range(z, z + future - 1):
            vy, vx = self._sample_velocity_at_global(t=k, y=yk, x=xk)
            yk = yk + self.vel_scale * vy
            xk = xk + self.vel_scale * vx
            centers[(k - (z - past)) + 1] = (yk, xk)

        # integrate backward
        yk, xk = yc, xc
        for k in range(z - 1, z - past - 1, -1):
            vy, vx = self._sample_velocity_at_global(t=k, y=yk, x=xk)
            yk = yk - self.vel_scale * vy
            xk = xk - self.vel_scale * vx
            centers[(k - (z - past))] = (yk, xk)

        # crop per frame
        for i, t in enumerate(range(t0, t1)):
            cy, cx = centers[i]
            Y = oy + cy
            X = ox + cx
            YY = np.repeat(Y, win_w, axis=1)
            XX = np.repeat(X, win_h, axis=0)
            out[i] = map_coordinates(
                img[t], [YY, XX],
                order=self.interp_order,
                mode=self.border_mode,
                prefilter=False
            )

        return out

    def _sample_velocity_at_global(self, t, y, x):
        """
        Sample velocity at global (y,x) for frame t:
          - use per-frame global center (cy_g, cx_g)
          - map to local indices (i_loc, j_loc) relative to the local grid center
          - map_coordinates to fetch (vy, vx)
        """
        T, Hv, Wv, _ = self.velocity.shape
        if t < 0: t = 0
        if t > T - 1: t = T - 1

        cy_g, cx_g = self.centers[t]
        i_loc = (y - cy_g) + self._loc_cy
        j_loc = (x - cx_g) + self._loc_cx

        vy = map_coordinates(
            self.velocity[t, :, :, 0],
            [[i_loc], [j_loc]],
            order=self.interp_order,
            mode=self.border_mode,
            prefilter=False
        )[0]
        vx = map_coordinates(
            self.velocity[t, :, :, 1],
            [[i_loc], [j_loc]],
            order=self.interp_order,
            mode=self.border_mode,
            prefilter=False
        )[0]
        if not np.isfinite(vy): vy = 0.0
        if not np.isfinite(vx): vx = 0.0
        return float(vy), float(vx)
