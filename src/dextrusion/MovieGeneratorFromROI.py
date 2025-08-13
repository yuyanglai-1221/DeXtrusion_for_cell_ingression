from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import os, math, random
import tifffile
from glob import glob
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure
from dextrusion.RoiUtils import read_rois

"""
BSD 3-Clause License

Copyright (c) 2022, Gaëlle  LETORT and Alexis Villars
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

class MovieGeneratorFromROI(Sequence):
    """ Generate the input sequences (small and short windows) around ROIs position

    Generate the given windows and save them in a temporary folder, from which it is read during the training of the network. It allows not to load all the data before training, but to read it only when necessary.
    The temporary folder will be deleted at the end ofr training.
    """

    def __init__(self,
                 data_path,
                 batch_size=1,
                 frame_shape=(5,5),
                 win_halfsize = (25,25),
                 val_ratio=0.2,
                 balance=True,
                 ncat=3,
                 cat_names = None,
                 naug=1,
                 augment_withnoise = True,
                 add_nothing_windows = 10,
                 tmp_directory_path = None,
                 verbose = True,
                 **kwargs):

        self.data_path= data_path
        if data_path is not None:
            if not os.path.isdir(data_path):
                raise Exception("Folder "+data_path+" not found")
            self.dir_list = glob(os.path.join(data_path, '*.tif'))
       
        if tmp_directory_path is None:
            randid = random.randint(1,1000000)
            self.tempdir = data_path+"/tmp_data_from_roi_"+str(randid)+"/"
        else:
            self.tempdir = tmp_directory_path
        self.frame_shape = frame_shape
        self.half_size = win_halfsize
        self.shape = (np.sum(frame_shape),)+(2*win_halfsize[0]+1, 2*win_halfsize[1]+1)
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.ncat = ncat
        self.augment = naug
        self.augment_noise = augment_withnoise
        self.nothing = 1.2  ## slight shift, nothing more present
        self.add_nothing = add_nothing_windows ## Read _nothing.zip ROI file to reinforce no event windows (FP otherwise) and augment it XX times
        self.balance = True
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
        """ check if window of center (z,y,x) contains the roi point """
        if roi[0] < z-self.frame_shape[0] or roi[0] > (z+self.frame_shape[1]):
            return False
        if roi[1] < (y-self.half_size[0]) or roi[1] > (y+self.half_size[0]+1):
            return False
        return roi[2] > (x-self.half_size[1]) and roi[2] < (x+self.half_size[1]+1)
    
    def not_contain_rois(self,rois, z, y, x):
        """ Check that window centered in (z,y,x) don't contain a roi in the list """
        for roi in rois:
            if self.contain_roi(roi, z, y, x):
                return False
        return True
    
    def split_val(self):
        """ Separate train and validation sets """
        npts = len(self.catlist)
        indexes = np.arange(npts)
        np.random.shuffle(indexes)

        nbval = int(self.val_ratio * npts)
        nbtrain = npts - nbval
        val = np.random.permutation(indexes)[:nbval]
        indexes = np.array([i for i in indexes if i not in val])
        self.valindex = val
        self.trainindex = indexes
    
    def create_lists(self):
        """ Load train folder and list windows """
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
            
            ## get number of rois in each category
            crois = []
            nrois = np.zeros(self.ncat)
            for cat in range(1,self.ncat):
                roifilename = basename+self.catnames[cat]
                if os.path.isfile(roifilename):
                    rois = read_rois(roifilename)
                    crois = crois + rois
                    nrois[cat] = len(rois)
     
            if self.verbose:
                print("Nrois "+str(max(nrois)))
              
            ### Get rois of each category and balance/augment them if necessary
            for cat in range(1,self.ncat):
                roifilename = basename+self.catnames[cat]
                if os.path.isfile(roifilename):
                    rois =  read_rois(roifilename)
                    if nrois[cat]<0.75*max(nrois) and self.balance:
                        nrois[cat] = 0.85*max(nrois)
                    num = self.add_and_augment_rois(rois, nrois, cat, num, img, purename)

            ### Get "Nothing" rois
            lim = max(nrois)*self.nothing*self.augment*3000   ## stop after too many trials (can create some imbalance
            it = 0
            ndata = 0
            while ndata < (max(nrois)*self.nothing*self.augment) and (it<lim):
                it += 1
                x =  self.half_size[1] + np.random.randint(nx-2*self.half_size[1])
                y =  self.half_size[0] + np.random.randint(ny-2*self.half_size[0])
                z =  self.frame_shape[0] + np.random.randint(nz-self.frame_shape[1])
                if self.not_contain_rois(crois, z, y, x):
                    rimg = self.get_roi_img(z, y, x, img)
                    if rimg is not None:
                        self.add_point(rimg, 0, num, purename, np.random.uniform(0,1.0)<0.5)
                        ndata += 1
                        num += 1
    
            ### Add ROIs that contain no event, reinforcement to reduce false positive
            if self.add_nothing > 1:
                num = self.add_nothing_rois(basename, num, img, purename)

    def add_nothing_rois(self, basename, num, img, purename):
        roifilename = basename+"_nothing.zip"
        if os.path.isfile(roifilename):
            rois =  read_rois(roifilename)
            for iroi in range(len(rois)):
                z, y, x = self.get_roi_pos(rois, iroi)
                rimg = self.get_roi_img(z, y, x, img)
                if rimg is not None:
                    self.add_point(rimg, 0, num, purename, True)
                    num = num + 1

            ## augment, of double augment times
            if self.augment>1:
                ntarget = len(rois)*self.add_nothing
                ndone = 0
                lim = ntarget*1000   ## stop after too many trials (can create some imbalance
                it = 0
                while (ndone < ntarget) and (it<lim):
                    it = it + 1
                    ind = random.randrange(len(rois))
                    z, y, x = self.get_roi_pos(rois, ind)
                    rimg = self.get_roi_img(z, y, x, img)
                    if rimg is not None:
                        self.add_point(rimg, 0, num, purename, False)
                        num = num + 1
                        ndone = ndone + 1
        return num
        


    def add_and_augment_rois(self, rois, nrois, cat, num, img, purename):
        ncat = 0
        for iroi in range(len(rois)):
            z, y, x = self.get_roi_pos(rois, iroi)
            rimg = self.get_roi_img(z, y, x, img)
            if rimg is not None:
                self.add_point(rimg, cat, num, purename, True)
                num = num + 1
                ncat = ncat + 1

        ## augment if necessary
        lim = nrois[cat]*self.augment*2000   ## stop after too many trials (can create some imbalance
        it = 0
        while (ncat < nrois[cat]*self.augment) and (it<lim):
            it += 1
            ind = random.randrange(len(rois))
            z, y, x = self.get_roi_pos(rois, ind)
            rimg = self.get_roi_img(z, y, x, img)
            if rimg is not None:
                self.add_point(rimg, cat, num, purename, False)
                num = num + 1
                ncat = ncat + 1
        return num


    def get_roi_pos(self, rois, ind):
        """ 
            Get the roi position and add a small random noise around it
        """
        roi = rois[ind]
        ## center position
        xroi = roi[2]
        yroi = roi[1]
        frame = roi[0]
        ## draw randomly around the center point
        y = math.floor(yroi + np.random.uniform(-1.0,1.0)*self.half_size[0]*0.2)
        x = math.floor(xroi + np.random.uniform(-1.0,1.0)*self.half_size[1]*0.2)   
        z = math.floor(frame + np.random.uniform(-1.0,1.0)*2)
        return (z, y, x)


    def add_point(self, im, cat, num, name, orig):
        """
            Add a window in category cat
            Save a tiff image of the window in the temp folder
            And it to the lists
        """
        outname = self.tempdir+"cat"+str(cat)+"_"+name+"_roi"+str(num)+".tif"
        im = np.uint8(self._min_max_scaling(im)*255)
        tifffile.imwrite(outname, im, imagej=True)
        self.filelist.append(outname)
        self.catlist.append(cat)
        if orig:
            self.donoise.append(False)
        else:
            self.donoise.append(True)

    def write_img(self, im, ind, num, folder="conv"):
        """ save image im as a tif file """
        resdir = self.data_path+"/results_"+folder+"/"
        if not os.path.exists(resdir):
            os.makedirs(resdir)
        fname = os.path.basename(self.get_batch_names(ind)[num])
        outname = resdir+fname
        im = np.uint8(self._min_max_scaling(im)*255)
        tifffile.imwrite(outname, im, imagej=True)

    def get_roi_img(self, z, y, x, img):
        """ 
            Get the image window around roi 
            Return None if the roi is in the border (all the window doesn't fit)
        """
        if (x-self.half_size[1])<0 or (x+self.half_size[1]+1)>img.shape[2]:
            return None
        if (y-self.half_size[0])<0 or (y+self.half_size[0]+1)>img.shape[1]:
            return None
        if (z-self.frame_shape[0])<0 or (z+self.frame_shape[1])>img.shape[0]:
            return None
        return img[z-self.frame_shape[0]:z+self.frame_shape[1],y-self.half_size[0]:y+self.half_size[0]+1,x-self.half_size[1]:x+self.half_size[1]+1]

    def get_validation_generator(self):
        return self.__class__(
            data_path = self.data_path,
            batch_size = self.batch_size,
            frame_shape = self.frame_shape,
            win_halfsize = self.half_size,
            val_ratio = self.val_ratio,
            balance = self.balance,
            naug = self.augment,
            augment_noise = self.augment_noise,
            verbose = self.verbose,
            ncat = self.ncat,
            tmp_directory_path = self.tempdir,
            _validation = True,
            _cat = [self.catlist[i] for i in self.valindex],
            _files = [self.filelist[i] for i in self.valindex],
            _donoise = [self.donoise[i] for i in self.donoise]
        )         
                
                
    def __len__(self):
        nfiles = len(self.trainindex)
        return math.floor(nfiles/self.batch_size)
            
    def __getitem__(self, idx):
        img_batch = np.empty((self.batch_size,)+self.shape+(1,))
        cat_batch = [0]*self.batch_size
        
        for batch in range(self.batch_size):
            index = self.trainindex[(batch+idx*self.batch_size)%len(self.trainindex)]
            img = tifffile.imread(self.filelist[index])
            noising = self.donoise[index]
            ## introduce some randomization by flipping the movie
            if np.random.uniform(0,1.0)<0.3:    
                img = np.flip(img,axis=1)
            if np.random.uniform(0,1.0)<0.3:
                img = np.flip(img,axis=2)

            if noising and self.augment>1 and self.augment_noise:
                ## random gama exposure filter
                if np.random.uniform(0,1.0)<0.25:
                    gama = 0.6+0.8*np.random.uniform(0,1.0)
                    img = exposure.adjust_gamma(img, gama)
                
                ## random gama exposure filter on one time-frame only (shadow, brightness)
                if np.random.uniform(0,1.0)<0.3:
                    gama = 1.5*np.random.uniform(0,1.0)
                    frame = np.random.randint(0,np.sum(self.frame_shape))
                    img[frame,] = exposure.adjust_gamma(img[frame,], gama)
                
                
                if np.random.uniform(0,1.0)<0.25:
                    sig = 1.5*np.random.uniform(0,1.0)
                    img = gaussian_filter(img, sigma=sig)
                if np.random.uniform(0,1.0)<0.4:
                    noise = np.random.normal(0,1,img.shape)
                    sig = 1.5*np.random.uniform(0,1.0)
                    noise = gaussian_filter(noise, sigma=sig)
                    img = img + noise

                ## put a random square to 0
                if np.random.uniform(0,1.0)<0.15:
                    sqsize = np.random.randint(1,12)
                    mask = np.full((self.half_size[0]*2+1,self.half_size[1]*2+1), 1)
                    indx = np.random.randint(0,self.half_size[0]*2-sqsize)
                    indy = np.random.randint(0,self.half_size[1]*2-sqsize)
                    mask[indx:(indx+sqsize),indy:(indy+sqsize)] = 0
                    mask = np.repeat(mask[np.newaxis,:,:], np.sum(self.frame_shape), axis=0)
                    img = img * mask
                ## put a random square to 1
                if np.random.uniform(0,1.0)<0.15:
                    sqsize = np.random.randint(1,12)
                    mask = np.full((self.half_size[0]*2+1,self.half_size[1]*2+1), 0)
                    indx = np.random.randint(0,self.half_size[0]*2-sqsize)
                    indy = np.random.randint(0,self.half_size[1]*2-sqsize)
                    mask[indx:(indx+sqsize),indy:(indy+sqsize)] = 1
                    mask = np.repeat(mask[np.newaxis,:,:], np.sum(self.frame_shape), axis=0)
                    img = img + mask


            img_batch[batch,:,:,:,0] = img # np.transpose(src, axes=[2,1,0])
            cat_batch[batch] = self.catlist[index]
        
        cat_batch = to_categorical(cat_batch, self.ncat)
        return self._min_max_scaling(img_batch), cat_batch
    
    def get_batch_names(self, idx):
        fnames = []
        for batch in range(self.batch_size):
            index = self.trainindex[(batch+idx*self.batch_size)%len(self.trainindex)]
            fnames.append(self.filelist[index])
        
        return fnames
   
    def clean_tempdir(self):
        """ Clean the temporary folder """
        import shutil
        shutil.rmtree(self.tempdir)
        
    def on_epoch_end(self):
        np.random.shuffle(self.trainindex)
   
    def _min_max_scaling(self, data):
        n = data - np.min(data)
        d = np.max(data) - np.min(data) 
        if (d>0):
            return n/d
        return n
   

