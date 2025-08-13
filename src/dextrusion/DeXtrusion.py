import functools
import os, random, re, time, cv2, gc
import numpy as np
from math import floor, ceil, isnan
from numpy import zeros, uint8, expand_dims
from numpy import min as npmin
from numpy import max as npmax
from tifffile import imread, imwrite
from skimage.segmentation import watershed
from skimage.morphology import local_maxima, disk
from skimage.filters import rank
from scipy.ndimage import zoom, label, distance_transform_edt, maximum_filter, minimum_filter, median_filter, uniform_filter
from scipy.ndimage import measurements, zoom
from scipy.ndimage import sum as ndsum

import dextrusion.RoiUtils as ru
from dextrusion.Network import Network  

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

class DeXtrusion:
    """

        Detect extrusion/other events position in a whole movie 
        
        - (Re)Train a neural network to do the detection
        - Use trained neural network(s) on a movie to detect events

        DeXtrusion uses moving windows and used the trained network(s) on each window to calculate its probability to contain an event.
        The probability map can be converted to a list of point events, and exported as Fiji ROIs.

        DeXtrusion is distributed as a python module, installable via pip or via the setup file.

    """

    def __init__(self, verbose=True):
        """ Create class object. 
        :param verbose: if should print information messages while running
        """
        self.model = Network(verbose)    ## initialize neural network     
        self.verbose = verbose           ## print information messages while running   
        self.clear()

    def clear(self):
        """ Reset the object """
        self.nframes = (0,0)
        self.half_size = (0,0)
        self.ncat = 0
        self.catnames = None
        self.nfilters = 0
        self.batch_size = 0
        self.cell_diameter = 0
        self.extrusion_duration = 0
        

    ################################################################################"
    ###### Parameters/configuration

    def set_parameters(self, nframes=(5,5), half_size=(22,22), ncat=4, catnames=None, nb_filters=8, batch_size=30, model_path="./model", cell_diameter=25, extrusion_duration=4.5):
        """ Set the parameters of deXtrusion network, DeXNet
        :param nframes: number of frames before and after the event to use in the sliding windows
        :param half_size: half size of the sliding window in X and Y
        :param ncat: number of event type to detect
        :param catnames: the names of the events (should be the same as the end of the training ROI files)
        :param nb_filters: size of the convolutional neural network before going through the temporal network
        :param model_path: path to where the model is saved/to be saved
        :param batch_size: batch size used for training of the network
        :param cell_diameter: typical diameter of a cell, in pixels, used to resize the movies if necessary
        :param extrusion_duration: typical number of frames in which the extrusion is visible, used to resize temporally the movies f necessary
        """
        self.nframes = nframes
        self.half_size = half_size
        self.ncat = ncat
        self.catnames = catnames
        if self.catnames is None: 
            self.catnames = ["", "_cell_death.zip", "_cell_sop.zip", "_cell_division.zip"]
        self.nfilters = nb_filters
        self.batch_size = batch_size
        self.model_path = model_path
        self.cell_diameter = cell_diameter
        self.extrusion_duration = extrusion_duration
        
        self.write_configuration()  ## write a file with all these paramters to allow to reload it
    
    def retrain_parameters( self, model_path="./model_retrain", datapath="./", aug=3, epoch=10 ):
        """ Load network parameters and change only those to retrain
        """
        self.model_path = model_path
        self.write_configuration()
        self.add_configuration_info( epoch, aug, path=datapath )
        
    def write_configuration(self):
        """ Write the values of the parameter used in a configuration file (named config.cfg) to allow to load the network later """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        fname = self.model_path+"config.cfg"
        with open(fname, 'w') as f:
            f.write('### DeXtrusion configuration file - dont edit ###\n')
            f.write('nb_events_category = '+str(self.ncat)+'\n')
            f.write('events_category_names = '+str(self.catnames)+'\n')
            f.write('window_half_size = '+str(self.half_size)+'\n')
            f.write('nb_temporal_frames = '+str(self.nframes)+'\n')
            f.write('cell_diameter = '+str(self.cell_diameter)+'\n')
            f.write('extrusion_duration = '+str(self.extrusion_duration)+'\n')
            f.write('neuralnetwork_nb_filters = '+str(self.nfilters)+'\n')
            f.write('neuralnetwork_batch_size = '+str(self.batch_size)+'\n')

    def add_configuration_info(self, nepochs=50, naug=1, add_nothing_windows=0, path=""):
        """ Complete the configuration file (file of used parameters) with parameters used for network training 
        :param nepochs: number of iteration to do for training
        :param naug: number of augmentation done to the training data (1 for no augmentation)
        :param add_nothing_windows: add false positive ROIs (saved in file filename_nothing.zip) to force training on this data as control
        :param path: path to the training data
        """
        params = "neuralnetwork_nb_epochs = "+str(nepochs)+"\n"
        params = params + "datatraining_augmentation = "+str(naug)+"\n"
        params = params + "datatraining_addnothingwindows = "+str(add_nothing_windows)+"\n"
        params = params + "datatraining_path = "+path+"\n"
        fname = self.model_path+"config.cfg"
        with open(fname, 'a') as f:
                f.write(params)

    def read_configuration(self, modelpath):
        """ Read configuration file that is automatically saved in the model folder 'modelpath' to set the parameters """
        configfile = open(os.path.join(modelpath,'config.cfg'), 'r')
        lines = configfile.readlines()
        for line in lines:
            vals = line.split("=")
            if len(vals) >= 1:
                if vals[0].strip() == "cell_diameter":
                    self.cell_diameter = float(vals[1])
                if vals[0].strip() == "extrusion_duration":
                    self.extrusion_duration = float(vals[1])
                if vals[0].strip() == "nb_events_category":
                    self.ncat = int(vals[1])
                if vals[0].strip() == "events_category_names":
                    #print(vals[1])
                    self.read_catnames(vals[1])
                    #print(self.catnames)
                if vals[0].strip() == "window_half_size":
                    self.half_size = self.read_tuple(vals[1])
                if vals[0].strip() == "nb_temporal_frames":
                    self.nframes = self.read_tuple(vals[1])
                if vals[0].strip() == "neuralnetwork_nb_filters":
                    self.nfilters = int(vals[1])
                if vals[0].strip() == "neuralnetwork_batch_size":
                    self.batch_size = int(vals[1])
        if self.catnames is None:
            self.catnames = ["", "_cell_death.zip", "_cell_sop.zip"]
        
        #self.group_size = group_size  ## batch of windows that will be runned simultanously in the network (compromise between running speed and GPU utilization)
        
        ## Bounds: size of area to fill with probability of the window calculated (not all the window, around the center)
        self.bounds = ( floor(self.nframes[0]*0.2), floor(self.nframes[1]*0.2)+1, floor(self.half_size[0]*0.3), floor(self.half_size[1]*0.3) )
    
    
    def read_tuple(self, instr):
        return tuple(map(int, re.findall(r'[0-9]+', instr)))

    def read_catnames(self, instr):
        """ Read the event names from the line in the configuration file """
        self.catnames = []
        events = re.split(',|\[|\]|\n', instr)
        for ind in range(1,len(events)-2):
            self.catnames.append(events[ind].strip().replace("'",""))


    ################################################################################################ 
    ###### Neural network (model) calls
    
    def initialize_model(self, nframes=(5,5), half_size=(22,22), ncat=4, nb_filters=8, batch_size=30, model_path="./model"):
        shape = (np.sum(nframes),) + (half_size[0]*2+1, half_size[1]*2+1)
        self.model.create_model(shape, self.ncat, self.nfilters)

    def train_model(self, train_generator, validation_generator, epochs=50, save=True, plot=True):
        """ Train the neural network with the current parameters 
            The trained network will be save in the model_path directory
            :param train_generator: generator that creates the windows used for training from the ROI files
            :param validation_generator: creates the validation windows to check the training evolution
            :param epochs: number of training iteration to do
            ;param save: save the trained network at the end
            :param plot: show the training curves during training
        """
        if self.verbose:
            start_time = time.time()
        self.model.train(train_generator, validation_generator, epochs=epochs, plot=plot)
        if self.verbose:
            print("----- Training in %s minutes ----" % ((time.time()-start_time)/60) )
        if save:
            self.model.save(self.model_path)
            self.write_configuration()
    
    def evaluate_model_prediction(self, test_generator):
        self.model.evaluate_prediction(test_generator)

    def save_model(self):
        self.model.save(self.model_path)

    def load_model(self, modelpath):
        """ Load a model and its parameters
        """
        if self.model is None:
            self.model = Network(self.verbose)
        self.model_path = modelpath
        self.model.reset(modelpath)
        self.read_configuration(modelpath)
    
    def look_at_latent_features(self, test):
        """ Gives access to the layer before the GRU, containint the vector of latent features for each image of the input sequences
        """
        td = self.model.get_timedis_layer()
        if self.verbose:
            print("Latent features shape:")
            print(td.summary())
        print(test.data_path)
        self.model.predict_convolution(test)

    ############################################################################################
    ###### Movie manipulation

    def resize_image(self, img, ratioxy=1, ratioz=1):
        return zoom(img, (ratioz, ratioxy, ratioxy))
    
    def check_image(self, img):
        """ Check the format of the movie
        """
        if len(img.shape) <= 2:
            print("Error, input is not a movie !")
            return -1
        if len(img.shape) > 3:
            print("Error, input has too many dimensions \n Should be a gray scale 3D image, check that it is not RGB")
            return -1
        return 1

    def to_init_shape(self, img):
        """ Resize back to the original movie size
        """
        ratioxy = self.init_shape[1]/img.shape[1]
        ratioz = self.init_shape[0]/img.shape[0]
        return self.resize_image(img, ratioxy=ratioxy, ratioz=ratioz)

    ###########################################################################################
    ###### Detect events from movie: calculate probability map
        
    def divide(self, res, nres):
        """ Divide the proba results by number of windows that filled it """
        nres[nres==0] = 1
        try:
            if self.probamap is not None:
                del self.probamap
                gc.collect()
                self.probamap = None
        except:
            self.probamap = None
        self.probamap = zeros((self.ncat-1,)+self.scaled_shape, dtype="uint8")
        for c in range(self.ncat-1):
            self.probamap[c,:,:,:] = uint8(res[c,(self.nframes[0]):(res.shape[1]-self.nframes[1]),:,:]/nres[(self.nframes[0]):(res.shape[1]-self.nframes[1]),:,:]*255)

    def do_one_group(self, img, pred_channels, npred, group):
        """ Categorize one batch of windows (one group), put the results in pred_channels """
        if self.verbose:
            print("     - Do one group: "+str(group))
        
        # Prepare the batch of windows to run through the network
        nind = min(self.group_size, self.nw[0]*self.nw[1]*self.nw[2]-group)
        img_group = [ self.fill_group(img, group+cur_ind) for cur_ind in range(0,nind) ]
        

        # Do the prediction with the network
        img_group = np.array(img_group)/255 # normalize   
        self.model.reset(self.model_path)
        pred = self.model.predict_batch(img_group)

        # Fill the proba results in the correct place on the pred_channels image
        for c in range(self.ncat-1):
            do = list(map(functools.partial(self.place_element, pred_channels[c], npred, pred[:,1+c], group, c), range(self.group_size)))

    def fill_group(self, img, gind):
        """ Get the window from the original image and place it in the batch of windows to be analysed """
        cur = self.get_index(gind) 
        curimg = img[(cur[0]-self.nframes[0]):(cur[0]+self.nframes[1]), (cur[1]-self.half_size[0]):(cur[1]+self.half_size[0]+1),(cur[2]-self.half_size[1]):(cur[2]+self.half_size[1]+1)]
        
        ## Scale intensities in the window to run through network
        d = npmax(curimg) - npmin(curimg) 
        if (d>0):
            curimg = (curimg - npmin(curimg))/d
        else:
            curimg = (curimg-npmin(curimg))
        return expand_dims( uint8(curimg*255), axis=3)
    
    def get_index(self, gind):
        index = (self.nframes[0] + self.shiftz + int((gind/self.nw[3])%self.nw[0])*self.dzxy[0], self.half_size[0]+ self.shiftxy + int((gind/self.nw[2])%self.nw[1])*self.dzxy[1], self.half_size[1]+ self.shiftxy + int(gind%self.nw[2])*self.dzxy[1])
        return index

    def update_nwins(self, imgshape):
        nx = int(ceil((imgshape[2]-self.half_size[1]*2-1-self.shiftxy)/(self.dzxy[1]*1.0)))
        ny = int(ceil((imgshape[1]-self.half_size[0]*2-1-self.shiftxy)/(self.dzxy[1]*1.0)))
        nz = int(ceil((imgshape[0]-np.sum(self.nframes)-self.shiftz)/(self.dzxy[0]*1.0)))
        self.nw = (nz, ny, nx, ny*nx, nz*ny*nx)

    def place_element(self, pred_c, npred, pred, group, chan, ind):
        """ Place the results of the window in the correct place of the results image """
        # current index
        gind = group+ind
        
        # out of range
        if gind >= self.nw[4]:
            return 0
        
        index = self.get_index(gind)

        pred_c[(index[0]-self.bounds[0]):(index[0]+self.bounds[1]),
                (index[1]-self.bounds[2]):(index[1]+self.bounds[2]+1),
                (index[2]-self.bounds[3]):(index[2]+self.bounds[3]+1)] += pred[ind]
        if chan==0:
            npred[(index[0]-self.bounds[0]):(index[0]+self.bounds[1]),
                (index[1]-self.bounds[2]):(index[1]+self.bounds[2]+1),
                (index[2]-self.bounds[3]):(index[2]+self.bounds[3]+1)] += 1
        return 0
    
    def detect_events_onmovie(self, moviepath, models=[], cell_diameter=25, extrusion_duration=4.5, dxy=25, dz=2, group_size=150000, outfolder=None):
        """ Detect events from the movie by using sliding windows and trained network
        :param moviepath: path to the movie to do the detection on
        :param models: list of the names of one (or more) networks to use for the dection
        :param cell_diameter: typical size (in pixels) of a cell, used to resize if necessary the movie
        :param extrusion_duration: typical temporal duration of an extrusion (number of frames on which it is visible), to resize if necessary
        :param dxy: spatial step of sliding window
        :param dz: temporal step of sliding window
        :param group_size: computational parameter: number of windows runned at the same time for detection. Depends on the computer capacity (more = faster)
        :param outfolder: where to save the results. By default will create a folder called "results" in the same folder as the input movie
        """
        if self.verbose:
            start_time = time.time()
        img = imread(moviepath)
        imdir = os.path.dirname(moviepath)
        output_name = os.path.basename(moviepath)
        output_name = os.path.splitext(output_name)[0]
        if outfolder is None:
            self.outpath = os.path.join(imdir, "results")
        else:
            self.outpath = outfolder
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        self.outname = os.path.join(self.outpath, output_name)
        self.detect_events(img, models, cell_diameter, extrusion_duration, dxy, dz, group_size=group_size)

        if self.verbose:
            print("---- Detection finished, took %s minutes ---" % ((time.time() - start_time)/60))
        del img
        gc.collect()

    def detect_events(self, img, model_paths=[], cell_diameter=25, extrusion_duration=4.5, dxy=25, dz=2, group_size=150000):
        """ Predict events on an image (already loaded in img parameter, return labelled image """
       
        ### load model parameters and check image
        if len(model_paths) <= 0:
            print("Choose model(s) to use (should be a list) !")
            return
        self.load_model(model_paths[0])
        
        if self.check_image(img) < 0:
            return
        self.group_size = group_size

        ### resize if necessary to have same scale as training
        self.init_shape = img.shape
        if self.verbose:
            print("  Initial image shape: "+str(img.shape))
        if abs(cell_diameter-self.cell_diameter) > (self.cell_diameter*0.3):
            ratio = self.cell_diameter/cell_diameter
            img = self.resize_image(img, ratioxy=ratio, ratioz=1)

        ### resize temporally if necessary
        if abs(extrusion_duration-self.extrusion_duration) > (self.extrusion_duration*0.3):
            ratio = self.extrusion_duration/extrusion_duration
            img = self.resize_image(img, ratioxy=1, ratioz=ratio)
        self.scaled_shape = img.shape
        
        ### add frames before and after to sreen all the times with the windows
        if self.verbose:
            print("  Rescaled image shape: "+str(img.shape))
        first = img[0,]
        img = np.concatenate( (np.repeat(first[np.newaxis,], self.nframes[0], axis=0), img), axis=0)
        last = img[len(img)-1,]
        img = np.concatenate( (img, np.repeat(last[np.newaxis,], self.nframes[1], axis=0)), axis=0)
        if self.verbose:
            print("  Extended image shape: "+str(img.shape))

        ### initialize parameters
        self.dzxy = (dz, dxy)
        pred_channels = zeros((self.ncat-1,)+img.shape, dtype="float16") 
        npred = zeros(img.shape, dtype="uint8")
        
        ## run for each model in the models list
        nmod = len(model_paths)
        for cmod in range(nmod):
            self.shiftz = floor(cmod*dxy/nmod)
            self.shiftxy = floor(cmod*dz/nmod)
            self.load_model(model_paths[cmod])
            self.update_nwins(img.shape)
            if self.verbose:
                print("   Doing windows with shift: "+str(self.shiftxy)+" "+str(self.shiftz)+" and model "+model_paths[cmod])
                print("   Number of windows to categorize: "+str(self.nw[4]))
            do = list(map(functools.partial(self.do_one_group, img, pred_channels, npred), range(0,self.nw[4],self.group_size) ))
        
        ## divide result map by the number of predictions added in each pixel
        self.divide(pred_channels, npred)
        del pred_channels
        del npred
  
    ##################################################################################################
    ####### Postprocess results: probability maps, Fiji ROIs
    
    def rescale_position(self, z, y, x, probshape=None):
        if probshape is not None:
            probashape = probshape
        else:
            probashape = self.probamap.shape
        ratioxy = self.init_shape[1]/probashape[2]
        ratioz = self.init_shape[0]/probashape[1]
        z = floor(z*ratioz)
        y = floor(y*ratioxy)
        x = floor(x*ratioxy)
        return z, y, x

    def resize_probamap(self, cat=1):
        ''' resize to initial size proba map '''
        interp_img = zoom(self.probamap[cat-1], (self.dzxy[0], self.dzxy[1], self.dzxy[1]), grid_mode=False)
        diffz = interp_img.shape[0]-self.init_shape[0]
        res = zeros(self.init_shape, dtype="uint8")
        res[:, self.half_size[0]:(self.half_size[0]+interp_img.shape[1]), self.half_size[1]:(self.half_size[1]+interp_img.shape[2])] = interp_img[0:(interp_img.shape[0]-diffz),:,:]
        return res

    def write_rawproba_maps(self, endname="_rawproba.tif", astime=True):
        """ Save the raw probability maps of all the different event category in .tif movies (not rescaled) """
        for cat in range(1, self.ncat):
            self.write_rawproba_map(cat, endname, astime)

    def write_rawproba_map(self, cat=1, endname="_rawproba.tif", astime=True):
        ''' Draw raw probability map of event category 'cat' in a .tif movie (raw values) '''
        res = self.probamap[cat-1]
        catname = self.catnames[cat]
        catname = os.path.splitext(catname)[0]
        outfile = self.outname+catname+endname
        if astime:
            imwrite(outfile, res, imagej=True, metadata={'axes':'TYX'})
        else:
            imwrite(outfile, res, imagej=True, metadata={'axes':'ZYX'})
        if self.verbose:
            print("Raw probability map "+outfile+" saved")
    
    def write_probamaps(self, imgpath=None, cat=1, astime=True):
        """ Write probability map(s) scaled to original movie size """
        if cat is None:
            cats = range(1,self.ncat)
        else:
            cats = [cat]
        
        for icat in cats:
            if imgpath is None:
                img = self.probamap[icat-1]
            else:
                ### get the movie file directory, name and size
                img = self.get_rawproba_frompath(imgpath, cat=icat)
            self.rescale_and_write( img, cat=icat, endname="_proba.tif", imgtype="Probability map", astime=astime)


    def write_cleanedprobamap(self, cat=1, volume_threshold=800, proba_threshold=180, disxy=10, distime=4, imgpath=None, endname="_proba.tif", astime=True):
        ''' From raw probability map, draw the cleaned and rescaled probability map of event category 'cat' in a .tif movie '''

        res = self.clean_probamap(imgpath, cat=cat, threshold=125, mindxy=disxy, mindt=distime, volume_threshold=volume_threshold, proba_threshold=proba_threshold)
        self.rescale_and_write( res, cat=cat, endname=endname, imgtype="Cleaned probability map", astime=astime )

    def rescale_and_write(self, img, cat=1, endname=".tif", imgtype="Movie", astime=True ):
        """ Rescale if necessary the image to initial image shape, and save it as .tif file """
        if self.init_shape != img.shape:
            if self.verbose:
                print("Rescaling from "+str(img.shape)+" to "+str(self.init_shape))
            img = self.to_init_shape(img)

        catname = self.catnames[cat]
        catname = os.path.splitext(catname)[0]
        outfile = self.outname+catname+endname
        if astime:
            imwrite(outfile, img, imagej=True, metadata={'axes':'TYX'})
        else:
            imwrite(outfile, img, imagej=True, metadata={'axes':'ZYX'})
        if self.verbose:
            print(imgtype+" "+outfile+" saved")
   
    def get_rawproba_frompath( self, imgpath, cat=1):
        ### get the movie file directory, name and size
        im = imread(imgpath)
        self.init_shape = im.shape
        im = None
        self.outpath = os.path.join( os.path.dirname(imgpath), "results")
        output_name = os.path.basename(imgpath)
        output_name = os.path.splitext(output_name)[0]
        catname = self.catnames[cat]
        catname = os.path.splitext(catname)[0]
        probapath = os.path.join(self.outpath, output_name+catname+"_rawproba.tif")
        img = imread(probapath)
        self.outname = os.path.join(self.outpath, output_name)
        return img
    
    def clean_probamap(self, imgpath, cat=1, threshold=125, mindxy=5, mindt=3, volume_threshold=800, proba_threshold=180):
        """ Transform input image to event, looking at object shape """
        if imgpath is None:
            img = self.probamap[cat-1]
        else:
            img = self.get_rawproba_frompath( imgpath, cat=cat)

        bimg, labels, llabels, vols, vals = self.rawproba_to_volumes( img, threshold=125, mindxy=mindxy, mindt=mindt )
        cleanproba = np.zeros(img.shape, dtype="uint8")
        for lab, vol, val in zip(llabels, vols, vals):
            if not isnan(vol):
                if (vol>volume_threshold) and (val>proba_threshold):
                    cleanproba[labels==lab] = floor(val)
        return cleanproba


    def get_rois(self, cat=None, volume_threshold=1000, proba_threshold=180, disxy=10, dist=4):
        """ From probamap, get ROIs from centroids of positive (proba>=proba_threshold) volumes 
        :param cat: number of the event (in the list of events) to process. If None, will process all the events.
        :param volume_threshold: threshold of the positive detection volume to keep it and transform it to ROI
        :param proba_threshold: threshold of probability of the event detection to consider the pixel as positive
        :param disxy: threshold spatial distance to consider two events to be the same
        :param dist: temporal distance to consider two events to be the same
        """
        
        if cat is None:
            cats = range(1,self.ncat)
        else:
            cats = [cat]
        
        for icat in cats:
            binimg, labels, llabels, vols, vals = self.rawproba_to_volumes(self.probamap[icat-1], 125, mindxy=disxy, mindt=dist)
            coords = measurements.center_of_mass(binimg, labels=labels, index=llabels)
            del binimg
            gc.collect()
            rois = []
            for (pt, vol, val) in zip(coords, vols, vals):
                if not isnan(pt[0]) and (vol>volume_threshold) and (val>proba_threshold):
                    roiz, roiy, roix = self.rescale_position( pt[0], pt[1], pt[2] )
                    rois.append(ru.create_roi((roiz, roiy, roix), cat=icat))
            
            outfile = self.outname+self.catnames[icat]
            ru.write_rois(outfile, rois, verbose=self.verbose)

    def rawproba_to_volumes(self, img, threshold, mindxy=5, mindt=3, to_image=False):
        """ Transform raw probamap to separated event volumes (local maxima and watershed) """
        binimg = np.copy(img)
        binimg[img>threshold]=1
        binimg[img<=threshold]=0
        # local center of the contigous positive volume
        binimg = minimum_filter(binimg, size=(0, 2, 2))
        distance = distance_transform_edt(binimg)
        maxs = local_maxima(distance, connectivity=16)
        maxs = maximum_filter(maxs, size=(mindt, mindxy, mindxy))
    
        maxlabels, num_labels = label(maxs>0)
        labels = watershed(-distance, maxlabels, mask=binimg)
        # measure center of mass position and size and intensity of each positive volume
        num_labels = np.max(labels)
        listlabels = np.arange(1,num_labels+1)
        vols = ndsum(binimg, labels, listlabels)
        vals = ndsum(img, labels, listlabels)
        vals = vals/vols
        del distance
        del maxs
    
        return binimg, labels, listlabels, vols, vals

    
    def get_rois_fromrawproba_path(self, imagepath=None, cat=1, volume_threshold=800, proba_threshold=180, disxy=10, dist=4, outfile=None):
        """ 
           Clean the raw probamap, get rois and rescale to original image size 
           Read the raw probamap in the results folder.
        """
        if imagepath is None:
            img = self.probamap[cat-1]
        else:
            img = self.get_rawproba_frompath( imagepath, cat=cat)

        bimg, labels, llabels, vols, vals = self.rawproba_to_volumes( img, threshold=125, mindxy=disxy, mindt=dist )
        coords = measurements.center_of_mass(bimg, labels=labels, index=llabels)
        rois = []
        for (pt, vol, val) in zip(coords, vols, vals):
            if not isnan(pt[0]) and (vol>volume_threshold) and (val>proba_threshold):
                roiz, roiy, roix = self.rescale_position( pt[0], pt[1], pt[2], ((1,))+img.shape )
                rois.append(ru.create_roi((roiz, roiy, roix), cat=cat))
            
        if outfile is None:
            outfile = self.outname+self.catnames[cat]
        ru.write_rois(outfile, rois, verbose=self.verbose)

    def compare_rois(self, cat, gtroisfile, resroisfile=None, distance_xy=15, distance_t=4):
        """ Compare two ROIs files and measure accuracy metrics (precision, recall)
        :param cat: number of the event (in the event names list) to compare the ROIs
        :param gtroisfile: manual (or other source) detections ROIs file
        :param resroisfile: dextrusion output (or other source) detection ROIs file. If None, look for the file in the results folder, where dextrusion save the results
        :param distance_xy: spatial distance (in pixels) to consider two ROIs as the same
        :param distance_t: temporal distance (number of frames) to consider two ROIs as the same
        """
        if resroisfile is None:
            resroisfile = self.outname+self.catnames[cat]
        return ru.compare_rois(resroisfile, gtroisfile, distance_xy=distance_xy, distance_t=distance_t)
    
    def write_falsepositives(self, cat, resfile, gtfile, distance_xy=15, distance_t=4, outputfile=None):
        """ Find all false positives detection and save them in a Fiji file
        """
        if outputfile is None:
            self.outpath = os.path.dirname(resfile)
            output_name = os.path.basename(resfile)
            output_name = os.path.splitext(output_name)[0]
            outputfile = os.path.join(self.outpath, output_name+"_FP.zip")
        ru.write_falsepositives(resfile, gtfile, outputfile=outputfile, distance_xy=distance_xy, distance_t=distance_t)

    def write_falsenegatives(self, cat, resfile, gtfile, distance_xy=15, distance_t=4, outputfile=None):
        """ Find all false negatives detection and save them in a Fiji file
        """
        if outputfile is None:
            self.outpath = os.path.dirname(resfile)
            output_name = os.path.basename(resfile)
            output_name = os.path.splitext(output_name)[0]
            outputfile = os.path.join(self.outpath, output_name+"_FN.zip")
        ru.write_falsenegatives(resfile, gtfile, outputfile=outputfile, distance_xy=distance_xy, distance_t=distance_t)




