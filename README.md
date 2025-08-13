# ![DeXtrusion](https://github.com/yuyanglai-1221/DeXtrusion_for_cell_ingression/blob/main/images/DeX.png) DeXtrusion

DeXtrusion is  a machine learning based python pipeline to detect cell extrusions in epithelial tissues movies. It can also detect cell divisions and SOPs, and can easily be trained to detect other dynamic events. 

This repository contains the code source of the python DeXtrusion library, the trained neural networks (ready to use) with Jupyter notebooks to run, train or retrain DeXtrusion networks, and Fiji macros for visualization/analysis of the results.

* [Presentation](#presentation)
* [Installation](#installation)
* [Usage](#usage)
    * [Troubleshooting](#trouble-shooting)
* [Data](#data)
* [DeXtrusion networks](#dexnets-dextrusion-neural-networks)
* [Fiji macros](#fiji-macros)
* [Jupyter notebooks](#jupyter-notebooks)
* [References](#references)
* [Modification and Comments from Yuyang](#modification-and-comments-from-yuyang)

## Presentation
DeXtrusion takes as input a movie of an epithelium and outputs the **spatio-temporal location of cell extrusion events** or other event as cell divisions. 
The movie is discretized into small overlapping rolling windows which are individually [classified for event detection](#event-classification) by a trained [neural network](#deXNets-deXtrusion-neural-networks). Results are then put together in event [probability map](#event-probability-map) for the whole movie or as [spatio-temporal points](#event-spot-localization) indicating each event. 

<p align="center">
  <img src="https://gitlab.pasteur.fr/gletort/dextrusion/-/raw/main/images/SequenceExtrusion.png" alt="Extrusion detection example" width="600">
  <br>
  <em>Example of the detection of an extrusion event (probability map, red).</em>
</p>



### Event classification
The movie is discretized into small windows that span all the movie in a rolling windows fashion with overlap between the different windows. 
Each window is classified by a trained neural network as containing or not a cellular event as cell extrusion. The different classes available in the main DeXtrusion network are:
- No event
- Cell extrusion/cell death
- Cell division
- SOPs

It is easy to add or remove a cellular event to this list, but necessitates to train new neural networks for this. Jupyter notebook is available in this repository to do it.

### Event probability map
Each window is associated an event probability which allow to generate an events probability map on the whole movie. This probability maps can be refined to precise spatio-temporal spots for each event.
The results can be visualized by overlaying the initial movie and all the probability maps saved by DeXtrusion in Fiji with the saved by DeXtrusion with the [`deXtrusion_overlayProbabilityMaps.ijm`](https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/ijmacros/deXtrusion_overlayProbabilityMaps.ijm) macro.

<p align="center">
  <img src="https://github.com/yuyanglai-1221/DeXtrusion_for_cell_ingression/blob/main/images/SequenceProbamaps.png?raw=1" alt="Example probability maps">
</p>

Example of probability maps (green: division, red: extrusion, blue: SOP

### Event spot localization
An event is visible in the probability map as a volume of connected pixels of high probability. To convert the probability map to a list of event, we place the event localization at the center of each of these high probability volumes.
Moreover, to reduce false positive detections, the volumes can be thresholded to keep only big enough volume of high enough probability values. 
The list of spots obtained in this way are saved in ROIS `.zip` file that can be open in Fiji through the `ROIManager` tool. The macro [`deXtrusion_showROIs.ijm`](#https://gitlab.com/yuyanglai-1221/DeXtrusion_for_cell_ingression/blob/main/ijmacros/deXtrusion_showROIs.ijm) allows to directly visualize with Fiji the results saved by DeXtrusion. 


## Installation
DeXtrusion is distributed as a python module, compatible with `python3`. 
You can install it in a virtual environment to be sure to have the required versions.
**DeXtrusion** can be installed either manually through the setup file or with pip:

* DeXtrusion can be directly installed via pip. In your python virtual environment, enter: `pip install dextrusion` to install it.
You should also download the trained neural network [DeXNet](#dexnets-dextrusion-neural-networks) that you want to use from this repository.

* To install manually **DeXtrusion**, clone this github repository, and inside this repository type: `python setup.py install`. 
If you want to install it in a virtual environment, you should have activated it before.
It will install all the necessary dependencies.


### Detailled installation on Linux (Ubuntu)
Here's a step by step command lines to run in a Terminal to install DeXtrusion on Linux from scratch. It is creating a new virtual environment on which DeXtrusion will be installed (not necessary but recommended). Tested with `python 3.8`.

* Installation with `pip`
```sh
python3 -m venv dextrusenv               ## create a virtual environment with python 3 called dextrusenv (you can choose the name)
source dextrusenv/bin/activate           ## activate the virtual environment: now python commands will be runned in that environment
pip install dextrusion                  ## install DeXtrusion, download all the necessary dependencies, can take time
```
DeXtrusion is installed and can be runned with:

```sh
python -m dextrusion 	                  ## run DeXtrusion. Next time to run it again with the same networks, you only have to do this line
```

DeXtrusion needs neural networks to classify each sliding window. To used our trained neural networks (in the DeXNets repository), you have to download and uncompress it. From the command line:

```sh
mkdir DeXNets							## create directory on which neural networks will be put
mkdir DeXNets/notum_ExtSOPDiv			## create directory for the neural networks traind with 4 events available in this repository
cd DeXNets/notum_ExtSOPDiv               ## go inside the desired neural networks directory (here the network trained with 4 events)
wget https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/DeXNets/notum_ExtSOPDiv/notumExtSOPDiv0.zip   ## download the first neural network
wget https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/DeXNets/notum_ExtSOPDiv/notumExtSOPDiv1.zip   ## download the second neural network
unzip notumExtSOPDiv0.zip                ## uncompress it. It is now ready to be used
unzip notumExtSOPDiv1.zip
cd ../..                                 ## go back to main DeXtrusion directory
```
 
* You can also clone this repository and install DeXtrusion from the source code `python setup.py install` instead of using pip installation.


### Detailled installation on Windows
You need to have anaconda installed to run python and create a new virtual environment. On the command prompt, write:
```cmd
$ C:\ProgramData\Anaconda3\Scripts\Activate     ## activate conda (should be the path of your anaconda installation)
$ conda create --name dextrusionenv python=3.8  ## to create virtual environment called dextrusionenv
$ conda activate dextrusionenv                  ## activate the virtual environment that you juste created
$ pip install dextrusion                        ## install DeXtrusion in it
$ python -m dextrusion                          ## Run it
```

### Choose the neural network DeXNets to use
DeXtrusion needs neural networks to classify each sliding window. To used our trained neural networks (in the [DeXNets repository](./DeXNets)), you have to download and uncompress it. 
From DeXtrusion user interface, to select a network to run, you have to go inside its uncompressed directory (click on `Browse` and go there). If you want to use several networks together for the detection for better results (see our paper), the networks folders should be placed within the same directory. Go inside that directy with the interface, and it will use all the networks presents in that directory.

## Usage

DeXtrusion can be used directly within python. 
To run an already trained network and detect cell events, run `python3 -m dextrusion` in your virtual environment. It will open a graphical interface to perform the detection.

We also propose [Jupyter notebooks](https://gitlab.pasteur.fr/gletort/dextrusion/-/tree/main/jupyter_notebooks) in this repository, to use dextrusion options. 
You can find a notebook to perform the detection, train a new neural network, retrain a network or evaluate the performance of the network compared to a manual detection.

### Trouble shooting

Encountered errors/issues and solutions:

#### Internal error copying input tensor

If you get this error message, finishing by `InternalError: Failed copying input tensor from ... ` it is likely to be an `Out of Memory` problem if you are trying to run DeXtrusion with too many windows running in parallel (at once). To change this, decrease the `groupsize` parameter which controls this number.


<details>
<summary>
See full error message
</summary>

```
---------------------------------------------------------------------------
InternalError                             Traceback (most recent call last)
Cell In[7], line 14
     11     print( "Detecting events on movie "+str(image) )
     13 ## Detect events
---> 14 dexter.detect_events_onmovie( image, models\=model, 
     15 cell_diameter\=cell_diameter, extrusion_duration\=extrusion_duration, 
     16 dxy\=shift_xy, dz\=shift_t, 
     17 group_size\=groupsize )
     19 ## Save probability maps to file, to visualize the results
     20 if save_proba:

File ~\\AppData\\Local\\miniforge3\\envs\\dextrusion_env\\lib\\site-packages\\dextrusion\\DeXtrusion.py:375, in DeXtrusion.detect_events_onmovie(self, moviepath, models, cell_diameter, extrusion_duration, dxy, dz, group_size, outfolder)
    373     os.makedirs(self.outpath)
    374 self.outname \= os.path.join(self.outpath, output_name)
--> 375 self.detect_events(img, models, cell_diameter, extrusion_duration, dxy, dz, group_size\=group_size)
    377 if self.verbose:
    378     print("---- Detection finished, took %s minutes ---" % ((time.time() - start_time)/60))

File ~\\AppData\\Local\\miniforge3\\envs\\dextrusion_env\\lib\\site-packages\\dextrusion\\DeXtrusion.py:434, in DeXtrusion.detect_events(self, img, model_paths, cell_diameter, extrusion_duration, dxy, dz, group_size)
    432         print("   Doing windows with shift: "+str(self.shiftxy)+" "+str(self.shiftz)+" and model "+model_paths[cmod])
    433         print("   Number of windows to categorize: "+str(self.nw[4]))
--> 434     do \= list(map(functools.partial(self.do_one_group, img, pred_channels, npred), range(0,self.nw[4],self.group_size) ))
    436 ## divide result map by the number of predictions added in each pixel
    437 self.divide(pred_channels, npred)

File ~\\AppData\\Local\\miniforge3\\envs\\dextrusion_env\\lib\\site-packages\\dextrusion\\DeXtrusion.py:302, in DeXtrusion.do_one_group(self, img, pred_channels, npred, group)
    300 img_group \= np.array(img_group)/255 # normalize
    301 self.model.reset(self.model_path)
--> 302 pred \= self.model.predict_batch(img_group)
    304 # Fill the proba results in the correct place on the pred_channels image
    305 for c in range(self.ncat-1):

File ~\\AppData\\Local\\miniforge3\\envs\\dextrusion_env\\lib\\site-packages\\dextrusion\\Network.py:182, in Network.predict_batch(self, img_batch)
    181 def predict_batch(self, img_batch):
--> 182     return self.model.predict(img_batch)

File ~\\AppData\\Local\\miniforge3\\envs\\dextrusion_env\\lib\\site-packages\\keras\\utils\\traceback_utils.py:67, in filter_traceback.<locals>.error_handler(*args, **kwargs)
     65 except Exception as e:  # pylint: disable=broad-except
     66   filtered_tb \= _process_traceback_frames(e.__traceback__)
---> 67   raise e.with_traceback(filtered_tb) from None
     68 finally:
     69   del filtered_tb

File ~\\AppData\\Local\\miniforge3\\envs\\dextrusion_env\\lib\\site-packages\\tensorflow\\python\\framework\\constant_op.py:102, in convert_to_eager_tensor(value, ctx, dtype)
    100     dtype \= dtypes.as_dtype(dtype).as_datatype_enum
    101 ctx.ensure_initialized()
--> 102 return ops.EagerTensor(value, ctx.device_name, dtype)

InternalError: Failed copying input tensor from /job:localhost/replica:0/task:0/device:CPU:0 to /job:localhost/replica:0/task:0/device:GPU:0 in order to run _EagerConst: Dst tensor is not initialized. 
```

</details>


#### 


## Data

Data used for the training of our neural networks (raw movies with manual anotation of events) are freely available on Zenodo: [https://doi.org/10.5281/zenodo.7586394](https://zenodo.org/records/7586394). 

In the Zenodo repository, you have movies and the corresponding manually annotated ROI files. The ROIs will have the same name as the movie, followed, by "_cell_*event*.zip", where `event` is the corresponding event that were annotated (death/extrusion, division, sop).

The description of all these movies (microscope used, biological marker, resolution...) are given in the [Supplemental Table S1](https://cob.silverchair-cdn.com/cob/content_public/journal/dev/150/13/10.1242_dev.201747/1/dev201747supp.pdf?Expires=1742676929&Signature=LcovagyqTBBmVv-YTgKCBOhFvT2-oJ1CFMn4GZQm4gZjjP2V5eeI9qCn~DIg6XRzLAoSd1h7q2BLcX8AelIhF3WogXQIMNRJMKBQxXzQmHSS0yDEvRu2aA9CAPRbk1L5P9k8vlPBQ389TraYMSor~2vSGf5sC4cRcZg5sKvCPLaCZpJiZYTZvnx9OYauV0cIFod1N3UZf48nJEXCeCaNx70709OwwIWVPvvjA~ZyutlET4f~ZpZq3vWDcCZsu7aYfDdTjpDAYTiyNf~qodN3N5zBKLmr~8nHY6xgUtQvpLiOyRVWjxOoMjs3M3sYWMP2h-txXug0VWw606cNJX7YAA__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA) of our publication. The number in the `Movie` column in the table corresponds to the number in the name of the movie in the Zenodo.


## DeXNets: deXtrusion neural networks
DeXtrusion was trained on E-cadherin tagged drosophilia notum but can be easily adapted to other markers/biological conditions. Retraining of DeXtrusion network may be necessary when images are quite different.

In the `deXNets` folder of this repository, we proposed several trained networks:
- `notum_Ext`: trained on drosophilia notum movies with only extrusion events
- `notum_ExtSOP`: trained on drosophilia notum movies with extrusion and SOP events
- `notum_ExtSOPDiv`: trained on drosophilia notum movies with extrusion, division and SOP events
- `notum_all`: trained on drosophilia notum movies with extrusion, division and SOP events, on all our annotated data (training and test). Networks to use by default.

Download them and unzip to be ready to use in DeXtrusion.

### Train/retrain a new DeXNet
If you want to train/retrain a neural network, to add a new event to detect or to improve its performance on new datasets, Jupyter notebooks are proposed in this repository: [Jupyter notebooks](https://gitlab.pasteur.fr/gletort/dextrusion/-/tree/main/jupyter_notebooks). You can just follow step by step the notebook.

If you want to change the architecture of DeXNet, you will have to change it in the `src/dextrusion/Network.py` file. The architecture is defined in the `action_model` function.


## Fiji macros

Fiji macros to prepare data for DeXtrusion neural network training or to handle DeXtrusion results are available in the [`Fiji macros`](https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/ijmacros/) folder of this repository.

* [deXtrusion_checkAndClearROIs.ijm](https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/ijmacros/deXtrusion_checkAndClearROIs.ijm) allows to check manually all the detected ROIs and to keep only the correct ones. 
It shows each event (point ROI) detected by deXtrusion as a pop-up window centered temporally and spatially around the event.
The user is asked wether the detection is correct (if there is an event in the current window) by pressing `y` or `n`.
If yes, the ROI is kept, else it will be removed from the list of ROIs.


* [deXtrusion_scoreROIs_Random.ijm](https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/ijmacros/deXtrusion_scoreROIs_Random.ijm) allows to score manually the precision of the detected events.
The macro will open randomly a given number of ROIs, and show a window centered temporally and spatially around the event.
The user is asked wether the detection is correct (if it contains the event) by pressing `y` or `n`.
The macro will output the count of correct and false detections and the precision. 
The ROIs file will not be edited and the ROIs left after running this macro should not be saved as it deletes all the ROIs explored to be sure not to draw them twice.

* [deXtrusion_overlayProbabilityMaps.ijm](https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/ijmacros/deXtrusion_overlayProbabilityMaps.ijm) allows to visualize the probability maps together with the original image.
The user is asked for the original image to visualize and will open the probability maps in the `results` folder where they should be saved by default.
The directory to look in can be changed in the macro. 

* [deXtrusion_showROIs.ijm](https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/ijmacros/deXtrusion_showROIs.ijm) displays the original movie (chosen by the user) and the ROI file found in the `results` folder. 
The location of the ROI file can be changed in the macro by changing the `resfold` parameter and the event to show (by default extrusion) can be changed in the `type` parameter.

* [deXtrusion_subsetMovieAndRois.ijm](https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/ijmacros/deXtrusion_subsetMovieAndRois.ijm) save a temporal subset of a movie, and keep only the corresponding ROIs. 


## Jupyter notebooks
We provide in this repository several jupyter notebooks to run, train/retrain, evaluate DeXtrusion. 
* [detectEventsOnMovie.ipynb](https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/jupyter_notebooks/detectEventsOnMovie.ipynb) allows you to run DeXtrusion with the graphical interaface to choose the parameters. 
* [deXtrusion_TrainNetwork.ipynb](https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/jupyter_notebooks/deXtrusion_TrainNetwork.ipynb) to train a new DeXNet network with your own data (or using our data publicly available, see [data section](#Data)).
* [deXtrusion_RetrainNetwork.ipynb](https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/jupyter_notebooks/deXtrusion_RetrainNetwork.ipynb) to retrain an already trained neural network with new data.
* [deXtrusion_CompareRois.ipynb](https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/jupyter_notebooks/deXtrusion_CompareRois.ipynb) to compare two ROIs files (manual detection vs DeXtrusion detection) and gives the score. It can also generate ROIs files of the false detection (false positives and/or false negatives).
* [deXtrusion_testClassification_OnWindows.ipynb](https://gitlab.pasteur.fr/gletort/dextrusion/-/blob/main/jupyter_notebooks/deXtrusion_testClassification_OnWindows.ipynb): run the trained neural network (DeXNet) on test data on generated windows, and print the resulting confusion matrix of the classification (DeXtrusion classification compared to the manual classification).

#### Notebooks usage
* If you have Jupyter installed on your computer, to add your virtual environment to the environment Jupyter can see:
```
ipython kernel install --user --name=dextrusionenv
```
You can refer to this [tutorial](https://www.geeksforgeeks.org/using-jupyter-notebook-in-virtual-environment/) for instructions on creating virtual environment and using it with jupyter.

* You can also install Jupyter Lab or Jupyter notebook in your virtual environment if you don't have it installed in your computer:
```sh
pip install jupyterlab
jupyter-lab
```
Refer to [Jupyter](https://jupyter.org/) webpage for more information.

## References
To know more about DeXtrusion, refer to our paper: [Villars, Letort et al. *Development* 2023](https://journals.biologists.com/dev/article/150/13/dev201747/321184/DeXtrusion-automatic-recognition-of-epithelial).

## License
DeXtrusion is distributed open-source under the BSD-3 license, see the license file in this repository.

When you use DeXtrusion source code, neural networks or data for your projects, please cite our paper. 

## Modification and Comments from Yuyang
The above README is the original version for dextrusion. In this section, I will show you how to create an environment for DeXtrusion on Apple Silicon Mac; the Script floder which used to introduce the velocity field into the movie cropping; I will also explain the notebook that retraining and testing part I used.

### Installing DeXtrusion on Apple Silicon Mac (CPU version of TensorFlow 2.15)
1. Create a new environment
```
conda create -n dextrusion-mac311 python=3.11 -y
conda activate dextrusion-mac311
```
2. Install TensorFlow (CPU version) and core dependencies
```
pip install -U pip setuptools wheel
conda install -y -c conda-forge "tensorflow=2.15.*" "tensorflow-estimator=2.15.*" "tensorboard=2.15.*" "numpy>=1.26,<2"
```
3. Install scientific computing and image processing dependencies required by DeXtrusion
```
pip install "scikit-image==0.21.*" "tifffile>=2023.8" "pillow<11" \
         "opencv-python==4.8.1.78" "matplotlib<3.8" "roifile==2021.6.6" \
         "scikit-learn==1.4.*" ipython
```
4. Install DeXtrusion (skip old TF dependencies)
```
pip install "dextrusion==0.0.7" --no-deps
```
5. Verify installation
```
python - <<'PY'
import tensorflow as tf, numpy as np, dextrusion
print("TF:", tf.__version__)
print("NumPy:", np.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
print("dextrusion OK")
PY
```
Expected output:
```
TF: 2.15.x
NumPy: 1.26.x
GPUs: []
dextrusion OK
```
6. Launch the DeXtrusion GUI
```
python -m dextrusion
```
### Script folder
The script folder here includes two files:
* [flow_gen.py](https://github.com/yuyanglai-1221/DeXtrusion_for_cell_ingression/blob/main/Script/flow_gen.py) is modified from [MovieGeneratorFromROI.py](https://github.com/yuyanglai-1221/DeXtrusion_for_cell_ingression/blob/main/src/dextrusion/MovieGeneratorFromROI.py), which build a class `FlowFollowingMovieGeneratorFromROI` with function `__init__` for initializing the order of sampling points, `get_roi_img` for getting the moving cropped movies from ROIs and velocity field, and `_sample_velocity_at_global` for calculating the velocity for a given point by bilinear weighted sum of the four surrounding samples.
* [test_flow_crop.py](https://github.com/yuyanglai-1221/DeXtrusion_for_cell_ingression/blob/main/Script/test_flow_crop.py) can extract the cropped movie with fixed center or flow center from the ROIs. It requires a `PIV_field.mat` which records the coordinates of sampling points `(x,y)`, and their velocity `(u,v)`, and it also needs a `tissuemovie.tif`.

### Yuyang's tips for retraining and testing
For the retraining:
* One needs to make a file folder with a `A.tif` and its ROIs with the name `A_cell_death.zip` (for cell ingression), `A_cell_division.zip` (for cell division), `A_sop.zip` (for SOP), and `A_nothing` (for nothing).
* The retraining code can added nothing case automatically, since the events of cell ingression and cell division are so rare, I did not turn it off.
* The retraining is based on an original model, and dextrusion paper provides eight: two for [notum_all_original](https://github.com/yuyanglai-1221/DeXtrusion_for_cell_ingression/tree/main/DeXNets/notum_all/notum_all_original), two for [notum_Ext](https://github.com/yuyanglai-1221/DeXtrusion_for_cell_ingression/tree/main/notum_Ext), two for [notum_ExtSOP](https://github.com/yuyanglai-1221/DeXtrusion_for_cell_ingression/tree/main/notum_ExtSOP), and two for [notum_notum_ExtSOPDiv](https://github.com/yuyanglai-1221/DeXtrusion_for_cell_ingression/tree/main/notum_ExtSOPDiv). My retrained models lie in [notum_all_retrain](https://github.com/yuyanglai-1221/DeXtrusion_for_cell_ingression/tree/main/DeXNets/notum_all/notum_all_retrain).
* I felt sorry that I have mistaken the definition of accuracy with the one in eLife paper: the accuracy in the training means the ratio that the number of right predictions in the validation sets to the total number of it.

For the testing:
* [deXtrusion_CompareRois.ipynb](https://github.com/yuyanglai-1221/DeXtrusion_for_cell_ingression/blob/main/jupyter_notebooks/deXtrusion_CompareRois.ipynb) can compare two different ROIs and get the precision and recall. I have also added a 4th part in this Jupyter notebook which can use two probability maps to get a mixed one.
* [deXtrusion_GetRoisFromRawProbaMaps.ipynb](https://github.com/yuyanglai-1221/DeXtrusion_for_cell_ingression/blob/main/jupyter_notebooks/deXtrusion_GetRoisFromRawProbaMaps.ipynb) can generate ROIs from probability maps. I didn't know how to use the second part of the Jupyter notebook so I wrote a third part to do it.
