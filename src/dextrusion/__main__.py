"""
        DeXtrusion: detect events from movie script

        Detect extrusions (or another event) on a movie.

    Used already trained neural network(s) to detect extrusions or other events from movie(s).

    DeXtrusion calculates a probability map which reflects the probability of having an extrusion (or other event) for each pixel at each time of the movie. The position of an event will be at the centroid of a high probability volume.

    
"""

import os, sys
import dextrusion.DialogDeXtrusion as dd
import dextrusion.DeXtrusion as dextrusion
from glob import glob


def get_models(modeldir, talkative=True):
    ''' Get all the models in the "modeldir" if it contains several subdirectory, else take only current model '''
    if not os.path.exists( os.path.join(modeldir,"config.cfg") ):
        return glob(modeldir+"/*/", recursive = False)
    else:
        return [modeldir]


def main():
    # default parameters
    talkative = True  ## print info messages
    groupsize = 150000       ## number of windows that are runned through the neural network at the same time - depends on the computer processing capabilities
                         ## higher will run faster, but too high can fail
    imname = os.getcwd()    ## folder where data are saved to initialize the file browser
    modeldir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "DeXNets", "notum_ExtSOPDiv")  ## folder where models are saved to initialize the file browser
    dexter = dextrusion.DeXtrusion(verbose=talkative)

    # Choose parameters through graphical interface
    dexter = dextrusion.DeXtrusion(verbose=talkative)
    diag = dd.DialogDeXtrusion()
    diag.dialog_main(modeldir, imname)

    cell_diameter = float(diag.cell_diameter)
    extrusion_duration = float(diag.extrusion_duration)
    shift_xy = int(diag.dxyval)
    shift_t = int(diag.dtval)
    save_proba = bool(diag.saveProba)
    save_proba_one = bool(diag.saveProbaOne)
    save_rois = bool(diag.saveRois)
    cat = int(diag.event_ind)
    catnames = diag.event
    roi_threshold = float(diag.threshold)  ## min average probability in volume to keep ROI
    roi_volume = float(diag.proba_vol)     ## min positive volume to keep ROI
    groupsize = int(diag.group_size)

    modeldir = diag.modeldir
    model = get_models(modeldir, talkative)

    imname = diag.imagepath
    if talkative:
        print("Using network(s) "+str(model))
        print("On movie(s) "+str(imname))

    ## Merge "peaks" if distance < disxy (spatial) & dist (time) 
    disxy = 10
    distime = 4
    # Parameters: astime: save tif image as temporal stack (else slice stack)
    temporal = True

    for image in imname:
        print(image)
        if os.path.exists(image):
            if talkative:
                print( "Detecting events on movie "+str(image) )
                                                
            ## Detect events
            dexter.detect_events_onmovie( image, models=model, 
                                          cell_diameter=cell_diameter, extrusion_duration=extrusion_duration,                                                                           dxy=shift_xy, dz=shift_t,                                                                                                                   group_size=groupsize )
                                                                          
            ## Save probability maps to file, to visualize the results
            if save_proba:
                dexter.write_probamaps(cat=None, astime=temporal)
                                                                                                        
            ## Save cleaned probability map(s) to file, keeping only events above the size and probability thresholds (for visualisation) 
            if save_proba_one:
                dexter.write_cleanedprobamap(cat=cat, volume_threshold=roi_volume, proba_threshold=roi_threshold, disxy=disxy, distime=distime, astime=temporal)
                                                                                                                                                        ## Look for centroid of "positive" volumes in the probability maps and convert it to a point event 
            ## Keep only ROIs that have a big enough volume ('volume_threshold') and with high enough probability ('proba_threshold') 
            # - put 0 and 0 to keep all detections
            if save_rois:
                dexter.get_rois(cat=cat, volume_threshold=roi_volume, proba_threshold=roi_threshold, disxy=disxy, dist=distime)

    return 1 

if __name__ == '__main__':
        sys.exit(main()) 
