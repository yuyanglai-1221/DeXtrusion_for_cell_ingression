import roifile
import numpy as np
from math import sqrt, pow, floor
from numpy import array, zeros

"""
Set of functions to handle ROIs files in DeXtrusion (compare, save and write)


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
"""

def write_rois(outfile, rois, verbose=True):
    if verbose:
        print("Writing "+str(len(rois))+" Rois in file "+outfile)
    roifile.roiwrite(outfile, rois, mode='w')

def create_roi(pt, cat=1, astime=True):
    """ Create an ImageJ Roi from point coordinates """
    croi = roifile.ImagejRoi()
    croi.version = 227
    croi.roitype = roifile.ROI_TYPE(10)
    #croi.name = str(val)
    croi.name = str(pt[0]+1).zfill(4)+'-'+str(pt[1]).zfill(4)+"-"+str(pt[2]).zfill(4)
    croi.n_coordinates = 1
    croi.left = int(pt[2])
    croi.top = int(pt[1])
    croi.position = pt[0] + 1
    if astime:
        croi.z_position = 1
        croi.t_position = pt[0]+1
    else:
        croi.z_position = pt[0]+1
        croi.t_position = 1
    croi.c_position = 1
    croi.integer_coordinates = array([[0,0]])
    croi.stroke_width=3
    if cat == 1:  ## color cell extrusion
        croi.stroke_color = b'\xff\xff\x00\x00'
    if cat == 2:  ## color cell sop
        croi.stroke_color = b'\xff\x00\x00\xff'
    if cat == 3:  ## color cell division
        croi.stroke_color = b'\xff\x00\xff\x00'
    return croi

def distance_rois(roi1, roi2):
    return sqrt( pow(roi1[2]-roi2[2],2) + pow(roi1[1]-roi2[1],2) )
    
def matched_roi(roi, lroi, dxy, dt, matched):
    closest = -1
    mind = dt+dxy*1.1
    for i in range(len(lroi)):
        curroi = lroi[i]
        if abs(curroi[0]-roi[0]) <= dt:
            if distance_rois(curroi, roi) <= dxy:
                if matched is None:
                    return True
                if (matched[i]==0):
                    mixdist = distance_rois(curroi, roi)+abs(curroi[0]-roi[0])
                    if mixdist < mind:
                        mind = mixdist
                        closest = i
    if closest < 0:
        return False

    if matched is not None:
        matched[closest] = 1
        return True
    return True
    
def read_rois(resfile):
    ''' read an imageJ Rois file and return the list (z, y, x) of the rois coordinates '''
    rois =  roifile.ImagejRoi.fromfile(resfile)
    listrois = []
    for roi in rois:
        xroi = roi.left
        yroi = roi.top
        frame = max(roi.position, roi.z_position, roi.t_position)-1
        #print(str(roi.position)+" "+str(roi.t_position)+" "+str(roi.z_position)+" "+str(frame))
        listrois.append((frame, yroi, xroi))
    return listrois

def clean_rois(resfile, volthres=10, probthres=180, writefile=True, verbose=True):
    ''' read an imageJ results Rois file and return the list (z, y, x) of the rois coordinates that are above given thresholds '''
    rois =  roifile.ImagejRoi.fromfile(resfile)
    listrois = []
    for roi in rois:
        xroi = roi.left
        yroi = roi.top
        frame = max(roi.z_position, roi.t_position)-1
        name = roi.name
        #print(str(roi.position)+" "+str(roi.t_position)+" "+str(roi.z_position)+" "+str(frame))
        namepart = name.split("-")
        if (float(namepart[3]) > volthres) and (float(namepart[4]) > probthres):
            if writefile:
                listrois.append(roi)
            else:
                listrois.append((frame, yroi, xroi))
    if writefile:
        write_rois(resfile+"_cleaned.zip", listrois, verbose)
        return None
    else:
        return listrois


def get_falsepositives(mrois, grois, distance_xy, distance_t):
    fp = []
    matched = zeros(len(grois))
    for roi in mrois:
        if not matched_roi(roi, grois, distance_xy, distance_t, matched):
            fp.append(create_roi((roi[0], roi[1], roi[2])))
    return fp
    
def get_falsenegatives(mrois, grois, distance_xy, distance_t):
    fn = []
    for roi in grois:
        if not matched_roi(roi, mrois, distance_xy, distance_t, None):
            fn.append(create_roi((roi[0], roi[1], roi[2])))
    return fn

def check_positives(mrois, grois, distance_xy, distance_t):
    tp = 0
    fp = 0
    matched = zeros(len(grois))
    for roi in mrois:
        if matched_roi(roi, grois, distance_xy, distance_t, matched):
            tp += 1
        else:
            fp += 1
    return (tp, fp)

def check_falseneg(mrois, grois, distance_xy, distance_t):
    fp = 0
    for roi in mrois:
        if not matched_roi(roi, grois, distance_xy, distance_t, None):
            fp += 1
    return fp
    
def write_falsenegatives(resfile, gtfile, outputfile, distance_xy=10, distance_t=3):
        ''' Compare Rois position: count number of FP, FN '''
        myrois = read_rois(resfile)
        gtrois = read_rois(gtfile)
        
        false_neg = get_falsenegatives(myrois, gtrois, distance_xy, distance_t)
        roifile.roiwrite(outputfile, false_neg, mode='w')
    
def write_falsepositives(resfile, gtfile, outputfile, distance_xy=10, distance_t=3):
        ''' Compare Rois position: count number of FP, FN '''
        myrois = read_rois(resfile)
        gtrois = read_rois(gtfile)
        
        false_pos = get_falsepositives(myrois, gtrois, distance_xy, distance_t)
        roifile.roiwrite(outputfile, false_pos, mode='w')

def compare_rois(resfile, gtfile, distance_xy=10, distance_t=3):
        ''' Compare Rois position: count number of FP, FN '''
        myrois = read_rois(resfile)
        gtrois = read_rois(gtfile)
        
        true_pos, false_pos = check_positives(myrois, gtrois, distance_xy, distance_t)
        false_neg = check_falseneg(gtrois, myrois, distance_xy, distance_t)
        print("True positives "+str(true_pos))
        print("False positives "+str(false_pos))
        print("False negatives "+str(false_neg))
        if (true_pos+false_pos) == 0:
            print("Precision "+str(0))
            print("Recall "+str(0))
            return [true_pos, false_pos, false_neg, 0, 0]
        print("Precision "+str(true_pos/(true_pos+false_pos)))
        print("Recall "+str(true_pos/(true_pos+false_neg)))
        return [true_pos, false_pos, false_neg, true_pos/(true_pos+false_pos), true_pos/(true_pos+false_neg)]
    
