# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:32:55 2015

@author: mdmiah
"""

import matplotlib as mpl
mpl.use('Agg') # Needed to work on server

import numpy as np
import random
import sys
import modelInputs

# ----------------------------------        ----------------------------------

def getRowsForImages(u, v, labels, brisks, colors):
    X = []
    y = []
    meta = []
    
    for i in xrange(u*4, (u*4)+4):
        for j in xrange(v*4, (v*4)+4):
            Xrow, yrow, metarow = modelInputs.getRowForCombination(i, j, labels, brisks, colors)
            X.append(Xrow)
            y.append(yrow)
            meta.append(metarow)
    
    return X, y, meta

def sampleVariant(labels, brisks, colors, start=0, end=None):
    if end is None:
        end = colors.shape[0]-1
    
    # Choose an image at random
    u = random.randint(start, end)
    label1 = labels[u][1]
    
    # Make sure minimum number of BRISK features exist
    i = u*4 + np.arange(4)
    brisk1 = brisks[i,1:]
    # If ANY of them have brisk features of lower than the threshold, then skip
    if np.sum( np.sum(brisk1,axis=1)<modelInputs.minBriskFeatures ):
        return sampleVariant(labels, brisks, colors, start, end)
    
    # Match with a color variant
    for m in xrange(20): # Search nearby for a color variant
        v = random.randint(u-50, u+50)
        v = start if v<start else end if v>end else v
        if u==v: # Don't match with itself
            continue
        label2 = labels[v][1]
        # Make sure minimum number of BRISK features exist
        j = v*4 + np.arange(4)
        brisk2 = brisks[j,1:]
        if np.sum( np.sum(brisk2,axis=1)<modelInputs.minBriskFeatures ):
            continue
        if label1==label2:
            return getRowsForImages(u, v, labels, brisks, colors)
    
    # If the randomly chosen image has no color variant,
    # sample again
    return sampleVariant(labels, brisks, colors)

def sampleNonVariant(labels, brisks, colors, start=0, end=None):
    if end is None:
        end = colors.shape[0]-1
    
    # Choose an image at random
    u = random.randint(start, end)
    label1 = labels[u][1]
    
    # Make sure minimum number of BRISK features exist
    i = u*4 + np.arange(4)
    brisk1 = brisks[i,1:]
    # If ANY of them have brisk features of lower than the threshold, then skip
    if np.sum( np.sum(brisk1,axis=1)<modelInputs.minBriskFeatures ):
        return sampleNonVariant(labels, brisks, colors, start, end)
    
    # Match random non variants
    label2 = label1
    while label1==label2:
        v = random.randint(start, end)
        # Make sure minimum number of BRISK features exist
        j = v*4 + np.arange(4)
        brisk2 = brisks[j,1:]
        if np.sum( np.sum(brisk2,axis=1)<modelInputs.minBriskFeatures ):
            continue
        label2 = labels[v][1]

    return getRowsForImages(u, v, labels, brisks, colors)

def save(no_of_pairs = 10000):
    labels, brisks, colors = modelInputs.loadHists()
    
    test_start_n = no_of_pairs * modelInputs.train_fraction # Fraction of dataset used in training
    test_start_u = np.int((colors.shape[0]-1) * modelInputs.train_fraction) # Fraction of images used in training
    
    with open("Cache/X1.csv", "w") as X_fh, open("Cache/y1.csv", "w") as y_fh, open("Cache/Xmeta1.csv", "w") as meta_fh:
        for n in xrange(no_of_pairs):
            if n<test_start_n:
                start = 0
                end = test_start_u-1
            else:
                start = test_start_u
                end = None
            
            X, y, meta = sampleVariant(labels, brisks, colors, start, end)
            X2, y2, meta2 = sampleNonVariant(labels, brisks, colors, start, end)
            X.extend(X2)
            y.extend(y2)
            meta.extend(meta2)
            
            np.savetxt(X_fh, X, delimiter=",", fmt="%f")
            np.savetxt(y_fh, y, delimiter=",", fmt="%d")
            np.savetxt(meta_fh, meta, delimiter=",", fmt="%d")
            
            if (n+1)%1000==0:
                percentage_completion = 100.0*np.float(n+1)/no_of_pairs
                sys.stdout.write(str(n+1)+" of "+str(no_of_pairs)+" done ("+str(percentage_completion)+"%)\r")
                sys.stdout.flush()
    print ""


