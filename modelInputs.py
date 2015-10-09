# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:32:55 2015

@author: mdmiah
"""

import matplotlib as mpl
mpl.use('Agg') # Needed to work on server

import numpy as np
import pandas as pd
import json
import random
import sys
import cv2
from scipy.spatial import distance
import brisk
import colorScheme

count = 10000
train_fraction = 0.8
minBriskFeatures = 30

# ----------------------------------        ----------------------------------

def getImage(i):
    path = "Data/images/"+str(i)+".jpg"
    img = cv2.imread(path)
    if img is None or img.size==0:
        return None
    img = cv2.resize(img, (500, 500))
    return img

def getImageRGBK(i):
    img = getImage(i)
    imgB = img[:,:,0]
    imgG = img[:,:,1]
    imgR = img[:,:,2]
    imgK = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return imgR, imgG, imgB, imgK

# ----------------------------------        ----------------------------------

def saveLabels(count=count):
    labels = pd.DataFrame()
    
    # Extract labels
    with open('Data/colour_duplicates.json') as fh:
        i = 0
        for line in fh:
            line = json.loads(line)
            code = str.replace(str(line["code"]), "-", "")
            labels = labels.append({"code":np.uint(code)}, ignore_index=True)
            i += 1
            if i>=count:
                break;
    
    # Save labels to file
    labels.index.name = "index"
    labels.code = labels.code.astype(np.uint)
    labels.to_csv("Data/labels1.csv")

def briskHist(img):
    des = brisk._features(img)
    hist = brisk._kmeans(des)
    return hist

def briskHistsRGBK(i):
    imgR, imgG, imgB, imgK = getImageRGBK(i)
    histR = briskHist(imgR)
    histG = briskHist(imgG)
    histB = briskHist(imgB)
    histK = briskHist(imgK)
    # Save info about the hists
    histR.insert(0, i)
    histR.insert(1, 0)
    histG.insert(0, i)
    histG.insert(1, 1)
    histB.insert(0, i)
    histB.insert(1, 2)
    histK.insert(0, i)
    histK.insert(1, 3)
    return histR, histG, histB, histK

def saveBriskHists(count=count):
    with open("Data/BRISKfeatures1.csv","w") as f_handle:
        for i in xrange(count):
            hists = briskHistsRGBK(i)
            hists = np.array(hists, dtype=np.uint)
            np.savetxt(f_handle, hists, delimiter=",", fmt="%d")

def saveColorHists(count=count):
    with open("Data/colorHist1.csv", "w") as f_handle:
        for i in xrange(count):
            hist, colors = colorScheme.getColorHist(i)
            hist = np.insert(hist, 0, i) # Save index of the image
            #colors = np.insert(colors, 0, i) # Save index of the image
            
            np.savetxt(f_handle, [hist], delimiter=",", fmt="%f")

def savePreprocessing(count=count):
    saveLabels(count)
    saveBriskHists(count)
    saveColorHists(count)

def loadHists():
    with open("Data/labels1.csv") as labels_fh, open("Data/BRISKfeatures1.csv") as brisk_fh, open("Data/colorHist1.csv") as color_fh:
        labels = pd.read_csv(labels_fh, sep=",", header=0, \
            true_values=['M'], false_values=['F']).values.astype(np.uint)

        brisks = pd.read_csv(brisk_fh, sep=",", header=None).values.astype(np.int)
        brisks = brisks[:,1:]

        colors = pd.read_csv(color_fh, sep=",", header=None).values.astype(np.float)
        colors = colors[:,1:]
        
        return labels, brisks, colors
    
    return None, None, None

# ----------------------------------        ----------------------------------

def getRowForCombination(i, j, labels, brisks, colors):
    # X
    u = np.int(i/4)
    label1 = labels[u][1]
    brisk1 = brisks[i, 1:]
    color1 = colors[u]
    b1sum = np.sum(brisk1) # Total number of BRISK features
    
    v = np.int(j/4)
    label2 = labels[v][1]
    brisk2 = brisks[j, 1:]
    color2 = colors[v]
    b2sum = np.sum(brisk1) # Total number of BRISK features
    
    # Cosine distances of histograms
    briskDist = distance.cosine(brisk1, brisk2)
    colorDist = distance.cosine(color1, color2)
    
    # ratio of each of the features
    with np.errstate(divide='ignore', invalid='ignore'):
        bRatio = np.asfarray(brisk1)/np.asfarray(brisk2)
        bRatio[(brisk2==0)|np.isnan(brisk1)|np.isnan(brisk2)] = 0
        bRatio[bRatio>1.0] = 1.0/bRatio[bRatio>1.0]
    # ratio of the total number of features
    bSumRatio = 0 if b1sum==b2sum==0 else b1sum/b2sum if b1sum<b2sum else b2sum/b1sum
    # ratio of the dominant colors
    with np.errstate(divide='ignore', invalid='ignore'):
        cRatio = color1/color2
        cRatio[(color2==0)|np.isnan(color1)|np.isnan(color2)] = 0
        cRatio[cRatio>1.0] = 1.0/cRatio[cRatio>1.0]
    
    extras = [briskDist, colorDist, b1sum, b2sum, bSumRatio]
    
    # y
    isVariant = label1==label2
    
    Xrow = np.concatenate((extras, brisk1, brisk2, cRatio)) # bRatio, color1, color2,
    yrow = [isVariant]
    metarow = [u, v, i, j]
    
    # Fix any invalid values
    Xrow[np.isnan(Xrow)] = 0
    Xrow[np.isinf(Xrow)] = 0
    
    return Xrow, yrow, metarow

def sampleVariant(labels, brisks, colors, start=0, end=None):
    if end is None:
        end = colors.shape[0]-1
    
    # Choose an image at random
    i = random.randint(start, end)
    u = np.int(i/4)
    label1 = labels[u,1]
    
    # Make sure minimum number of BRISK features exist
    brisk1 = brisks[i,1:]
    if np.sum(brisk1)<minBriskFeatures:
        return sampleVariant(labels, brisks, colors, start, end)
    
    # Match with a color variant
    for m in xrange(20): # Search nearby for a color variant
        j = random.randint(i-50, i+50)
        j = start if j<start else end if j>end else j
        v = np.int(j/4)
        if u==v: # Don't match with itself
            continue
        label2 = labels[v,1]
        brisk2 = brisks[j,1:] # Make sure minimum number of BRISK features exist
        if np.sum(brisk2)<minBriskFeatures:
            continue
        if label1==label2:
            return getRowForCombination(i, j, labels, brisks, colors)
    
    # If the randomly chosen image has no color variant,
    # sample again
    return sampleVariant(labels, brisks, colors, start, end)

def sampleNonVariant(labels, brisks, colors, start=0, end=None):
    if end is None:
        end = colors.shape[0]-1
    
    # Choose an image at random
    i = random.randint(start, end)
    u = np.int(i/4)
    label1 = labels[u,1]
    
    # Make sure minimum number of BRISK features exist
    brisk1 = brisks[i,1:]
    if np.sum(brisk1)<minBriskFeatures:
        return sampleNonVariant(labels, brisks, colors, start, end)
    
    # Match random non variants
    label2 = label1
    while label1==label2:
        j = random.randint(start, end)
        v = np.int(j/4)
        brisk2 = brisks[j,1:] # Make sure minimum number of BRISK features exist
        if np.sum(brisk2)<minBriskFeatures:
            continue
        label2 = labels[v,1]

    return getRowForCombination(i, j, labels, brisks, colors)

def save(no_of_pairs = 40000):
    labels, brisks, colors = loadHists()
    
    test_start_n = no_of_pairs * train_fraction # Fraction of dataset used in training
    test_start_i = np.int((colors.shape[0]-1) * train_fraction) # Fraction of images used in training
    
    with open("Cache/X0.csv", "w") as X_fh, open("Cache/y0.csv", "w") as y_fh, open("Cache/Xmeta0.csv", "w") as meta_fh:
        for n in xrange(no_of_pairs):
            if n<test_start_n:
                start = 0
                end = test_start_i-1
            else:
                start = test_start_i
                end = None
            
            X, y, meta = sampleVariant(labels, brisks, colors, start, end)
            X2, y2, meta2 = sampleNonVariant(labels, brisks, colors, start, end)
            
            np.savetxt(X_fh, [X, X2], delimiter=",", fmt="%f")
            np.savetxt(y_fh, [y, y2], delimiter=",", fmt="%d")
            np.savetxt(meta_fh, [meta, meta2], delimiter=",", fmt="%d")
            
            if (n+1)%10000==0:
                percentage_completion = 100.0*np.float(n+1)/no_of_pairs
                sys.stdout.write(str(n+1)+" of "+str(no_of_pairs)+" done ("+str(percentage_completion)+"%)\r")
                sys.stdout.flush()
    print ""

def load(n=0):
    with open("Cache/X"+str(n)+".csv") as X_fh, open("Cache/y"+str(n)+".csv") as y_fh, open("Cache/Xmeta"+str(n)+".csv") as meta_fh:
        X = pd.read_csv(X_fh, sep=",", dtype=np.float, header=None).values
        y = pd.read_csv(y_fh, header=None).values.astype(np.bool).flatten()
        meta = pd.read_csv(meta_fh, sep=",", dtype=np.int, header=None).values
        return X, y, meta


