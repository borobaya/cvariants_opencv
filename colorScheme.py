# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:16:11 2015

@author: mdmiah

Reference:
http://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/

"""

#
# Use distance threshold of 0.1 to determine whether images are similar enough
#

import numpy as np
import pandas as pd
from scipy.spatial import distance
import cv2
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import gc; gc.enable()
import time

n = 1

pathIn1 = "Data/images/"
pathIn2 = "Data/images2/"
n_clusters = 6

# ----------------------------------        ----------------------------------

def getImage(i, n=n):
    path = (pathIn1 if n==1 else pathIn2) +str(i)+".jpg"
    img = cv2.imread(path)
    if img is None or img.size==0:
        return None
#    img = histStretch(img)
    img = cv2.resize(img, (200, 200))
    return img

def histStretch(image):
    # Modified histogram stretching
    # Put 5th and 95th percentile as min and max
    percentiles = np.percentile(image, [5,95])
    image[image<percentiles[0]] = percentiles[0]
    image[image>percentiles[1]] = percentiles[1]
    image = cv2.normalize(image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return image

def showImage(i, n=n):
    img = getImage(i, n)
    cv2.imshow('Image', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

def showHists(i, n=n, n_clusters=n_clusters):
    showImage(i, n)
#    colorHistogram(i, n)
#    colorHistogram2D(i, n)
#    colorHistogram3D(i, n)
    colorCentroids(i, n, n_clusters)

# ----------------------------------        ----------------------------------

def colorHistogram(i, n=n):
    img = getImage(i, n)
    
    chans = cv2.split(img)
    colors = ("b", "g", "r")
    plt.figure(1)
    plt.clf()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    features = []
    
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [250], [0, 250])
        features.extend(hist)
        
        plt.plot(hist, color = color)
        plt.xlim([0, 256])

def colorHistogram2D(i, n=n):
    img = getImage(i, n)
    chans = cv2.split(img)
    
    fig = plt.figure(2, figsize=(7.5, 2), dpi=100)
    fig.set_tight_layout(True)
    plt.clf()
 
    # plot a 2D color histogram for green and blue
    ax = fig.add_subplot(131)
    hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None,
                        [124, 124], [0, 250, 0, 250])
    p = ax.imshow(hist, interpolation = "nearest")
    ax.set_title("2D Color Histogram for Green and Blue", fontsize=8)
    cbar = plt.colorbar(p)
    cbar.ax.tick_params(labelsize=6)
    plt.tick_params(labelsize=6)
     
    # plot a 2D color histogram for green and red
    ax = fig.add_subplot(132)
    hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None,
                        [124, 124], [0, 250, 0, 250])
    p = ax.imshow(hist, interpolation = "nearest")
    ax.set_title("2D Color Histogram for Green and Red", fontsize=8)
    cbar = plt.colorbar(p)
    cbar.ax.tick_params(labelsize=6)
    plt.tick_params(labelsize=6)
     
    # plot a 2D color histogram for blue and red
    ax = fig.add_subplot(133)
    hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None,
                        [124, 124], [0, 250, 0, 250])
    p = ax.imshow(hist, interpolation = "nearest")
    ax.set_title("2D Color Histogram for Blue and Red", fontsize=8)
    cbar = plt.colorbar(p)
    cbar.ax.tick_params(labelsize=6)
    plt.tick_params(labelsize=6)
     
    # The dimensionality of the 2D histograms
    print "2D histogram shape: %s, with %d values" % (
        hist.shape, hist.flatten().shape[0])

def colorHistogram3D(i, n=n):
    img = getImage(i, n)
    hist = cv2.calcHist([img], [0, 1, 2], None,
                        [25, 25, 25], [0, 250, 0, 250, 0, 250])
    print "3D histogram shape: %s, with %d values" % (
        hist.shape, hist.flatten().shape[0])

# ----------------------------------        ----------------------------------

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist

def plot_colors(hist, colors, width = 500, height = 50):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((height, width, 3), dtype = "uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for percent, color in zip(hist, colors):
        # plot the relative percentage of each color
        endX = startX + (percent * width)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), height),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

def descendOrderColors(hist, colors):
    # Order, from largest to smallest
    order = np.argsort(hist)
    order = order[::-1] # Reverse values in array

    colors2 = np.array([colors[i] for i in order])
    hist2 = np.array([hist[i] for i in order])
    
    return hist2, colors2

# http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array
def unique_rows(a):
    b = np.ascontiguousarray(a)
    unique_b = np.unique(b.view([('', b.dtype)]*b.shape[1]))
    return unique_b.view(b.dtype).reshape((unique_b.shape[0], b.shape[1]))

def mergeCloseColors(colors, tolerance=32):
    colors2 = colors.copy()
    tolerance **=2 # Square of distance is used, so square the tolerance value too
    
    distMatrix = [[ distance.sqeuclidean(colors[i],colors[j]) if i>j else 195075 \
        for j in xrange(len(colors))] for i in xrange(len(colors))]
    distMatrix = np.array(distMatrix)
    
    for i in xrange(len(colors)):
        for j in xrange(len(colors)):
            if i<j:
                continue
            
            if distMatrix[i, j] < tolerance:
#                print str(i), "<-", str(j), "\t", colors[i].astype(np.int), \
#                    "becoming", colors[j].astype(np.int)
                colors2[i] = colors[j]
                break

    #print "Unique Colors:\n", unique_rows(colors.astype(np.int))
    return colors2

def removeDuplicateColors(hist, colors):
    hist = hist.copy()
    colors = colors.copy()
    
    order = np.lexsort(colors.T)
    colors = colors[order]
    hist = hist[order] # Make sure to match order of colors
    diff = np.diff(colors, axis=0)
    ui = np.ones(len(colors), 'bool')
    ui[1:] = (diff != 0).any(axis=1) # Binary array of which indexes are unique
    
    for i in xrange(len(colors)):
        if not ui[i]:
            # Find index of color that this is a duplicate of
            for j in xrange(len(colors)):
                if ui[j] and (colors[i]==colors[j]).all():
                    # Transfer value to the hist of the original color
                    hist[j] += hist[i]
                    break
    
    unique_colors = colors[ui]
    unique_hist = hist[ui]
    return unique_hist, unique_colors

def getColorBarSimple(pixels, n_clusters=n_clusters):
#    clock = time.time()
    
    # Find centroids
    clt = KMeans(n_clusters=n_clusters)
    clt.fit(pixels)
    
#    print "K-Means: ", round(time.time()-clock,3), "seconds"
    
    hist = centroid_histogram(clt)
    colors = clt.cluster_centers_
    hist, colors = descendOrderColors(hist, colors)
    bar = plot_colors(hist, colors)
    return hist, colors, bar

def getColorBarMerged(hist, colors):
    hist = hist.copy()
    colors = colors.copy()
    
    # Attempt to merge similar colors together
    colors = mergeCloseColors(colors, tolerance=32)
    hist, colors = removeDuplicateColors(hist, colors)
    hist, colors = descendOrderColors(hist, colors)
    
    # Second Pass
    if True:
        colors = mergeCloseColors(colors, tolerance=48)
        hist, colors = removeDuplicateColors(hist, colors)
        hist, colors = descendOrderColors(hist, colors)
    
    bar = plot_colors(hist, colors)
    return hist, colors, bar

def showColorBar(bar, figurenum=3):
    # Show color bar
    fig = plt.figure(figurenum, figsize=(5.5,1), dpi=100)
    plt.clf()
    plt.axis("off")
    plt.imshow(bar)
    fig.set_tight_layout({'pad':0.0, 'h_pad':0.0, 'w_pad':0.0})
    plt.show()

def recolorImage(img, colors):
    pixels = img.reshape((img.shape[0]*img.shape[1], 3))
    
    clt = KMeans(n_clusters=len(colors))
    clt.cluster_centers_ = colors
    
    clock = time.time()
    
    labels = clt.fit_predict(pixels)
    
    print "K-Means fitting labels: ", round(time.time()-clock,3), "seconds"
    clock = time.time()
    
    quant = clt.cluster_centers_.astype(np.uint8)[labels]
    quant = quant.reshape(img.shape)
    quant = cv2.cvtColor(quant, cv2.COLOR_RGB2BGR)
    return quant

def colorCentroids(i, n=n, n_clusters=n_clusters):
    clock = time.time()
    
    # Get list of pixels colors
    img = getImage(i, n)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img, (100, 100))
    pixels = img2.reshape((img2.shape[0]*img2.shape[1], 3))
    
    # Show original color bar
    hist, colors, bar = getColorBarSimple(pixels, n_clusters)
    showColorBar(bar, figurenum=3)
    
    # Show color bar with similar colors merged
    hist2, colors2, bar2 = getColorBarMerged(hist, colors)
    showColorBar(bar2, figurenum=4)
    
    # Recolor image with new centroids
    print len(colors2), "unique colors"
    img2 = recolorImage(img, colors2)
    cv2.imshow('Image2', img2)
    
    print "Overall: ", round(time.time()-clock,3), "seconds"

def getColorHist(i, n=n, n_clusters=n_clusters, merge=True):
    # Get list of pixels colors
    img = getImage(i, n)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape((img.shape[0]*img.shape[1], 3))
    
    # Original color bar
    hist, colors, bar = getColorBarSimple(pixels, n_clusters)
    if merge:
        # Color bar with similar colors merged
        hist, colors, bar = getColorBarMerged(hist, colors)
    
    # Make all hists constant width
    hist = np.pad(hist, (0,n_clusters-len(hist)), 'constant', constant_values=(0))
    return hist, colors

def getColorHists(maxI, minI=0, n=n, n_clusters=n_clusters, merge=True):
#    clock = time.time()
    hists = [getColorHist(i,n,n_clusters,merge)[0] for i in xrange(minI, maxI+1)]
    hists = np.array(hists)
    
#    print "Overall: ", round(time.time()-clock,3), "seconds"
    return hists

def compareColorHists(i, j, n=n, n_clusters=n_clusters, merge=True):
    hist1, colors1 = getColorHist(i, n, n_clusters, merge)
    hist2, colors2 = getColorHist(j, n, n_clusters, merge)
    dist = distance.euclidean(hist1,hist2)
    areColorsDifferent = compareColors(colors1, colors2)
    return dist, areColorsDifferent

def compareAllColorHists(maxI, minI=0, n=n, n_clusters=n_clusters, merge=True):
    hists = getColorHists(maxI, minI, n, n_clusters, merge)
    distMatrix = [[ distance.euclidean(hists[i],hists[j]) if (j-i>0) else 1 \
        for i in xrange(len(hists))] for j in xrange(len(hists))]
    distMatrix = np.array(distMatrix)
    return distMatrix

def compareColors(colors1, colors2):
    count = len(colors1) if len(colors1)<len(colors2) else len(colors2)
    distances = [distance.euclidean(colors1[i], colors2[i]) for i in xrange(count)]
    avgDist = np.mean(distances)
    return avgDist>40

# ----------------------------------        ----------------------------------


#
# Improvements:
#  - Use color location information too
#  - Look at LSV values instead
#



# ----------------------------------        ----------------------------------






