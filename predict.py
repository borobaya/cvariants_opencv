# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:57:19 2015

@author: mdmiah
"""

import numpy as np
import random
import cv2
import sys
import modelInputs
import modelMake
import ensembleInputs
import ensembleMake

def predictionsFor(u, labels, brisks, colors):
    global probs
    count = colors.shape[0]
    train_size = np.int(count*modelInputs.train_fraction)

    # Dataset for model 1
    with open("Cache/X11.csv", "w") as X_fh, open("Cache/y11.csv", "w") as y_fh, open("Cache/Xmeta11.csv", "w") as meta_fh:
        for v in xrange(train_size, count):
            # Make sure minimum number of BRISK features exist
            k = v*4 + np.arange(4)
            brisk = brisks[k,1:]
            if np.sum( np.sum(brisk,axis=1)<modelInputs.minBriskFeatures ):
                continue
            
            X, y, meta = ensembleInputs.getRowsForImages(u, v, labels, brisks, colors)
            
            np.savetxt(X_fh, X, delimiter=",", fmt="%f")
            np.savetxt(y_fh, y, delimiter=",", fmt="%d")
            np.savetxt(meta_fh, meta, delimiter=",", fmt="%d")

    # Dataset for ensemble model
    X11, y11, meta11 = modelInputs.load(11)
    clf1 = modelMake.getModel(1) # Either 0 or 1
    ensembleMake.saveEnsembleData(clf1, X11, y11, meta11, 12)

    # Predict using ensemble model
    X12, y12, meta12 = modelInputs.load(12)
    clf2 = modelMake.getModel(2)
    z = clf2.predict_proba(X12)[:,1]
    
    # Sort results, displaying top 10 images for target
    probIndexes = np.argsort(z)[::-1]
    probs = np.column_stack([meta12[probIndexes,1], z[probIndexes]])
    probs = probs[:20]
    
    # Probabilities of color variants
    print "Probabilities of color variants:"
    print z[y12]
    
    # Create montage image of top matches
    indexes = np.concatenate([np.array([u]), probs[:15,0]])
    tiles = tiledImages(indexes)
    cv2.imwrite("Results/tiles/"+str(u)+".jpg", tiles) # Save tiles

    return probs

def predictForRandom(n=10):
    labels, brisks, colors = modelInputs.loadHists()
    count = colors.shape[0]
    train_size = np.int(count*modelInputs.train_fraction)
    
    i = 0
    while i<n:
        u = random.randint(train_size, count)
        # Make sure minimum number of BRISK features exist
        k = u*4 + np.arange(4)
        brisk = brisks[k,1:]
        if np.sum( np.sum(brisk,axis=1)<modelInputs.minBriskFeatures ):
            continue
        print "Making predictions for image", u
        predictionsFor(u, labels, brisks, colors)
        i += 1

def topPredictions(n=10, mx=3000):
    labels, brisks, colors = modelInputs.loadHists()
    count = colors.shape[0]
    train_size = np.int(count*modelInputs.train_fraction)
    test_size = count-train_size
    
    test_size = mx if test_size>mx else test_size
    count = train_size+test_size
    
    # Dataset for model 1 -- Make sure it doesn't create too big a file
    print "Creating datasets..."
    totCount = (test_size**2)/2 - (test_size/2)
    with open("Cache/X16.csv", "w") as X_fh, open("Cache/y16.csv", "w") as y_fh, open("Cache/Xmeta16.csv", "w") as meta_fh:
        i = 0
        chunk_m = 1000000
        m = -1
        for u in xrange(train_size, count):
            # Make sure minimum number of BRISK features exist
            k = u*4 + np.arange(4)
            brisk = brisks[k,1:]
            if np.sum( np.sum(brisk,axis=1)<modelInputs.minBriskFeatures ):
                continue
            
            for v in xrange(train_size, count):
                if v-u<=0:
                    continue
                m += 1 # Hack to reduce file size
                if m<chunk_m*0:
                    i += 1
                    continue
                if m>chunk_m*1:
                    break
                # Make sure minimum number of BRISK features exist
                k = v*4 + np.arange(4)
                brisk = brisks[k,1:]
                if np.sum( np.sum(brisk,axis=1)<modelInputs.minBriskFeatures ):
                    continue
                
                X, y, meta = ensembleInputs.getRowsForImages(u, v, labels, brisks, colors)
                
                np.savetxt(X_fh, X, delimiter=",", fmt="%f")
                np.savetxt(y_fh, y, delimiter=",", fmt="%d")
                np.savetxt(meta_fh, meta, delimiter=",", fmt="%d")
                
                if (i+1)%10000==0:
                    percentage_completion = 100.0*np.float(i+1)/totCount
                    sys.stdout.write(str(i+1)+" of "+str(totCount)+" done ("+str(percentage_completion)+"%)\r")
                    sys.stdout.flush()
                i += 1
    print "\nDone creating dataset"

    # Dataset for ensemble model
    X16, y16, meta16 = modelInputs.load(16)
    clf1 = modelMake.getModel(1) # Either 0 or 1
    ensembleMake.saveEnsembleData(clf1, X16, y16, meta16, 17)

    # Predict using ensemble model
    X17, y17, meta17 = modelInputs.load(17)
#    clf2 = modelMake.getModel(2)
#    z = clf2.predict_proba(X17)[:,1]
    z = np.min(X17, axis=-1) # Using minimum probability instead of ensemble model

    # Sort results
    probIndexes = np.argsort(z)[::-1]
    probs = np.column_stack([meta17[probIndexes, 0:2], z[probIndexes]])
    # Save top results to file
    #np.save("Results/TopMatches", probs)
    
    # Remove repetitions of same image
    probs = probs[np.sort(np.unique(probs[:,1], return_index=True)[1]),:]
    probs = probs[np.sort(np.unique(probs[:,0], return_index=True)[1]),:]

    # Display top image pairs
    for i in xrange(n):
        name = str(i)+"_"+str(int(np.round(probs[i, 2]*10**14)))
        print probs[i,2], name
        tiled = tiledImages(probs[i, 0:2].flatten(), 1, 2)
        cv2.imwrite("Results/matches/prob"+name+".jpg", tiled) # Save to file

    return probs[:20]

def predict(n=10):
    predictForRandom(n)
    topPredictions(n)
    print "done"

# ----------------------------------        ----------------------------------

def tiledImages(group, r=4, c=4):
    group = group.astype(np.int)
    
    count = r*c
    h = w = 200
    width = w * c
    height = h * r

    out = np.zeros((height,width,3), dtype='uint8')
    
    for i in xrange(count):
        img = modelInputs.getImage(group[i])
        img = cv2.resize(img, (w, h))
        row = np.int(i/c)
        col = i%c
        out[h*row:h*(row+1), w*col:w*(col+1)] = img
    
    return out

def showImage(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------------------        ----------------------------------





