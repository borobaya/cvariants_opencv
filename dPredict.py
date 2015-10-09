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
import deep1

def predictionsFor(u, features, labels):
    global probs
    count = 4000# features.shape[0]
    train_size = np.int(count*modelInputs.train_fraction)

    # Dataset for model 1
    with open("Cache/X105.csv", "w") as X_fh, open("Cache/y105.csv", "w") as y_fh, open("Cache/Xmeta105.csv", "w") as meta_fh:
        for v in xrange(train_size, count):
            if u==v:
                continue
        
            X, y, meta = deep1.getRowForCombination(u, v, features, labels)
            
            np.savetxt(X_fh, [X], delimiter=",", fmt="%f")
            np.savetxt(y_fh, [y], delimiter=",", fmt="%d")
            np.savetxt(meta_fh, [meta], delimiter=",", fmt="%d")

    # Dataset for ensemble model
    X105, y105, meta105 = deep1.loadInputs(105)
    clf101 = modelMake.getModel(101)
    z = clf101.predict_proba(X105)[:,1]
    
    # Sort results, displaying top 10 images for target
    probIndexes = np.argsort(z)[::-1]
    probs = np.column_stack([meta105[probIndexes,1], z[probIndexes]])
    probs = probs[:20]
    
    # Probabilities of color variants
    print "Probabilities of color variants:"
    print z[y105]
    
    # Create montage image of top matches
    indexes = np.concatenate([np.array([u]), probs[:15,0]])
    tiles = tiledImages(indexes)
    cv2.imwrite("Results/tiles/"+str(u)+".jpg", tiles) # Save tiles

    return probs

def predictForRandom(n=10):
    features = deep1.loadDeepFeatures()
    labels = deep1.loadLabels()[:,1]
    
    count = 4000 #features.shape[0]
    train_size = np.int(count*modelInputs.train_fraction)
    
    for i in xrange(n):
        u = random.randint(train_size, count)
        print "Making predictions for image", u
        predictionsFor(u, features, labels)

def topPredictions(n=10, mx=2000):
    features = deep1.loadDeepFeatures()
    labels = deep1.loadLabels()[:,1]
    
    count = 4000 # features.shape[0]
    train_size = np.int(count*modelInputs.train_fraction)
    test_size = count-train_size
    
    test_size = mx if test_size>mx else test_size
    count = train_size+test_size
    
    # Dataset for model 1 -- Make sure it doesn't create too big a file
    print "Creating datasets..."
    totCount = (test_size**2)/2 - (test_size/2)
    with open("Cache/X106.csv", "w") as X_fh, open("Cache/y106.csv", "w") as y_fh, open("Cache/Xmeta106.csv", "w") as meta_fh:
        i = 0
        chunk_m = 1000000
        m = -1
        for u in xrange(train_size, count):
            for v in xrange(train_size, count):
                if v-u<=0:
                    continue
                m += 1 # Hack to reduce file size
                if m<chunk_m*0:
                    i += 1
                    continue
                if m>chunk_m*1:
                    break
                
                X, y, meta = deep1.getRowForCombination(u, v, features, labels)
                
                np.savetxt(X_fh, [X], delimiter=",", fmt="%f")
                np.savetxt(y_fh, [y], delimiter=",", fmt="%d")
                np.savetxt(meta_fh, [meta], delimiter=",", fmt="%d")
                
                if (i+1)%10000==0:
                    percentage_completion = 100.0*np.float(i+1)/totCount
                    sys.stdout.write(str(i+1)+" of "+str(totCount)+" done ("+str(percentage_completion)+"%)\r")
                    sys.stdout.flush()
                i += 1
    print "\nDone creating dataset"

    # Dataset for ensemble model
    X106, y106, meta106 = modelInputs.load(106)
    clf101 = modelMake.getModel(101)
    z = clf101.predict_proba(X106)[:,1]

    # Sort results
    probIndexes = np.argsort(z)[::-1]
    probs = np.column_stack([meta106[probIndexes, 0:2], z[probIndexes]])
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





