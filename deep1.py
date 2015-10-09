# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:32:44 2015

@author: mdmiah
"""

#import matplotlib as mpl
#mpl.use('Agg') # Needed to work on server

import numpy as np
import pandas as pd
import random
import time
import sys
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import modelInputs
import modelMake
import dPredict

def loadDeepFeatures():
    # Load deep features
    with open("Data/deepFeatures1.npy") as fh:
        return np.load(fh)

def loadLabels():
    # Load group labels
    with open("Data/labels1.csv") as fh:
        labels = pd.read_csv(fh, sep=",", header=0, \
            true_values=['M'], false_values=['F']).values.astype(np.uint)
        return labels

def getRowForCombination(u, v, features, labels):
    feature1 = features[u]
    feature2 = features[v]
    label1 = labels[u]
    label2 = labels[v]
    
    # ratio of each of the features
    with np.errstate(divide='ignore', invalid='ignore'):
        fRatio = np.asfarray(feature1)/np.asfarray(feature2)
        fRatio[(feature2==0)|np.isnan(feature1)|np.isnan(feature2)] = 0
        fRatio[fRatio>1.0] = 1.0/fRatio[fRatio>1.0]

    Xrow = np.concatenate(( \
        #[distance.cosine(feature1, feature2)], \
        #[np.sum((feature1==feature2))], \
        #fRatio,
        feature1, feature2
        ))
    yrow = [label1==label2]
    metarow = [u, v]
    
#    # Fix any invalid values
#    Xrow[np.isnan(Xrow)] = 0
#    Xrow[np.isinf(Xrow)] = 0
    
    return Xrow, yrow, metarow

def sampleVariant(features, labels, start=0, end=None):
    if end is None:
        end = features.shape[0]-1
    
    # Choose an image at random
    u = random.randint(start, end)
    label1 = labels[u]
    
    # Match with a color variant
    for m in xrange(20): # Search nearby for a color variant
        v = random.randint(u-50, u+50)
        v = start if v<start else end if v>end else v
        if u==v: # Don't match with itself
            continue
        label2 = labels[v]
        if label1==label2:
            return getRowForCombination(u, v, features, labels)
    
    # If the randomly chosen image has no color variant,
    # sample again
    return sampleVariant(features, labels, start, end)

def sampleNonVariant(features, labels, start=0, end=None):
    if end is None:
        end = features.shape[0]-1
    
    # Choose an image at random
    u = random.randint(start, end)
    label1 = labels[u]
    
    # Match random non variants
    label2 = label1
    while label1==label2:
        v = random.randint(start, end)
        label2 = labels[v]

    return getRowForCombination(u, v, features, labels)

def saveTraining(no_of_pairs = 40000):
    features = loadDeepFeatures()
    labels = loadLabels()[:,1]
    
    test_start_n = no_of_pairs * modelInputs.train_fraction # Fraction of dataset used in training
    test_start_i = np.int((features.shape[0]-1) * modelInputs.train_fraction) # Fraction of images used in training
    
    with open("Cache/X101.csv", "w") as X_fh, open("Cache/y101.csv", "w") as y_fh, open("Cache/Xmeta101.csv", "w") as meta_fh:
        for n in xrange(no_of_pairs):
            if n<test_start_n:
                start = 0
                end = test_start_i-1
            else:
                start = test_start_i
                end = None
            
            X, y, meta = sampleVariant(features, labels, start, end)
            X2, y2, meta2 = sampleNonVariant(features, labels, start, end)
            
            np.savetxt(X_fh, [X, X2], delimiter=",", fmt="%f")
            np.savetxt(y_fh, [y, y2], delimiter=",", fmt="%d")
            np.savetxt(meta_fh, [meta, meta2], delimiter=",", fmt="%d")
            
            if (n+1)%1000==0:
                percentage_completion = 100.0*np.float(n+1)/no_of_pairs
                sys.stdout.write(str(n+1)+" of "+str(no_of_pairs)+" done ("+str(percentage_completion)+"%)\r")
                sys.stdout.flush()
    print ""

def loadInputs(n=101):
    with open("Cache/X"+str(n)+".csv") as X_fh, open("Cache/y"+str(n)+".csv") as y_fh, open("Cache/Xmeta"+str(n)+".csv") as meta_fh:
        X = pd.read_csv(X_fh, sep=",", dtype=np.float, header=None).values
        y = pd.read_csv(y_fh, header=None).values.astype(np.bool).flatten()
        meta = pd.read_csv(meta_fh, sep=",", dtype=np.int, header=None).values
        return X, y, meta

def load():
    X, y, meta = loadInputs()
    
    count = X.shape[0]
    train_size = np.int(count*modelInputs.train_fraction)
    
    # Scale appropriately
    scaler = StandardScaler()
    scaler.fit(X[:train_size]) # fit only on training data
    X = scaler.transform(X)
    
    # Save scaler
    joblib.dump(scaler, "Models/scaler101.pkl")
    
    Xtrain = X[:train_size]
    ytrain = y[:train_size]
    metatrain = meta[:train_size]
    Xtest = X[train_size:]
    ytest = y[train_size:]
    metatest = meta[train_size:]
    
    return Xtrain, Xtest, ytrain, ytest, metatrain, metatest

def run():
    tic = time.time()
    
    Xtrain, Xtest, ytrain, ytest, metatrain, metatest = load()
    print "Loaded dataset"
    clf = modelMake.makeModel(Xtrain, ytrain, 101, 'log')
    ztest, FP, FN = modelMake.test(clf, Xtest, ytest, metatest, 101)
    modelMake.testProba(clf, Xtest, ytest, metatest, 101, 101)
    
    m, s = divmod((time.time() - tic), 60)
    timeTakenString = "Time taken to run: "+str(int(m))+" minutes "+str(round(s,3))+" seconds"
    print timeTakenString
    with open("Results/Metrics101.txt", "a") as fh:
        fh.write("\n"+timeTakenString+"\n")
    
    dPredict.predict()


















