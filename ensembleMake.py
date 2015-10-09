# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:55:48 2015

@author: mdmiah

"""

import matplotlib as mpl
mpl.use('Agg') # Needed to work on server

import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import modelInputs
import modelMake
import ensembleInputs
import predict

def saveEnsembleData(clf, Xold, y, meta, n=2, scaler_n=2):
    nrows = 16
    
    # Creating data set for ensemble model
    z_proba = clf.predict_proba(Xold)[:,1]
    X = z_proba.reshape(z_proba.size/nrows,nrows)
    y = y[::nrows]
    meta = meta[::nrows]
    
#    # Extra features
#    X = np.column_stack([X, #np.mean(X,axis=-1), np.max(X,axis=-1)-np.min(X,axis=-1), \
#        np.mean(X<0.6,axis=-1), np.mean(X>0.8,axis=-1), np.mean(X>0.9,axis=-1)
#        ])
    
    # Scale
    scaler = joblib.load("Models/scaler"+str(scaler_n)+".pkl")
    X = scaler.transform(X)
    
    with open("Cache/X"+str(n)+".csv", "w") as X_fh, open("Cache/y"+str(n)+".csv", "w") as y_fh, open("Cache/Xmeta"+str(n)+".csv", "w") as meta_fh:
        np.savetxt(X_fh, X, delimiter=",", fmt="%f")
        np.savetxt(y_fh, y, delimiter=",", fmt="%d")
        np.savetxt(meta_fh, meta, delimiter=",", fmt="%d")

def saveEnsembleTraining(clf, X1train, X1test, y1train, y1test, meta1train, meta1test, n=2):
    Xold = np.row_stack([X1train, X1test])
    y = np.concatenate([y1train, y1test])
    meta = np.row_stack([meta1train, meta1test])
    
    nrows = 16
    train_size = np.int(modelInputs.train_fraction*Xold.shape[0]/nrows)
    
    # Creating data set for ensemble model
    z_proba = clf.predict_proba(Xold)[:,1]
    X = z_proba.reshape(z_proba.size/nrows,nrows)
    
#    # Extra features
#    X = np.column_stack([X, #np.mean(X,axis=-1), np.max(X,axis=-1)-np.min(X,axis=-1), \
#        np.mean(X<0.6,axis=-1), np.mean(X>0.8,axis=-1), np.mean(X>0.9,axis=-1)
#        ])
    
    # Save scaler
    scaler = StandardScaler()
    scaler.fit(X[:train_size]) # fit only on training data
    joblib.dump(scaler, "Models/scaler"+str(n)+".pkl")
    
    saveEnsembleData(clf, Xold, y, meta, n, n)

def loadEnsembleTraining(n=2):
    X, y, meta = modelInputs.load(n)
    count = X.shape[0]
    train_size = np.int(count*modelInputs.train_fraction)
    
    Xtrain = X[:train_size]
    ytrain = y[:train_size]
    metatrain = meta[:train_size]
    Xtest = X[train_size:]
    ytest = y[train_size:]
    metatest = meta[train_size:]
    
    return Xtrain, Xtest, ytrain, ytest, metatrain, metatest

def run1(lazy=False):
    X1train, X1test, y1train, y1test, meta1train, meta1test = modelMake.load(1)
    if not lazy:
        clf = modelMake.getModel(0)
    else:
        clf = modelMake.makeModel(X1train, y1train, 1, 'log')
    z1test, FP1, FN1 = modelMake.test(clf, X1test, y1test, meta1test, 1)
    modelMake.testProba(clf, X1test, y1test, meta1test, 20, 1)
    
    # Second layer
    saveEnsembleTraining(clf, X1train, X1test, y1train, y1test, meta1train, meta1test, 2)

def run2():
    X2train, X2test, y2train, y2test, meta2train, meta2test = loadEnsembleTraining()
    clf2 = modelMake.makeModel(X2train, y2train, 2, 'log')
    z2test, FP2, FN2 = modelMake.test(clf2, X2test, y2test, meta2test, 2)
    modelMake.testProba(clf2, X2test, y2test, meta2test, 20, 2)

def run(count=1000, lazy=True):
    start = time.time()

    if not lazy:
        modelInputs.save(count)
    ensembleInputs.save(count)
    
    if not lazy:
        modelMake.run()
    run1(lazy)
    run2()
    
    m, s = divmod((time.time() - start), 60)
    timeTakenString = "Time taken to run: "+str(int(m))+" minutes "+str(round(s,3))+" seconds"
    print timeTakenString
    with open("Results/Metrics2.txt", "a") as fh:
        fh.write("\n"+timeTakenString+"\n")
    
    predict.predict()


