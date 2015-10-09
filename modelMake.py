# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:55:48 2015

@author: mdmiah

"""

import matplotlib as mpl
mpl.use('Agg') # Needed to work on server

import numpy as np
import pandas as pd
import random
import cv2
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import modelInputs

def load(n=0):
    print "Loading data..."
    X, y, meta = modelInputs.load(n)
    count = X.shape[0]
    train_size = np.int(count*modelInputs.train_fraction)
    
    # Scale appropriately
    scaler = StandardScaler()
    scaler.fit(X[:train_size]) # fit only on training data
    X = scaler.transform(X)
    
    # Save scaler
    joblib.dump(scaler, "Models/scaler"+str(n)+".pkl")
    
    Xtrain = X[:train_size]
    ytrain = y[:train_size]
    metatrain = meta[:train_size]
    Xtest = X[train_size:]
    ytest = y[train_size:]
    metatest = meta[train_size:]
    
    return Xtrain, Xtest, ytrain, ytest, metatrain, metatest

def makeModel(X, y, n=0, choice='log'):
    clf = None
    
    log_params = {
         "loss": ["log"],
         "penalty": ["l1", "l2"], # "elasticnet"
         "alpha": 10.0**-np.arange(1,7),
         "epsilon": 10.0**np.arange(2,7),
         "n_iter": [5],
         "shuffle": [True]
        }
    huber_params = {
         "loss": ["modified_huber"],
         "penalty": ["l1", "l2", "elasticnet"],
         "alpha": 10.0**-np.arange(1,7),
         "epsilon": 10.0**np.arange(0,7),
         "n_iter": [5],
         "shuffle": [True]
        }
    SVM_params = {
        'C': 10.0**-np.arange(0,7),
        'gamma': 10.0**-np.arange(0,7),
        'kernel': ['linear'], # 'poly', 'rbf'
        'probability': [True]
        }
    RF_params = {
        'n_estimators' : [1000],
        'bootstrap' : [False]
        }
    NN_params = {}
    
    if choice=='log':
        params = log_params
        model = SGDClassifier()
    elif choice=='huber':
        params = huber_params
        model = SGDClassifier()
    elif choice=='svm':
        params = SVM_params
        model = svm.SVC(C=1)
    elif choice=='rf':
        params = RF_params
        model = RandomForestClassifier(n_estimators=1000, bootstrap=False)
        #clf = RandomForestClassifier(n_estimators=1000, bootstrap=False)
    
    # Set up Grid Search
    print "Grid search..."
    clf = GridSearchCV(model, params, n_jobs=2, scoring='f1')
    clf.fit(X, y)
    clf = clf.best_estimator_
    print clf
    
    # Mini-batch learning with Replacement
    
    # TODO
    #
    # ...
    # clf.partial_fit(X[i:j], y[i:j])
    #
    
    # Save classifier model
    joblib.dump(clf, "Models/model"+str(n)+".pkl")
    return clf

def getModel(n=0):
    clf = joblib.load("Models/model"+str(n)+".pkl") 
    return clf

def test(clf, Xtest, ytest, metatest, n=0):
    print "Testing..."
    ztest = clf.predict(Xtest)
    
    #with open("Results/Z"+str(n)+".csv", "w") as fh:
    #    np.savetxt(fh, [ztest], fmt="%d\n")
    
    metrics =  "Cross-tabulation of test results:\n"
    metrics +=  pd.crosstab(ytest, ztest, rownames=['actual'], colnames=['preds']).to_string()
    metrics += "\n\n"
    
    metrics +=  "Classification Report:\n"
    metrics +=  classification_report(ytest, ztest)
    metrics += "\n"
    
    # Save metrics
    print metrics
    with open("Results/Metrics"+str(n)+".txt", "w") as fh:
        fh.write(metrics)

    # Look at False Positives and Negatives
    FP = metatest[(ytest==False) & (ztest==True)]
    FN = metatest[(ytest==True) & (ztest==False)]

    return ztest, FP, FN

def testProba(clf, Xtest, ytest, metatest, no_of_bins = 10, n=0):
    ztest_proba = clf.predict_proba(Xtest)[:,1]
    
    # Calculate fraction_of_matches vs. probability
    binwidth = 1.0/no_of_bins
    x = (np.array(range(no_of_bins))+0.5)*binwidth
    y = []
    
    for i in xrange(no_of_bins):
        l = i * binwidth
        u = (i+1) * binwidth
        indexes = (ztest_proba>=l) & (ztest_proba<u)
        matches = np.sum(indexes & ytest, dtype=np.float)
        total = np.sum(indexes, dtype=np.float)
        fraction = 0 if total==0 else matches / total
        y.append(fraction)
    
    y = np.array(y)
    
    # Save x, y to file
    #fractions = np.column_stack([x,y])
    #with open("Results/Fractions"+str(n)+".csv", "w") as fh:
    #    np.savetxt(fh, fractions, delimiter=",", fmt="%f")
    
    # Draw graph
    fig = plt.figure(n)
    plt.clf()
    plt.plot(x, y)
    plt.title("Fraction of Matches v. Probability", fontsize=10)
    plt.xlabel("Prediction Probability", fontsize=8)
    plt.ylabel("Fraction of Correct Matches", fontsize=8)
    plt.tick_params(labelsize=6)
    fig.set_tight_layout(True)
    fig.savefig("Results/Fractions"+str(n)+".png")
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(ytest, ztest_proba)
    roc_auc = auc(fpr, tpr)
    with open("Results/Metrics"+str(n)+".txt", "a") as fh:
        fh.write("\nArea under the ROC curve : %f\n" % roc_auc)
    
    # Plot ROC curve
    fig = plt.figure(n)
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    fig.savefig("Results/ROC"+str(n)+".png")
    
    # Plot Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(ytest, ztest_proba)
    area = auc(recall, precision)
    with open("Results/Metrics"+str(n)+".txt", "a") as fh:
        fh.write("Area under the Precision-Recall curve: %0.2f\n" % area)
    
    fig = plt.figure(n)
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC=%0.2f' % area)
    plt.legend(loc="lower left")
    fig.savefig("Results/PrecisionRecall"+str(n)+".png")

def run():
    Xtrain, Xtest, ytrain, ytest, metatrain, metatest = load(0)
    clf = makeModel(Xtrain, ytrain, 0, 'log')
    ztest, FP, FN = test(clf, Xtest, ytest, metatest, 0)
    testProba(clf, Xtest, ytest, metatest, 20, 0)

# ----------------------------------        ----------------------------------

def tiledImages(i, j):
    img1 = modelInputs.getImage(i)
    img2 = modelInputs.getImage(j)
    
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    if (len(img1.shape)==2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if (len(img2.shape)==2):
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1,:cols1,:] = img1 # Place the first image to the left
    out[:rows2,cols1:cols1+cols2,:] = img2 # Place the next image to the right of it
    
    return out

def showTiledImages(i, j):
    img = tiledImages(i, j)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def viewMistakes(FP, FN):
    print "False Positives (incorrectly identified as matches)..."
    for i in random.sample(range(len(FP)), 10):
        row = FP[i]
        u = row[0]
        v = row[1]
        print row
        showTiledImages(u, v)
    
    print "False Negatives (matches not identified)..."
    for i in random.sample(range(len(FN)), 10):
        row = FN[i]
        u = row[0]
        v = row[1]
        print row
        showTiledImages(u, v)

# ----------------------------------        ----------------------------------
