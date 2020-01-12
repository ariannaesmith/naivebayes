#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:11:25 2019

@author: ariannasmith
"""

import models as m
import numpy as np
import glob
import warnings
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def sgd(train, test, model):
    testMatrix = np.zeros((1000, 20000))

    if model == "bow":
        # Creating x and y with training data
        bow = m.BoW(train)
        matrix = bow[1]
        trainX = matrix[:, :-1]
        trainY = matrix[:, -1]
        vocab = bow[0]
        
    else: 
        bern = m.Bernoulli(train)
        matrix = bern[1]
        trainX = matrix[:, :-1]
        trainY = matrix[:, -1]
        vocab = bern[0]  
        
    fileCount = 0

    # Create matrix for testing data
    hamPath = test + "/ham"
    spamPath = test + "/spam"
    hamspam = [hamPath, spamPath]
    
    # For each class
    for emailType in hamspam:    
        emails = sorted(glob.glob(emailType + "/*.txt"))
        
        remove = ['a', 'an', 'the', 'she', 'he', 'and', 'of', 'in', \
                  'on', 'with', 'do', 'did', 'are', 'is', 'or', 'at', \
                  '.', '?', ',', '-', '/']
        
        for email in emails:
            frequency = {}
            filtered = []
    
            if hamspam.index(emailType) == 0:
                testMatrix[fileCount][0] = 0
            
            else:
                testMatrix[fileCount][0] = 1
            
            with open(email, 'rt', errors = 'ignore') as f:
                textStr = f.read()
                textStr = textStr.split()
    
                # Removing stop words and common punctuation
                for word in textStr:
                    if word not in remove:
                        
                        # Ignore new words that were not in training data
                        if word not in filtered and word in vocab:
                            filtered.append(word)  
                        
                        # Add to frequency list                                     
                        if word not in frequency:
                            frequency[word] = 1
                        
                        # Update count of word if model is Bag of Words
                        # Otherwise it stays as Bernoulli model
                        elif model == "bow":
                            count = frequency.get(word)
                            frequency[word]= count + 1
                
                for word in filtered:
                    testMatrix[fileCount][vocab.index(word) + 1] = frequency[word]
            
            fileCount += 1  
    
    # Remove unused space that was pre-allocated
    vocabSize = len(vocab)
    finalMatrix = testMatrix[:fileCount, :vocabSize + 1]
   
    yTrue = finalMatrix[:, 0]
    xTest = finalMatrix[:, 1:]
    
    params = {
            "loss" : ["hinge", "log", "modified_huber", "squared_loss"],
            "penalty" : ["none", "l2"],
            "alpha" : [0.0001, 0.001],
            "max_iter" : [500, 1000],
            "n_iter_no_change" : [5, 15] 
    }
    
    thisModel = SGDClassifier(shuffle = True)
    clf = GridSearchCV(thisModel, param_grid = params)

    clf.fit(trainX, trainY)
    print(clf.best_score_)
    print(clf.best_estimator_)
    yTest = clf.predict(xTest)
    print(classification_report(yTrue, yTest))
