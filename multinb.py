#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 12:06:12 2019

@author: ariannasmith
"""

import models as m
import numpy as np
import glob
import math
from sklearn.metrics import classification_report



def trainMultiNB(trainData):
    print("Using", trainData)
    print("Training Multibomial Naive Bayes")
    bow = m.BoW(trainData)
    vocab = bow[0]
    matrix = bow[1]
    N = bow[1].shape[0]
    
    classes = [0, 1]
    condProb = np.zeros((len(classes), len(vocab)))
    prior = []
    

    #For each class, determine conditional probability of terms
    for x in classes:
        # Create submatrix for each class
        if x == 0:
            Nc = N - sum(matrix[:,-1])
            submatrix = matrix[:int(Nc), :-1]
            prior.append(Nc/N)
        else:
            Nc = sum(matrix[:, -1])
            submatrix = matrix[(N - int(Nc)):, :-1]
            prior.append(Nc/N)
        
        # Pre-allocate array for word frequencies
        colFreq = np.zeros(len(vocab))

        # Add frequency of each word for class
        for t in range(submatrix.shape[1]):
            columnSum = sum(submatrix[:, t])
            colFreq[t] = columnSum
       # Sum frequencies of all words for class
        totalFreq = sum(colFreq)

        
        for t in range(submatrix.shape[1]):
            num = sum(submatrix[:, t]) + 1
            condProb[x][t] = num/(totalFreq + N)
    return vocab, prior, condProb

def applyMultiNB(trainResults, testData):     
    print("Testing Multinomial Naive Bayes")
    vocab = trainResults[0]
    prior = trainResults[1]
    condProb = trainResults[2]
    hamPath = testData + "/ham"
    spamPath = testData + "/spam"
    hamspam = [hamPath, spamPath]
    
    allEmails = []
    # For each class
    for emailType in hamspam:    
        emails = sorted(glob.glob(emailType + "/*.txt"))
        allEmails.append(emails)
    allEmails = [x for y in allEmails for x in y]
    matchList = []
    yTrue = []
    yTest = []
    # For each document, classify as ham or spam based on training data
    for email in allEmails:
        typestr = email.split('/')
        if typestr[2] == "ham":
            trueClass = 0
            yTrue.append(0)
        else:
            trueClass = 1
            yTrue.append(1)
        score = [0, 0]
        logPrior = [0, 0]
        with open(email, 'rt', errors = 'ignore') as f:
            textStr = f.read()
            textStr = textStr.split()
            indexList = []
            
            # Removing stop words and common punctuation
            for word in textStr:
                remove = ['a', 'an', 'the', 'she', 'he', 'and', 'of', 'in', \
                           'on', 'with', 'do', 'did', 'are', 'is', 'or', 'at', \
                           '.', '?', ',', '-', '/']
                if word not in remove:
                    # Find index of words to calculate conditional probability for
                    if word in vocab and (vocab.index(word) not in indexList):
                        indexList.append(vocab.index(word))

            for emailType in hamspam:   
                logPrior[hamspam.index(emailType)] = math.log(prior[hamspam.index(emailType)])
                score[hamspam.index(emailType)] = logPrior[hamspam.index(emailType)]
         
                # Calculating score for each class
                for index in indexList:
                    score[hamspam.index(emailType)] += \
                        math.log(condProb[hamspam.index(emailType), index])
        
        # Finding MAP and comparing it to true class
        argmax = score.index(max(score))
        yTest.append(argmax)
        if trueClass == argmax:
            matchList.append(1)
        else:
            matchList.append(0)
    percentage = sum(matchList)/len(matchList)*100
    print(round(percentage, 2), "% accuracy", sep = '')
    print(classification_report(yTrue, yTest))

