#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 02:09:49 2019

@author: ariannasmith
"""
import models as m
import numpy as np
import glob
import math
from sklearn.metrics import classification_report


def logRegTrain(traindata, model, testdata):
    # Create matrix based on model specified
    if model == "bow":
        bowModel = m.BoW(traindata)
        matrix = bowModel[1]
        vocab = bowModel[0]
        print("Running Logistic Regression on Bag of Words")
        
    else: 
        bern = m.Bernoulli(traindata)
        matrix = bern[1]
        vocab = bern[0] 
        print("Running Logistic Regression on Bernoulli")
    

    
    # Add initial column for x0, all values are 1
    x0 = np.ones((matrix.shape[0])).reshape(matrix.shape[0], 1)
    matrix = np.concatenate((x0, matrix), axis = 1)
    
    # Split training data into "train" and "validation"
    tvSplit = split(matrix)
    train = tvSplit[0]
    validation = tvSplit[1]
    
    eta = 0.005
    iterations = 50

    lambdaList = [1, 5, 10]
    weightsList = []
    
    # Learning weights for all lambdas
    for l in range(len(lambdaList)):
        print("Learning for", lambdaList[l])
        weights = np.zeros((len(vocab) + 1))
        
        for iteration in range(iterations):
            predErrorList = []
            
            for doc in range(train.shape[0] - 1):
                # Calculate prediction error
                p1 = math.exp(np.dot(train[doc, :-1], weights))
                p2 = p1 / (1 + p1)
                p = train[doc, -1] - p2
                # Append prediction error to list for use in next section
                predErrorList.append(p)
                
            for i in range(len(weights)):
                z = 0
                # Calculating part of equation dependent on doc[l]
                for doc in range(train.shape[0] - 1):
                    z += train[doc][i] * predErrorList[doc]
                    
                weights[i] = weights[i] + (eta * z) - (eta * lambdaList[l] * weights[i])               
        weightsList.append(weights)
        
        
    
    totalPredictions = []
    
    
    for y in range(len(weightsList)):    
        predictList = []
        for doc in range(validation.shape[0]):     
            prediction = np.dot(weightsList[y], validation[doc, :-1])
            
            if prediction > 0:
                prediction = 1                
            else:
                prediction = 0
                
            predictList.append(prediction)           
        totalPredictions.append(predictList)     
    
    trueList = validation[:, -1]
    
    totalMatchList = []
    accuracyList = []
    
    for listt in range(len(totalPredictions)):
        matchList = []
        for x in range(len(trueList)):
            
            if trueList[x] == totalPredictions[listt][x]:
                matchList.append(1)
                
            else:
                matchList.append(0)
        totalMatchList.append(matchList)    
         
    for a in range(len(totalMatchList)):
        accuracy = sum(totalMatchList[a]) / len(totalMatchList[a]) * 100
        accuracyList.append(accuracy)
    
    for a in range(len(accuracyList)):
        print("Lambda of ", lambdaList[a], " gives ", \
              round(accuracyList[a], 2),  "% accuracy", sep = '')

    maxAccuracy = max(accuracyList)
    bestLambda = lambdaList[accuracyList.index(maxAccuracy)]


    # Now we learn weights on full training set   
    weightsT = np.zeros((len(vocab) + 1))
    print("Learning on full training set")
    for iteration in range(iterations):
        predError = []
        for doc in range(train.shape[0] - 1):
            # Calculate prediction error
            p1 = math.exp(np.dot(train[doc, :-1], weightsT))
            p2 = p1 / (1 + p1)
            p = train[doc, -1] - p2
            # Append prediction error to list for use in next section
            predError.append(p)
        for i in range(len(weightsT)):
            z = 0
            for doc in range(train.shape[0] - 1):
                z += train[doc][i] * predError[doc]
            weightsT[i] = weightsT[i] + eta * z - eta * bestLambda * weightsT[i]
    
    return weightsT, bestLambda, vocab, testdata, model
    


def logRegApply(weights, lambdaT, vocab, test, model): 
    testMatrix = np.zeros((1000, 20000))
    weightsT = weights
    
    # Read in test data and create matrix
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
    matrix = testMatrix[:fileCount, :vocabSize + 1]

    # Rearrange so that class column is the last column    
    classColumn = matrix[:, 0].T.reshape(matrix.shape[0] , 1)
    featureColumns = matrix[:, 1:]
    rearrange = np.concatenate((featureColumns, classColumn), 1)
    
    # Add initial column for x0, all values are 1
    x0 = np.ones((rearrange.shape[0])).reshape(rearrange.shape[0], 1)
    rearrange = np.concatenate((x0, rearrange), axis = 1)
    
    
    
    # Now run weights on test data
    predictTList = [] 
    print("Running on test data")
    for doc in range(rearrange.shape[0]):
        predictionT = np.dot(weightsT, rearrange[doc, :-1])
        if predictionT > 0:
            predictionT = 1
        else:
            predictionT = 0
        predictTList.append(predictionT)
    
    trueTList = rearrange[:, -1]
    
    matchTList = []
    
    for x in range(len(trueTList)):
        if trueTList[x] == predictTList[x]:
            matchTList.append(1)
        else:
            matchTList.append(0)
    accuracyT = sum(matchTList) / len(matchTList) * 100
    
    
    print(round(accuracyT, 2), "% accuracy with lambda ", lambdaT, sep = '')
    print(classification_report(trueTList, predictTList))


   
    
def split(train):
    # Split into ham and spam matrices, and then take 70/30 split of each
    # to create training and validation datasets
    spamCount = int(sum(train[:, -1]))
    hamCount = int(train.shape[0] - spamCount)

    hamMatrix = train[:hamCount, :]
    spamMatrix = train[hamCount:, :]
    
    tHamCount = int(round(hamCount*0.7))
    
    tSpamCount = int(round(spamCount*0.7))
    
    trainMatrix = np.concatenate((hamMatrix[:tHamCount, :], spamMatrix[:tSpamCount]))
    validMatrix = np.concatenate((hamMatrix[tHamCount:, :], spamMatrix[tSpamCount:, :]))

    return trainMatrix, validMatrix
