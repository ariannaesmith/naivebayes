#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 00:00:08 2019

@author: ariannasmith
"""

import numpy as np
import glob


def BoW(dataset):    
    hamPath = dataset + "/ham"
    spamPath = dataset + "/spam"
    hamspam = [hamPath, spamPath]
        
    vocab = []
    # Preallocate array space
    matrix = np.zeros((1000, 20000))
    fileCount = 0

    for emailType in hamspam:    
        emails = sorted(glob.glob(emailType + "/*.txt"))
        typestr = emailType.split('/')
        remove = ['a', 'an', 'the', 'she', 'he', 'and', 'of', 'in', \
                  'on', 'with', 'do', 'did', 'are', 'is', 'or', 'at', \
                  '.', '?', ',', '-', '/']

        for email in emails:
            frequency = {}
            filtered = []

            if typestr[2] == "ham":
                matrix[fileCount][0] = 0
            else:
                matrix[fileCount][0] = 1
            with open(email, 'rt', errors = 'ignore') as f:
                textStr = f.read()
                textStr = textStr.split()

                # Removing stop words and common punctuation
                for word in textStr:
                    if word not in remove:
                        if word not in filtered:
                            filtered.append(word)  
                        # Add to vocabulary
                        if word not in vocab:
                            vocab.append(word)
                        # Add to frequency list                                     
                        if word not in frequency:
                            frequency[word] = 1
                        # Update count of word
                        else:
                            count = frequency.get(word)
                            frequency[word]= count + 1
                for word in filtered:
                    matrix[fileCount][vocab.index(word) + 1] = frequency[word]
            fileCount += 1  

    # Remove unused space that was pre-allocated
    vocabSize = len(vocab)
    finalMatrix = matrix[:fileCount, :vocabSize + 1]
   
    # Rearrange so that class column is the last column    
    classColumn = finalMatrix[:, 0].T.reshape(finalMatrix.shape[0] , 1)
    featureColumns = finalMatrix[:, 1:]
    rearrange = np.concatenate((featureColumns, classColumn), 1)
     
    return vocab, rearrange

def Bernoulli(dataset):      
    hamPath = dataset + "/ham"
    spamPath = dataset + "/spam"
    hamspam = [hamPath, spamPath]
        
    vocab = []
    # Preallocate array space
    matrix = np.zeros((1000, 20000))
    fileCount = 0

    for emailType in hamspam:    
        emails = sorted(glob.glob(emailType + "/*.txt"))
        
        typestr = emailType.split('/')
        remove = ['a', 'an', 'the', 'she', 'he', 'and', 'of', 'in', \
                  'on', 'with', 'do', 'did', 'are', 'is', 'or', 'at', \
                  '.', '?', ',', '-', '/']

        for email in emails:
            appearance = {}
            filtered = []

            if typestr[2] == "ham":
                matrix[fileCount][0] = 0
            else:
                matrix[fileCount][0] = 1
            with open(email, 'rt', errors = 'ignore') as f:
                textStr = f.read()
                textStr = textStr.split()

                # Removing stop words and common punctuation
                for word in textStr:
                    if word not in remove:
                        if word not in filtered:
                            filtered.append(word)  
                        # Add to vocabulary
                        if word not in vocab:
                            vocab.append(word)
                        # Add to frequency list                                     
                        if word not in appearance:
                            appearance[word] = 1
                for word in filtered:
                    matrix[fileCount][vocab.index(word) + 1] = appearance[word]
            fileCount += 1          

    # Remove unused space that was pre-allocated
    vocabSize = len(vocab)
    finalMatrix = matrix[:fileCount, :vocabSize + 1]
   
    # Rearrange so that class column is the last column    
    classColumn = finalMatrix[:, 0].T.reshape(finalMatrix.shape[0] , 1)
    featureColumns = finalMatrix[:, 1:]
    rearrange = np.concatenate((featureColumns, classColumn), 1)
        
    return vocab, rearrange
