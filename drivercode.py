#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:37:56 2019

@author: ariannasmith
"""

import sys
import multinb as multi
import discretenb as discrete
import sgdclassifier as sgdc
import logreg as lg

# Inputs as ./drivercode.py dataset algorithm model

if sys.argv[1] == "first":
    train = "first/train"
    test = "first/test"
elif sys.argv[1] == "enron1":
    train = "enron1/train"
    test = "enron1/test"
elif sys.argv[1] == "enron4":
    train = "enron4/train"
    test = "enron4/test"
    
    
    
if sys.argv[2] == "multinb":
    print("Running multinomial Naive Bayes")
    trainResults = multi.trainMultiNB(train)
    multi.applyMultiNB(trainResults, test)
    
elif sys.argv[2] == "discretenb":
    print("Running discrete Naive Bayes")
    trainResults = discrete.trainDiscreteNB(train)
    discrete.applyDiscreteNB(trainResults, test)
    
elif sys.argv[2] == "sgd":
    if sys.argv[3] == "bagofwords":
        print("Running SGD on Bag of Words")
        model = "bow"
    else:
        print("Running SGD on Bernoulli")
        model = "bern"
    sgdc.sgd(train, test, model)
    
elif sys.argv[2] == "logreg":
    if sys.argv[3] == "bagofwords":
        model = "bow"
    else:
        model = "bern"
    abc = lg.logRegTrain(train, model, test)
    lg.logRegApply(abc[0], abc[1], abc[2], abc[3], abc[4])