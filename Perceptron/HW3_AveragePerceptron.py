# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:38:08 2024

@author: daisy
"""

# Import Libraries
import numpy as np
import pandas as pd

# Import bank note data
testData = r"C:\Users\daisy\Documents\GitHub\Machine-Learning-DaisyQuach\Perceptron\bank-note\test.csv"
dfTest = pd.read_csv(testData, names=['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Label'], header=None,index_col=False)

trainingData = r"C:\Users\daisy\Documents\GitHub\Machine-Learning-DaisyQuach\Perceptron\bank-note\train.csv"
dfTrain = pd.read_csv(trainingData, names=['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Label'], header=None,index_col=False)


# Other user-functions
def addBias(df):                         # bias b = ones column vector in beginning of array
    return np.column_stack((np.ones(len(df)), df))


# Average Perceptron algorithm
def averagePerceptron(df, r, T):
    
    # Initialize weight vectors to be zero vector
    dfShape = df.shape
    rows = dfShape[0]
    cols = dfShape[1] - 1             # Exclude the label column
    wLearned = np.zeros(cols)         # Weight vector matches # of columns/attributes
    a = np.zeros(cols)                # Average weight vector matches # of columns/attributes
     
    # Initialize storage saving vectors
    aIter = []
    aIter.append(a)

    # Setup inputs for algorithm and change y_i-->{0,1}-->{-1,1}:
    X = df.iloc[:,:-1].to_numpy()
    y = df.iloc[:,len(df.columns)-1].to_numpy()
    for i in range(0,rows):
        if y[i] == 0:
            y[i] = -1
     
        
    for e in range(1, T):     # Run for set amount of epochs
                
        # For each training example:
        for i in range(0,rows):
            check4Err = y[i] * sum(wLearned.transpose() * X[i])
            
            if check4Err <= 0:                                 # If there is an error
                wLearned = wLearned + r*y[i]*X[i]              # Update weight vector
                a = a + wLearned                               # Update average weight vector
                
        # Values to store per epoch, e:
        aIter.append(a)
    
    return a


# Initial parameters
T = 10

# Test for different learning rates 
r = 1
# r = 0.5
# r = 0.25
# r = 0.125
# r = 0.0625
# r = 0.03125
# r = 0.015625
# r = 0.0078125
# r = 0.00390625
# r = 0.001953125
# r = 0.000976563
# r = 0.000488282
# r = 0.000244141
# r = 0.00012207
# r = 0.000061035
# r = 0.000030518
# r = 0.000015259
# r = 0.000007629


# Run average perceptron
# aLearned = averagePerceptron(dfTrain, r, T)
biasedDFTrain = pd.DataFrame( addBias(dfTrain) )
biasedDFTest = pd.DataFrame( addBias(dfTest) )
aLearned = averagePerceptron(biasedDFTrain, r, T)



print("Using learning rate,r:", r)
print("Final Learning Weight [b,w1,...,w4]:", aLearned)

###############################################################################
###############################################################################

# Make predictions based on WLearned
def makePredictions(wLearned,df):
    rows = len(df)
    predictions = []
    X = df.iloc[:,:-1].to_numpy()
    
    for i in range (0, rows):
        currPrediction = np.sign( sum(wLearned.transpose() * X[i]) )   # sgn(w^T x)
        
        # Redefine signs to match labels of dataset
        if currPrediction == -1.0:
            currPrediction = 0
        else:
            currPrediction = 1
        
        # Record prediction for current datapoint
        predictions.append(currPrediction)
        
    return predictions 

predictionsWTest = makePredictions(aLearned,biasedDFTest)
# predictionsWTraining = makePredictions(aLearned,biasedDFTrain)


# Compute Average Prediction Error
def checkErr(prediction,df):
    rows = len(prediction)
    numErr = []
    for i in range(0,rows):
        if prediction[i] == df.Label[i]:
            currErr = 0
            numErr.append(currErr)
        else:
            currErr = 1
            numErr.append(currErr)
    
    numErr = sum(numErr)
    predErr = numErr/rows
    return predErr

predErrWTest = checkErr(predictionsWTest,dfTest)
# predErrWTraining = checkErr(predictionsWTraining,dfTrain)









