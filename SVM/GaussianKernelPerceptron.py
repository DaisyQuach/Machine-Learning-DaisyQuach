# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:43:04 2024

@author: daisy
"""


#%% Import Libraries
import numpy as np
import pandas as pd

#%% Import bank note data
testData = r"C:\Users\daisy\Documents\GitHub\Machine-Learning-DaisyQuach\SVM\bank-note\test.csv"
dfTest = pd.read_csv(testData, names=['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Label'], header=None,index_col=False)

trainingData = r"C:\Users\daisy\Documents\GitHub\Machine-Learning-DaisyQuach\SVM\bank-note\train.csv"
dfTrain = pd.read_csv(trainingData, names=['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Label'], header=None,index_col=False)


# Convert labels to be in {-1,1}
def convertLabels(df):
    N = len(df)
    for i in range(N):
        if df.Label[i] == 0:
            df.Label[i] = -1
    return df

dfTest = convertLabels(dfTest)
dfTrain = convertLabels(dfTrain)


#%%  Dual Perceptron

def dualPerceptron(df, gamma):
    
    # Initialize 
    dfShape = df.shape
    rows = dfShape[0]
    cols = dfShape[1] - 1    
    countsVec = np.zeros(rows)             # Initialize counts to be zero
    
    # Setup inputs
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, len(df.columns) - 1].to_numpy()
    
    # Initialize storage saving vectors
    guessVec = []
    
    # Compute the Gaussian Kernel Gram matrix
    K = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            K[i, j] = y[i] * y[j] * np.exp(-np.linalg.norm(X[i] - X[j])**2 / gamma)
            
            
    for i in range(rows):
        sumTerms = []
        for j in range(rows):
            currTerm = countsVec[j] * K[i,j]
            sumTerms.append(currTerm)
            
        currGuess = np.sign( y[i] * sum(sumTerms) )
        guessVec.append(currGuess)
        
        if guessVec[i] != y[i]:
            countsVec[i] = countsVec[i] + 1
        
        
        # elif guessVec == y:
        #     break
        
        
    return countsVec, K


#%% Initialize & Run Algorithm

# Hyperparameters
gammas = [0.1, 0.5, 1, 5, 100]

# Run algorithm
counts0, K0 = dualPerceptron(dfTrain, gammas[0])
counts1, K1 = dualPerceptron(dfTrain, gammas[1])
counts2, K2 = dualPerceptron(dfTrain, gammas[2])
counts3, K3 = dualPerceptron(dfTrain, gammas[3])
counts4, K4 = dualPerceptron(dfTrain, gammas[4])



#%% Make predictions based on WLearned
def makePredictions(counts, K, df):

    # Setup inputs
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, len(df.columns) - 1].to_numpy()

    predictions = []
    for i in range(len(X)):
        sumVec = []
        for j in range(len(X)):
            sumTerm = counts[i] * y[i] * K[i,j]
            sumVec.append(sumTerm)
            
        currPrediction = np.sign(sum(sumVec))
        predictions.append(1 if currPrediction > 0 else -1)
        
    return predictions

predictionsTest0 = makePredictions(counts0, K0, dfTest)
predictionsTrain0 = makePredictions(counts0, K0, dfTrain)

predictionsTest1 = makePredictions(counts1, K1, dfTest)
predictionsTrain1 = makePredictions(counts1, K1, dfTrain)

predictionsTest2 = makePredictions(counts2, K2, dfTest)
predictionsTrain2 = makePredictions(counts2, K2, dfTrain)

predictionsTest3 = makePredictions(counts3, K3, dfTest)
predictionsTrain3 = makePredictions(counts3, K3, dfTrain)

predictionsTest4 = makePredictions(counts4, K4, dfTest)
predictionsTrain4 = makePredictions(counts4, K4, dfTrain)

#%% Compute Average Prediction Error
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

errorTest0 = checkErr(predictionsTest0,dfTest)
errorTrain0 = checkErr(predictionsTrain0,dfTrain)

errorTest1 = checkErr(predictionsTest1,dfTest)
errorTrain1 = checkErr(predictionsTrain1,dfTrain)

errorTest2 = checkErr(predictionsTest2,dfTest)
errorTrain2 = checkErr(predictionsTrain2,dfTrain)

errorTest3 = checkErr(predictionsTest3,dfTest)
errorTrain3 = checkErr(predictionsTrain3,dfTrain)

errorTest4 = checkErr(predictionsTest4,dfTest)
errorTrain4 = checkErr(predictionsTrain4,dfTrain)

