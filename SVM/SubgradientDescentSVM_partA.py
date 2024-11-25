# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:36:34 2024

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


#%% Other user-functions
def addBias(df):                         # bias b = ones column vector in beginning of array
    return np.column_stack((np.ones(len(df)), df))


#%%  Primal Stochastic Sub-Gradient Descent

def primalStochasticSubGradientDescent(df, T, C, gamma0, a):
    
    # Initialize 
    dfShape = df.shape
    rows = dfShape[0]
    cols = dfShape[1] - 1             # Exclude the label column
    wLearned = np.zeros(cols)         # Weight vector matches # of columns/attributes

     
    # Initialize storage saving vectors
    wIter = []
    wIter.append(wLearned)
    objectiveIter = []

     
    for e in range(1, T):     # Iterate RANDOMLY based on available examples
        
        # Shuffle the data:
        shuffled_df = df.sample(rows)                              # Shuffle the dataset
        shuffled_df = shuffled_df.reset_index(drop=True)           # Reset the index of the shuffled DataFrame
        
        # Setup inputs for algorithm
        X = shuffled_df.iloc[:,:-1].to_numpy()
        y = shuffled_df.iloc[:,len(shuffled_df.columns)-1].to_numpy()
        
        
        
        
        # For each training example:
        for i in range(0,rows):
            currMargin = y[i] * sum(wLearned.transpose() * X[i])
            gamma_t = gamma0 / (1 + (gamma0 * i)/(a) )              # Update gamma_t for every example
            
            if currMargin <= 1:                                      
                wLearned = wLearned - gamma_t*C*rows*y[i]*X[i]       # Update weight vector
        
        # Values to store per epoch, e:
        wIter.append(wLearned)

        # Compute objective function as function of epochs
        hingeLoss = np.maximum(0, 1 - y * np.dot(X,wLearned)).sum()
        objective = 0.5 * np.dot(wLearned,wLearned) + C*hingeLoss
        objectiveIter.append(objective)
    
    return wLearned, objectiveIter



#%% Initialize & Run Algorithm

# Hyperparameters
T = 100
C = [(100/873),(500/873),(700/873)]

# Tunable parameters (for convergence)
gamma0 = 0.01
a = 100

# Run algorithm
biasedDFTrain = pd.DataFrame( addBias(dfTrain) )
biasedDFTest = pd.DataFrame( addBias(dfTest) )

WLearned_C1, ObjectiveVals1 = primalStochasticSubGradientDescent(biasedDFTrain, T, C[0], gamma0, a)
WLearned_C2, ObjectiveVals2 = primalStochasticSubGradientDescent(biasedDFTrain, T, C[1], gamma0, a)
WLearned_C3, ObjectiveVals3 = primalStochasticSubGradientDescent(biasedDFTrain, T, C[2], gamma0, a)

print("Final Learning Weight [b,w1,...,w4] for C1:", WLearned_C1)
print("Final Learning Weight [b,w1,...,w4] for C2:", WLearned_C2)
print("Final Learning Weight [b,w1,...,w4] for C3:", WLearned_C3)


#%% Plot objective function w/ number of updates to check convergence
import matplotlib.pyplot as plt
plt.plot(range(1, T), ObjectiveVals1)
plt.plot(range(1,T), ObjectiveVals2)
plt.plot(range(1,T), ObjectiveVals3)
plt.xlabel('Number of Epochs')
plt.ylabel('Objective Value')
plt.title('Objective Function at $a = $' + str(a) + ' and $\gamma_0 = $' + str(gamma0))
plt.legend(['C = 100/873','C = 500/873', 'C = 700/873'])
plt.show()
    

#%% Make predictions based on WLearned
def makePredictions(wLearned,df):
    rows = len(df)
    predictions = []
    X = df.iloc[:,:-1].to_numpy()
    
    for i in range (0, rows):
        currPrediction = np.sign( sum(wLearned.transpose() * X[i]) )   # sgn(w^T x)
        
        # Record prediction for current datapoint
        predictions.append(currPrediction)
        
    return predictions 

predictionsTest1 = makePredictions(WLearned_C1,biasedDFTest)
predictionsTest2 = makePredictions(WLearned_C2,biasedDFTest)
predictionsTest3 = makePredictions(WLearned_C3,biasedDFTest)

predictionsTrain1 = makePredictions(WLearned_C1,biasedDFTrain)
predictionsTrain2 = makePredictions(WLearned_C2,biasedDFTrain)
predictionsTrain3 = makePredictions(WLearned_C3,biasedDFTrain)


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

errorTest1 = checkErr(predictionsTest1,dfTest)
errorTest2 = checkErr(predictionsTest2,dfTest)
errorTest3 = checkErr(predictionsTest3,dfTest)

errorTrain1 = checkErr(predictionsTrain1,dfTrain)
errorTrain2 = checkErr(predictionsTrain1,dfTrain)
errorTrain3 = checkErr(predictionsTrain1,dfTrain)