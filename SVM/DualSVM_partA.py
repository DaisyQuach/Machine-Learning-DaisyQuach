# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:36:34 2024

@author: daisy
"""

#%% Import Libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize

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

def dualSVM(df, C):
    
    # Initialize 
    dfShape = df.shape
    rows = dfShape[0]
    cols = dfShape[1] - 1             # Exclude the label column

    # Setup inputs for algorithm 
    X = df.iloc[:,:-1].to_numpy()
    y = df.iloc[:,len(df.columns)-1].to_numpy()
    

    # Compute the Kernal Gram matrix (K_ij = y_i * y_j * x_i^T * x_j)
    K = np.dot(X, X.T) * np.dot(y, y.T)
    
    # Objective function
    def objectiveFunc(alpha):
        return 0.5 * np.dot(alpha, np.dot(K, alpha)) - np.sum(alpha)
    
    # Equality constraint: sum(alpha * y) = 0
    constraints = {
        'type': 'eq',
        'fun': lambda alpha: np.dot(alpha, y),
    }

    # Bounds for alpha: 0 <= alpha <= C
    bounds = [(0, C) for _ in range(rows)]

    # Initial guess for alpha
    alpha0 = np.zeros(rows)

    # Solve the optimization problem
    result = minimize(
        objectiveFunc,
        alpha0,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'
    )

    # Extract the optimal Lagrange multipliers
    alphas = result.x

    # Support vectors have non-zero Lagrange multipliers
    sv = alphas > 1e-5
    alphas_sv = alphas[sv]
    X_sv = X[sv]
    y_sv = y[sv]

    # Compute the weight vector w
    # w_sv = np.sum(alphas_sv[:, None] * y_sv * X_sv, axis=0)
    sumTerms = pd.DataFrame()
    for i in range(len(alphas_sv)):
        currTerm = alphas_sv[i] * y_sv[i] * X_sv[i]
        sumTerms[i] = currTerm.T
    w_sv = np.sum(sumTerms.T)
    w_sv = w_sv.T


    # Compute the bias b
    # b_sv = np.mean(y_sv - np.dot(X_sv, w_sv))
    avgTerms = []
    for i in range(len(y_sv)):
        currTerm = y_sv[i] - np.dot(w_sv.T, X_sv[i])  
        avgTerms.append(currTerm)
    b_sv = np.mean(avgTerms)
    
    
    return w_sv, b_sv

    



#%% Initialize & Run Algorithm

# Hyperparameters
T = 100
C = [(100/873),(500/873),(700/873)]



# Run algorithm
w1, b1 = dualSVM(dfTrain, C[0])
w2, b2 = dualSVM(dfTrain, C[1])
w3, b3 = dualSVM(dfTrain, C[2])

print("C1 Weights (w):", w1)
print("C1 Bias (b):", b1)
print("C2 Weights (w):", w2)
print("C2 Bias (b):", b2)
print("C3 Weights (w):", w3)
print("C3 Bias (b):", b3)


#%% Make predictions based on WLearned
def makePredictions(w,b,df):
    rows = len(df)
    predictions = []
    X = df.iloc[:,:-1].to_numpy()
    
    for i in range (0, rows):
        currPrediction = np.sign( sum(w.transpose() * X[i] + b) )   # sgn(w^T x + b)
        
        # Record prediction for current datapoint
        predictions.append(currPrediction)
        
    return predictions 

predictionsTest1 = makePredictions(w1,b1,dfTest)
predictionsTest2 = makePredictions(w2,b2,dfTest)
predictionsTest3 = makePredictions(w2,b3,dfTest)

predictionsTrain1 = makePredictions(w1,b1,dfTrain)
predictionsTrain2 = makePredictions(w2,b2,dfTrain)
predictionsTrain3 = makePredictions(w3,b3,dfTrain)


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