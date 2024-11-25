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

def dualSVM(df, C, gamma):
    # Initialize 
    dfShape = df.shape
    rows = dfShape[0]
    cols = dfShape[1] - 1

    # Setup inputs
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, len(df.columns) - 1].to_numpy()

    # Compute the Gaussian Kernel Gram matrix
    K = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            K[i, j] = y[i] * y[j] * np.exp(-np.linalg.norm(X[i] - X[j])**2 / gamma)

    # Objective function
    def objectiveFunc(alpha):
        return 0.5 * np.dot(alpha, np.dot(K, alpha)) - np.sum(alpha)

    # Constraints
    constraints = {
        'type': 'eq',
        'fun': lambda alpha: np.dot(alpha, y),
    }
    bounds = [(0, C) for _ in range(rows)]

    # Solve the optimization problem
    alpha0 = np.zeros(rows)
    result = minimize(objectiveFunc, alpha0, bounds=bounds, constraints=constraints, method='SLSQP')

    # Extract support vectors
    alphas = result.x
    sv = alphas > 1e-5
    alphas_sv = alphas[sv]
    X_sv = X[sv]
    y_sv = y[sv]

    # Compute the bias term b
    # b_sv = np.mean([y_sv[i] - np.sum(alphas_sv * y_sv * 
    #             np.exp(-np.linalg.norm(X_sv[i] - X_sv[j])**2 / gamma)) for i in range(len(y_sv))])
    
    avgTerms = []
    for k in range(len(y_sv)):
        preSum = []
        for i in range(len(y_sv)):
            currKer = np.exp(-np.linalg.norm(X_sv[i] - X_sv[k])**2 / gamma)
            currSumTerm = alphas_sv[i] * y_sv[i] * currKer
            preSum.append(currSumTerm)
            
        currTerm = y_sv[k] - sum(preSum)
        avgTerms.append(currTerm)
        
    b_sv = np.mean(avgTerms)

    return alphas, X_sv, y_sv, b_sv

    



#%% Initialize & Run Algorithm

# Hyperparameters
T = 100
C = [(100/873),(500/873),(700/873)]
gammas = [0.1, 0.5, 1, 5, 100]


# Run algorithm
# alpha00, X_sv00, y_sv00, b_sv00 = dualSVM(dfTrain, C[0],gammas[0])
alpha10, X_sv10, y_sv10, b_sv10 = dualSVM(dfTrain, C[1],gammas[0])
# alpha20, X_sv20, y_sv20, b_sv20 = dualSVM(dfTrain, C[2],gammas[0])

# alpha01, X_sv01, y_sv01, b_sv01 = dualSVM(dfTrain, C[0],gammas[1])
alpha11, X_sv11, y_sv11, b_sv11 = dualSVM(dfTrain, C[1],gammas[1])
# alpha21, X_sv21, y_sv21, b_sv21 = dualSVM(dfTrain, C[2],gammas[1])

# alpha02, X_sv02, y_sv02, b_sv02 = dualSVM(dfTrain, C[0],gammas[2])
alpha12, X_sv12, y_sv12, b_sv12 = dualSVM(dfTrain, C[1],gammas[2])
# alpha22, X_sv22, y_sv22, b_sv22 = dualSVM(dfTrain, C[2],gammas[2])

# alpha03, X_sv03, y_sv03, b_sv03 = dualSVM(dfTrain, C[0],gammas[3])
alpha13, X_sv13, y_sv13, b_sv13 = dualSVM(dfTrain, C[1],gammas[3])
# alpha23, X_sv23, y_sv23, b_sv23 = dualSVM(dfTrain, C[2],gammas[3])

# alpha04, X_sv04, y_sv04, b_sv04 = dualSVM(dfTrain, C[0],gammas[4])
alpha14, X_sv14, y_sv14, b_sv14 = dualSVM(dfTrain, C[1],gammas[4])
# alpha24, X_sv24, y_sv24, b_sv24 = dualSVM(dfTrain, C[2],gammas[4])

#%% Make predictions based on WLearned
def makePredictions(alphas, X_sv, y_sv, b_sv, gamma, df):
    X = df.iloc[:, :-1].to_numpy()
    predictions = []
    for x in X:
        pred = np.sum([
            alphas[i] * y_sv[i] * np.exp(-np.linalg.norm(x - X_sv[i])**2 / gamma)
            for i in range(len(X_sv))
        ]) + b_sv
        predictions.append(np.sign(pred))
    return predictions

# predictionsTest00 = makePredictions(alpha00, X_sv00, y_sv00, b_sv00, gammas[0], dfTest)
# predictionsTrain00 = makePredictions(alpha00, X_sv00, y_sv00, b_sv00, gammas[0], dfTrain)

predictionsTest10 = makePredictions(alpha10, X_sv10, y_sv10, b_sv10, gammas[0], dfTest)
predictionsTrain10 = makePredictions(alpha10, X_sv10, y_sv10, b_sv10, gammas[0], dfTrain)

# predictionsTest20 = makePredictions(alpha20, X_sv20, y_sv20, b_sv20, gammas[0], dfTest)
# predictionsTrain20 = makePredictions(alpha20, X_sv20, y_sv20, b_sv20, gammas[0], dfTrain)

##########################################################################################

# predictionsTest01 = makePredictions(alpha01, X_sv01, y_sv01, b_sv01, gammas[1], dfTest)
# predictionsTrain01 = makePredictions(alpha01, X_sv01, y_sv01, b_sv01, gammas[1], dfTrain)

predictionsTest11 = makePredictions(alpha11, X_sv11, y_sv11, b_sv11, gammas[1], dfTest)
predictionsTrain11 = makePredictions(alpha11, X_sv11, y_sv11, b_sv11, gammas[1], dfTrain)

# predictionsTest21 = makePredictions(alpha21, X_sv21, y_sv21, b_sv21, gammas[1], dfTest)
# predictionsTrain21 = makePredictions(alpha21, X_sv21, y_sv21, b_sv21, gammas[1], dfTrain)

##########################################################################################

# predictionsTest02 = makePredictions(alpha02, X_sv02, y_sv02, b_sv02, gammas[2], dfTest)
# predictionsTrain02 = makePredictions(alpha02, X_sv02, y_sv02, b_sv02, gammas[2], dfTrain)

predictionsTest12 = makePredictions(alpha12, X_sv12, y_sv12, b_sv12, gammas[2], dfTest)
predictionsTrain12 = makePredictions(alpha12, X_sv12, y_sv12, b_sv12, gammas[2], dfTrain)

# predictionsTest22 = makePredictions(alpha22, X_sv22, y_sv22, b_sv22, gammas[2], dfTest)
# predictionsTrain22 = makePredictions(alpha22, X_sv22, y_sv22, b_sv22, gammas[2], dfTrain)

##########################################################################################

# predictionsTest03 = makePredictions(alpha03, X_sv03, y_sv03, b_sv03, gammas[3], dfTest)
# predictionsTrain03 = makePredictions(alpha03, X_sv03, y_sv03, b_sv03, gammas[3], dfTrain)

predictionsTest13 = makePredictions(alpha13, X_sv13, y_sv13, b_sv13, gammas[3], dfTest)
predictionsTrain13 = makePredictions(alpha13, X_sv13, y_sv13, b_sv13, gammas[3], dfTrain)

# predictionsTest23 = makePredictions(alpha23, X_sv23, y_sv23, b_sv23, gammas[3], dfTest)
# predictionsTrain23 = makePredictions(alpha23, X_sv23, y_sv23, b_sv23, gammas[3], dfTrain)

##########################################################################################

# predictionsTest04 = makePredictions(alpha04, X_sv04, y_sv04, b_sv04, gammas[4], dfTest)
# predictionsTrain04 = makePredictions(alpha04, X_sv04, y_sv04, b_sv04, gammas[4], dfTrain)

predictionsTest14 = makePredictions(alpha14, X_sv14, y_sv14, b_sv14, gammas[4], dfTest)
predictionsTrain14 = makePredictions(alpha14, X_sv14, y_sv14, b_sv14, gammas[4], dfTrain)

# predictionsTest24 = makePredictions(alpha24, X_sv24, y_sv24, b_sv24, gammas[4], dfTest)
# predictionsTrain24 = makePredictions(alpha24, X_sv24, y_sv24, b_sv24, gammas[4], dfTrain)



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

# errorTest00 = checkErr(predictionsTest00,dfTest)
# errorTrain00 = checkErr(predictionsTrain00,dfTrain)

errorTest10 = checkErr(predictionsTest10,dfTest)
errorTrain10 = checkErr(predictionsTrain10,dfTrain)

# errorTest20 = checkErr(predictionsTest20,dfTest)
# errorTrain20 = checkErr(predictionsTrain20,dfTrain)

##########################################################################################

# errorTest01 = checkErr(predictionsTest01,dfTest)
# errorTrain01 = checkErr(predictionsTrain01,dfTrain)

errorTest11 = checkErr(predictionsTest11,dfTest)
errorTrain11 = checkErr(predictionsTrain11,dfTrain)

# errorTest21 = checkErr(predictionsTest21,dfTest)
# errorTrain21 = checkErr(predictionsTrain21,dfTrain)

##########################################################################################

# errorTest02 = checkErr(predictionsTest02,dfTest)
# errorTrain02 = checkErr(predictionsTrain02,dfTrain)

errorTest12 = checkErr(predictionsTest12,dfTest)
errorTrain12 = checkErr(predictionsTrain12,dfTrain)

# errorTest22 = checkErr(predictionsTest22,dfTest)
# errorTrain22 = checkErr(predictionsTrain22,dfTrain)

##########################################################################################

# errorTest03 = checkErr(predictionsTest03,dfTest)
# errorTrain03 = checkErr(predictionsTrain03,dfTrain)

errorTest13 = checkErr(predictionsTest13,dfTest)
errorTrain13 = checkErr(predictionsTrain13,dfTrain)

# errorTest23 = checkErr(predictionsTest23,dfTest)
# errorTrain23 = checkErr(predictionsTrain23,dfTrain)

##########################################################################################

# errorTest04 = checkErr(predictionsTest04,dfTest)
# errorTrain04 = checkErr(predictionsTrain04,dfTrain)

errorTest14 = checkErr(predictionsTest14,dfTest)
errorTrain14 = checkErr(predictionsTrain14,dfTrain)

# errorTest24 = checkErr(predictionsTest24,dfTest)
# errorTrain24 = checkErr(predictionsTrain24,dfTrain)

