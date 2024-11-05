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


# Voted Perceptron algorithm
def votedPerceptron(df, r, T):
    
    # Initialize parameters
    dfShape = df.shape
    rows = dfShape[0]
    cols = dfShape[1] - 1             # Exclude the label column
    wLearned = np.zeros(cols)         # Weight vector matches # of columns/attributes
    m = 0                             # Number of initial mistakes is zero
    
    # Initialize storage saving vectors
    wIter = []
    wIter.append(wLearned)
    C = []
    C.append(1)     # Initialize first C to be 1
     
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
                wLearned = wLearned + r*y[i]*X[i]              # Update weight vector ONLY when there is a mistake
                m = m + 1                                      # Increase number of mistakes by one
                C.append(1)                                    # Update the number of predictions made by w_m
                
                wIter.append(wLearned)          # w_m+1 = w_m + r*y_i*x_i
                
            else:
                C[m] = C[m] + 1
        
    
    return wIter, C


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


# Run voted perceptron
# WVec, CVec = votedPerceptron(dfTrain, r, T)
biasedDFTrain = pd.DataFrame( addBias(dfTrain) )
biasedDFTest = pd.DataFrame( addBias(dfTest) )
WVec, CVec = votedPerceptron(biasedDFTrain, r, T)



###############################################################################
###############################################################################

# Make predictions based on learned weights, w, and number of predictions, C
def makePredictions(wVec, cVec, df):
    rows = len(df)
    predictions = []
    X = df.iloc[:,:-1].to_numpy()
    
    for i in range (0, rows):
        
        # Create vector of terms to sum
        terms2Sum = []
        for k in range(1, len(cVec)):
            currTerm = np.sign( sum(wVec[k].transpose() * X[i]) )   # sgn(w^T x)
            currTerm = cVec[k] * currTerm   # c * sgn(w^T x)
            terms2Sum.append(currTerm)
        
        totSumSgn = np.sign( sum(terms2Sum) )   # sgn( sum(c_k * sgn(w_k^T * x_i)) )
                
        # Redefine signs to match labels of dataset (only needed for final sign interpretation)
        if totSumSgn == -1.0:
            totSumSgn = 0
        else:
            totSumSgn = 1
        
        # Record prediction for current datapoint
        predictions.append(totSumSgn)
        
    return predictions 

predictionsWTest = makePredictions(WVec,CVec,biasedDFTest)
# predictionsWTraining = makePredictions(WVec,CVec,biasedDFTrain)


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
    numCorrect = rows - numErr
    return predErr, numCorrect

predErrWTest, numCorrect = checkErr(predictionsWTest,dfTest)
# predErrWTraining = checkErr(predictionsWTraining,dfTrain)



###############################################################################
###############################################################################

# Convert Results into Single Dataframe
vWeights = pd.DataFrame(index=range( len(CVec) ),columns=range( len(WVec[0])) )
for i in range(0,len(vWeights)):
    currW = WVec[i]
    for j in range(0,len(vWeights.columns)):
        vWeights[j][i] = currW[j]

votedWeightsTable = pd.DataFrame(list(zip(vWeights[:][0],vWeights[:][1],vWeights[:][2],vWeights[:][3], vWeights[:][4], CVec)), columns = ['b','w1','w2','w3','w4','C'])

print(votedWeightsTable.to_latex())


# Compare to Average Perceptron:
bAvg = sum(votedWeightsTable.b) #/ len(votedWeightsTable)
w1Avg = sum(votedWeightsTable.w1) #/ len(votedWeightsTable)
w2Avg = sum(votedWeightsTable.w2) #/ len(votedWeightsTable)
w3Avg = sum(votedWeightsTable.w3) #/ len(votedWeightsTable)
w4Avg = sum(votedWeightsTable.w4) #/ len(votedWeightsTable)


print("Sum of All Learning Weights [b,w1,...,w4]: [", bAvg, w1Avg, w2Avg, w3Avg, w4Avg,"]")
