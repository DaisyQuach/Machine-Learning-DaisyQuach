# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:02:44 2024

Python code implementing the stochastic gradient descent algorithm that tunes the rate r to ensure convergence.

@author: daisy
"""

# Import Libraries
import numpy as np
import pandas as pd
import math


# Import concrete data
testData = r"C:\Users\daisy\Documents\GitHub\Machine-Learning-DaisyQuach\LinearRegression\concrete\concrete\test.csv"
dfTest = pd.read_csv(testData, names=['Cement', 'Slag', 'Fly Ash', 'Water', 'SP', 'CourseAggr','Fine Aggr','Output'], header=None,index_col=False)

trainingData = r"C:\Users\daisy\Documents\GitHub\Machine-Learning-DaisyQuach\LinearRegression\concrete\concrete\train.csv"
dfTrain = pd.read_csv(testData, names=['Cement', 'Slag', 'Fly Ash', 'Water', 'SP', 'CourseAggr','Fine Aggr','Output'], header=None,index_col=False)


# Other user-functions
def addBias(X):                         # bias b = ones column vector in beginning of array
    return np.column_stack((np.ones(len(X)), X))


def costFunc(X, y, theta):              # Compute cost of given iteration
    m = len(y)
    h = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((h - y) ** 2)
    return cost


# Batch gradient descent algorithm
def gradient_descent(X, y, wLearned, r, maxIter, errThreshold):
    
    # Initialize storage saving data
    m = len(y)
    costIter = []
    wIter = []
    wIter.append(wLearned)
    errVec = []
     
     
    for t in range(1, maxIter):
        h = X.dot(wLearned)
        diff = h - y
        gradient = (1 / m) * X.T.dot(diff)
        wLearned = wLearned - r * gradient
        cost = costFunc(X, y, wLearned)
         
        # Values to store per iteration
        costIter.append(cost)
        wIter.append(wLearned)
     
        # Check for total error 
        errDiff = wIter[t] - wIter[t-1]
        for k in range(0,len(errDiff)-1):
            errDiff[k] = errDiff[k]**2
        error = math.sqrt(np.sum(errDiff))
        errVec.append(error)
            
        if error <= errThreshold:
            break        # stop iterating once error condition is met
    
           
    return wLearned, costIter, errVec



# Setup inputs for algorithm:
xTest = dfTest.iloc[:,:-1].to_numpy()
yTest = dfTest.iloc[:,len(dfTest.columns)-1].to_numpy()
xTrain = dfTrain.iloc[:,:-1].to_numpy()
yTrain = dfTrain.iloc[:,len(dfTest.columns)-1].to_numpy()
xBiased = addBias(xTrain)

# Initial parameters
w_0 = np.zeros(xBiased.shape[1])
errThreshold = 10**-6
maxIter = 1000

# Test for different learning rates 
# r = 1
# r = 0.5
# r = 0.25
# r = 0.125
# r = 0.0625
# r = 0.03125
# r = 0.015625
# r = 0.0078125
r = 0.00390625
# r = 0.001953125
# r = 0.000976563

# Run algorithm
WLearned, CostIter, Errors = gradient_descent(xBiased, yTrain, w_0, r, maxIter,errThreshold)

print("Final Parameters [b,w1,...,w7]:", WLearned)
# print("Iterated Costs:", CostIter)


# Plot the cost iterations
import matplotlib.pyplot as plt
plt.plot(CostIter)
plt.xlabel('Iterations, t')
plt.ylabel('Cost Value')
plt.title('Cost Function in Batch Gradient Descent Iterations')
plt.show()


# Use final weight vector to calculate cost function of test data
testDataCost = costFunc(xTest, yTest, WLearned[1:])
print("Cost of testing data:", testDataCost)
