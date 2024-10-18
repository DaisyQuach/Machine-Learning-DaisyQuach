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

def costFunc(X, y, w):              # Compute cost of given iteration
    m = len(y)
    h = X.dot(w)
    cost = (1 / (2 * m)) * np.sum((h - y) ** 2)
    return cost


# Stochastic gradient descent algorithm
def stochasticGradientDescent(X, y, wLearned, r, maxIter, errThreshold):
    
    # Initialize storage saving data
    m = len(y)
    wIter = []
    wIter.append(wLearned)
    costIter = []
     
     
    for i in range(0, maxIter):     # Iterate RANDOMLY based on available examples
        
        # Select random example:
        randInd = np.random.randint(0,m)
        Xi = np.array([X[randInd,:]])
        yi = np.ones(len(wLearned)) * y[randInd]

        # Compute weight while iterating through examples:
        h = Xi.dot(wLearned)
        diff = h - yi
        gradient = Xi.dot(diff)
        wLearned = wLearned - r * gradient
        cost = costFunc(Xi, yi, wLearned)
        

        # Values to store per iteration:
        costIter.append(cost)
        wIter.append(wLearned)
    
           
    return wLearned, costIter



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
# r = 0.00390625
# r = 0.001953125
# r = 0.000976563
# r = 0.000488282
r = 0.000244141
# r = 0.00012207
# r = 0.000061035
# r = 0.000030518
# r = 0.000015259
# r = 0.000007629


# Run algorithm
WLearned, CostIter = stochasticGradientDescent(xBiased, yTrain, w_0, r, maxIter, errThreshold)

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
