# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:10:42 2024

@author: daisy
"""

# Import Libraries
import numpy as np
import pandas as pd


# Import concrete data
testData = r"C:\Users\daisy\Documents\GitHub\Machine-Learning-DaisyQuach\LinearRegression\concrete\concrete\test.csv"
dfTest = pd.read_csv(testData, names=['Cement', 'Slag', 'Fly Ash', 'Water', 'SP', 'CourseAggr','Fine Aggr','Output'], header=None,index_col=False)

trainingData = r"C:\Users\daisy\Documents\GitHub\Machine-Learning-DaisyQuach\LinearRegression\concrete\concrete\train.csv"
dfTrain = pd.read_csv(testData, names=['Cement', 'Slag', 'Fly Ash', 'Water', 'SP', 'CourseAggr','Fine Aggr','Output'], header=None,index_col=False)


# Setup inputs and outputs:
xTest = dfTest.iloc[:,:-1].to_numpy()
yTest = dfTest.iloc[:,len(dfTest.columns)-1].to_numpy()
xTrain = dfTrain.iloc[:,:-1].to_numpy()
yTrain = dfTrain.iloc[:,len(dfTest.columns)-1].to_numpy()


# Analytical weight for training data
xTx_training = np.dot(xTrain.T,xTrain)
inv_xTx_training = np.linalg.inv(xTx_training)
xTy_training = np.dot(xTrain.T,yTrain)
w_train = np.dot(inv_xTx_training,xTy_training)

print("Analytical weights of training data:", w_train)

# Analytical weight for test data
xTx_test = np.dot(xTest.T,xTest)
inv_xTx_test = np.linalg.inv(xTx_test)
xTy_test = np.dot(xTrain.T,yTrain)
w_test = np.dot(inv_xTx_training,xTy_test)

print("Analytical weights of training data:", w_train)
