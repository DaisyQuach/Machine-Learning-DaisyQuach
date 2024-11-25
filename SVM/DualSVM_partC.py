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

    return alphas, X_sv, y_sv, b_sv, sv

    



#%% Initialize & Run Algorithm

# Hyperparameters
T = 100
C = [(100/873),(500/873),(700/873)]
gammas = [0.1, 0.5, 1, 5, 100]


# Run algorithm
alpha10, X_sv10, y_sv10, b_sv10, svInd10 = dualSVM(dfTrain, C[1],gammas[0])

alpha11, X_sv11, y_sv11, b_sv11, svInd11 = dualSVM(dfTrain, C[1],gammas[1])

alpha12, X_sv12, y_sv12, b_sv12, svInd12 = dualSVM(dfTrain, C[1],gammas[2])

alpha13, X_sv13, y_sv13, b_sv13, svInd13 = dualSVM(dfTrain, C[1],gammas[3])

alpha14, X_sv14, y_sv14, b_sv14, svInd14 = dualSVM(dfTrain, C[1],gammas[4])










# results = []
# gammas = [0.01, 0.1, 0.5, 1, 5]
# C_values = [100 / 873, 500 / 873, 700 / 873]

# for C in C_values:
#     for gamma in gammas:
#         alphas, X_sv, y_sv, b_sv, sv_indices = dualSVM(dfTrain, C, gamma)
#         num_sv = len(X_sv)
#         results.append((C, gamma, num_sv, sv_indices))
#         print(f"C: {C}, Gamma: {gamma}, Number of Support Vectors: {num_sv}")


# #%% Count overlapped support vectors for C = 500/873 over different gammas

# C_target = 500 / 873
# gamma_sv_data = [r for r in results if r[0] == C_target]

# # Compute overlap between consecutive gamma values
# for i in range(len(gamma_sv_data) - 1):
#     gamma1, sv_indices1 = gamma_sv_data[i][1], gamma_sv_data[i][3]
#     gamma2, sv_indices2 = gamma_sv_data[i + 1][1], gamma_sv_data[i + 1][3]
#     overlap = len(set(sv_indices1) & set(sv_indices2))
#     print(f"Overlap between Gamma {gamma1} and Gamma {gamma2}: {overlap}")


