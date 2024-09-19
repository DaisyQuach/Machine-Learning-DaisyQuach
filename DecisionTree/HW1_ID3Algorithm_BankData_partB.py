# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:04:20 2024

Code implementing the ID3 algorithm that supports information gain, majority error, and gini index.
Users are allowed to set the maximum tree depth.

@author: daisy
"""

# Import Libraries
import numpy as np
import pandas as pd


# Import bank data
testData = r"C:\Users\daisy\Documents\GitHub\Machine-Learning-DaisyQuach\DecisionTree\bank\test.csv"
dfTest = pd.read_csv(testData, names=['age', 'job', 'marital', 'education', 'default', 'balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y'], header=None,index_col=False)

trainingData = r"C:\Users\daisy\Documents\GitHub\Machine-Learning-DaisyQuach\DecisionTree\bank\train.csv"
df = pd.read_csv(trainingData, names=['age', 'job', 'marital', 'education', 'default', 'balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y'], header=None,index_col=False)

# Categorical data
job = ('admin.','unknown','unemployed','management','housemaid','entrepreneur','student','blue-collar','self-employed','retired','technician','services')
maritial = ('married','divorced','single')
education = ('unknown','secondary','primary','tertiary')
default = ('yes','no')
housing = ('yes','no')
loan = ('yes','no')
contact = ('unknown','telephone','cellular')
month = ('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec')
poutcome = ('unknown','other','failure','success')

# Numerical data (set median as the threshold)
age = ('aboveThresh','belowThresh')
balance = ('aboveThresh','belowThresh')
day = ('aboveThresh','belowThresh')
duration = ('aboveThresh','belowThresh')
campaign = ('aboveThresh','belowThresh')
pdays = ('aboveThresh','belowThresh')
previous = ('aboveThresh','belowThresh')

# Redefining numerical data in dataframes:
medTrain = df.median()
medTest = dfTest.median()



# Parameters
attributes = ['age', 'job', 'marital', 'education', 'default', 'balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
values = [age,job,maritial,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome]
labels = ['yes', 'no']
bestAttributeMethod = 'Information Gain'
maxTreeDepth = 15
counter = 0


# Create a Class that Defines a Node
class Node:
        def __init__(self, bestAttribute, branchValues, leafRules):
            self.bestAttribute = bestAttribute
            self.branchValues = branchValues
            self.leafRules = leafRules


# Gain-Related Functions (Entropy, Information Gain, Majority Error, Gini Index)
def entropy(S, labels):    
    numLabels = len(labels)
    quantLabel = list(range(0,numLabels))
    for i in range(0,numLabels):                                            # Run through all labels
        checkLabel = labels[i] in S['y'].values
        if checkLabel == True:
            quantLabel[i] = S['y'].value_counts()[labels[i]]                # Compute quantities for n labels
        else:
            quantLabel[i] = 0
    total = sum(quantLabel)
    probabilities = np.divide(quantLabel, total)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy
  
def majorityError(S, labels):    
    numLabels = len(labels)
    quantLabel = list(range(0,numLabels))
    for i in range(0,numLabels):                                            # Run through all labels
        checkLabel = labels[i] in S['y'].values
        if checkLabel == True:
            quantLabel[i] = S['y'].value_counts()[labels[i]]                # Compute quantities for n labels
        else:
            quantLabel[i] = 0    
    total = sum(quantLabel)
    probabilities = np.divide(quantLabel, total)
    majorityError = min(probabilities)
    return majorityError

def giniIndex(S, labels):    
    numLabels = len(labels)
    quantLabel = list(range(0,numLabels))
    for i in range(0,numLabels):                                            # Run through all labels
        checkLabel = labels[i] in S['y'].values
        if checkLabel == True:
            quantLabel[i] = S['y'].value_counts()[labels[i]]                # Compute quantities for n labels
        else:
            quantLabel[i] = 0
    total = sum(quantLabel)
    probabilities = np.divide(quantLabel, total)
    giniIndex = 1 - sum(probabilities**2)
    return giniIndex


# Other Functions
def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

def mostCommonLabel(colVec):
    return max(set(colVec), key=colVec.count)

def best_attribute(S,Attributes,Labels, BestAttributeMethod):
    if BestAttributeMethod == 'Information Gain':
        totEnt = entropy(S,Labels)
        infoGain = []
        for i in range(0, len(Attributes)):                                 # Calculate entropy for each attribute
            currExpEnt = []
            for j in range(0, len(values[i])):    
                currAttSet = S[S[Attributes[i]] == values[i][j]]            # Create temporary subset where attribute = value
                currExpEnt.insert(j, np.multiply( np.divide(len(currAttSet),len(S)), entropy(currAttSet, Labels)))
            infoGain.insert(i, totEnt - sum(currExpEnt))
        bestAttributeInd = infoGain.index(max(infoGain))
        bestAttribute = Attributes[bestAttributeInd]                        # Obtain best attribute
        
    elif BestAttributeMethod == 'Majority Error':
        totME = majorityError(S,Labels)
        infoGain = []
        for i in range(0, len(Attributes)):                                 # Calculate entropy for each attribute
            currExpEnt = []
            for j in range(0, len(values[i])):    
                currAttSet = S[S[Attributes[i]] == values[i][j]]            # Create temporary subset where attribute = value
                currExpEnt.insert(j, np.multiply( np.divide(len(currAttSet),len(S)), majorityError(currAttSet, Labels)))
            infoGain.insert(i, totME - sum(currExpEnt))
        bestAttributeInd = infoGain.index(max(infoGain))
        bestAttribute = Attributes[bestAttributeInd]                        # Obtain best attribute

    elif BestAttributeMethod == 'Gini Index':
        totGI = giniIndex(S,Labels)
        infoGain = []
        for i in range(0, len(Attributes)):                                 # Calculate entropy for each attribute
            currExpEnt = []
            for j in range(0, len(values[i])):    
                currAttSet = S[S[Attributes[i]] == values[i][j]]            # Create temporary subset where attribute = value
                currExpEnt.insert(j, np.multiply( np.divide(len(currAttSet),len(S)), giniIndex(currAttSet, Labels)))
            infoGain.insert(i, totGI - sum(currExpEnt))
        bestAttributeInd = infoGain.index(max(infoGain))
        bestAttribute = Attributes[bestAttributeInd]                        # Obtain best attribute

    else:
        print("Please choose an available best attribute method.")
        bestAttribute = None
        bestAttributeInd = None
        
    return bestAttribute, bestAttributeInd
        
def categorify_df(Dataframe,Median):
    numericalAtt = Median.index
    for i in range(0, len(Dataframe)):
        for j in range(0,len(Median)):
            if Dataframe[ numericalAtt[j] ][i] <= Median[j]:
                Dataframe[ numericalAtt[j] ][i] = 'belowThresh'
            else:
                Dataframe[ numericalAtt[j] ][i] = 'aboveThresh'
    return Dataframe
        
def replaceUnknowns(Dataframe):
    attributes = Dataframe.columns
    for i in range(0, len(Dataframe)):
        for j in range(0,len(Dataframe.columns)):
            if Dataframe[ attributes[j] ][i] == 'unknown':
                listCommonLabel = Dataframe[ attributes[j] ].mode()
                Dataframe[ attributes[j] ][i] = listCommonLabel[0]
            else:
                Dataframe[ attributes[j] ][i] = Dataframe[ attributes[j] ][i]
    return Dataframe

# Redefining data in dataframes:
df = categorify_df(df,medTrain)
dfTest = categorify_df(dfTest,medTest)
df = replaceUnknowns(df)
dfTest = replaceUnknowns(dfTest)
    

# ID3 Algorithm
def id3_algorithm(S,Attributes,Values,Labels,BestAttributeMethod, MaxTreeDepth,Counter):
    
    # Reset indices for the set/subset:
    S = S.reset_index(drop=True)
    # print("The current level is: ", Counter) 

    if Counter > MaxTreeDepth:                                              # Check if max tree depth is met every time it is called
        # print("The ID3 algorithm has been cutoff by specified max tree depth.")
        currNode = 'Max Depth'
    
    elif is_unique(S['y']) == True:                                         # Check if labels are all the same in given set
        # print("Labels are all the same. Return leaf node.")
        currNode = S['y'][0]                                                # This is a leaf node
        
    elif all(S) == False:                                                   # Check if there are any missing values
        # print("There is a missing value. Return leaf node w/ common label.")
        listCommonLabel = S.labels.mode()
        currNode = listCommonLabel[0]                                       # Create leaf node w/ common label
        
    else:                                                                   # Otherwise create node for tree
        # print('Checked for leaf nodes and missing values. Now running ID3 algorithm.')
        
        # Initialize the node:
        Counter += 1
        # print("The level just increased to: ", Counter) 
        currNode = Node(None,None,None)
        
        # Obtain best attribute w/ index & assign branches:
        [bestAttribute, bestAttributeInd] = best_attribute(df,Attributes,Labels, BestAttributeMethod)
        currNode = Node( bestAttribute,Values[bestAttributeInd],[None]*len(Values[bestAttributeInd]) )

        # Create corresponding subsets, attributes, and values for each branch:
        for i in range(0,len(Values[bestAttributeInd])):                    # Iterate through each applicable branch
            # print("For the current BA, the ", i, "value is running.")
            currSubset = S[S[Attributes[bestAttributeInd]] == Values[bestAttributeInd][i]]      # Create subset to determinate next node/leaf
            currAttributes = Attributes[:bestAttributeInd] + Attributes[bestAttributeInd+1:]    # Update attributes to exclude currBestAttribute
            currValues = Values[:bestAttributeInd] + Values[bestAttributeInd+1:]                # Update values too??? to exclude currBestAttribute values
            
            # Check subsets:
            if currSubset.empty:                                            # If the subset is empty
                listCommonLabel = S.y.mode()
                currNode.leafRules[i] = listCommonLabel[0]                  # Create leaf node w/ common label
            else:
                currNode.leafRules[i] = id3_algorithm(currSubset, currAttributes, currValues, Labels, BestAttributeMethod, MaxTreeDepth, Counter)
        
    return currNode

decisionTree = id3_algorithm(df,attributes,values,labels,bestAttributeMethod,maxTreeDepth,counter)




# Prediction Functions
def singlePrediction(DecisionTree,TestData,i):
    # Initialize:
    currNode = DecisionTree
    if isinstance(currNode,Node) == False:                                  # If the current node is a leaf node (not a node class)
        prediction = currNode
    else:
        currAtt = currNode.bestAttribute                                    # Attribute of the current node
        currValofAtt = TestData[currAtt][i]                                 # Value of the current attribute
        attInd = attributes.index(currAtt)                                  # Index of the current attribute
        valInd = values[attInd].index(currValofAtt)                         # Index of the value of the current attribute
        if isinstance(currValofAtt,Node) == True:                           # Check if current value of attribute is a node
            prediction = currNode.leafRules[valInd]
        else:
            currNode = currNode.leafRules[valInd]
            prediction = singlePrediction(currNode,TestData,i)
    return prediction


def fullPrediction(DecisionTree,TestData):
    predictions = [None]*len(TestData)
    for i in range (0, len(TestData)):
        predictions[i] = singlePrediction(DecisionTree, TestData,i)
    return predictions 


predictionsWTest = fullPrediction(decisionTree,dfTest)
predictionsWTraining = fullPrediction(decisionTree,df)



# Compute Error
def checkErr(Prediction,TestData):
    totEx = len(Prediction)
    numErr = [None]*totEx
    for i in range(0,totEx):
        if Prediction[i] == TestData.y[i]:
            numErr[i] = 0
        else:
            numErr[i] = 1
    numErr = sum(numErr)
    predErr = numErr/totEx
    return predErr

predictedErrWTest = checkErr(predictionsWTest,dfTest)
predictedErrWTraining = checkErr(predictionsWTraining,df)
