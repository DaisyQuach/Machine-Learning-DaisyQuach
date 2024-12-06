# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:56:46 2024

@author: daisy
"""

#%% Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


#%% Import bank note data
testData = r"C:\\Users\\daisy\\Documents\\GitHub\\Machine-Learning-DaisyQuach\\Neural Networks\\bank-note\\test.csv"
dfTest = pd.read_csv(testData, names=['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Label'], header=None, index_col=False)

trainingData = r"C:\\Users\\daisy\\Documents\\GitHub\\Machine-Learning-DaisyQuach\\Neural Networks\\bank-note\\train.csv"
dfTrain = pd.read_csv(trainingData, names=['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Label'], header=None, index_col=False)



#%% Function to create a flexible three-layer neural network
class FlexibleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        super(FlexibleNN, self).__init__()
        
        layers = []
        previous_size = input_size

        # Add hidden layers
        for size in hidden_sizes:
            layers.append(nn.Linear(previous_size, size))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            previous_size = size

        # Add output layer
        layers.append(nn.Linear(previous_size, output_size))
        layers.append(nn.Sigmoid())  # Assuming a binary classification task

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Xavier and He initialization functions
def initialize_weights(m, activation):
    if isinstance(m, nn.Linear):
        if activation == "tanh":
            nn.init.xavier_uniform_(m.weight)
        elif activation == "relu":
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0)

#%% Training function
def train_model(model, dataloader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return loss.item()

#%% Evaluation function
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.round() == labels).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)


#%% Prepare data for PyTorch
X_train = torch.tensor(dfTrain.iloc[:, :-1].values, dtype=torch.float32)
y_train = torch.tensor(dfTrain.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(dfTest.iloc[:, :-1].values, dtype=torch.float32)
y_test = torch.tensor(dfTest.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

results = {}

input_size = X_train.shape[1]
output_size = 1
num_epochs = 20
activation_functions = ["tanh", "relu"]
depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]

for activation in activation_functions:
    for depth in depths:
        for width in widths:
            hidden_sizes = [width] * depth
            
            # Initialize model
            model = FlexibleNN(input_size, hidden_sizes, output_size, activation)
            model.apply(lambda m: initialize_weights(m, activation))

            # Set up optimizer and loss
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.BCELoss()

            # Train the model
            train_model(model, train_loader, optimizer, criterion, num_epochs)

            # Evaluate the model
            train_loss, train_accuracy = evaluate_model(model, train_loader, criterion)
            test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)

            # Record results
            results[(activation, depth, width)] = {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy
            }

# Print results
for (activation, depth, width), metrics in results.items():
    print(f"Activation: {activation}, Depth: {depth}, Width: {width}")
    print(f"  Train Loss: {metrics['train_loss']:.4f}, Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Test Loss: {metrics['test_loss']:.4f}, Test Accuracy: {metrics['test_accuracy']:.4f}\n")
