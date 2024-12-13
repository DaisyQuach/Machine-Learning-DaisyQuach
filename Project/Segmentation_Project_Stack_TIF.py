# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:22:18 2024

@author: daisy
"""

#%% Import libraries and packages
import numpy as np
import cv2
import pandas as pd
import os
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#%% Set directories
train_dir = r'C:\Users\daisy\Desktop\UofU\2024 FALL\CS_6350_MachineLearning\Project\Data\Train'
test_dir = r'C:\Users\daisy\Desktop\UofU\2024 FALL\CS_6350_MachineLearning\Project\Data\Test'

# Define the range of frames to process
frame_range = range(9)  # Process frames 000 to 008

#%% Initialize results
accuracy_results = []

#%% Process each pair of training and testing images
for frame_num in frame_range:
    # Construct file paths
    train_path = os.path.join(train_dir, f"frame_{frame_num:03d}.tiff")
    test_path = os.path.join(test_dir, f"frame_{frame_num:03d}.tiff")

    # Load training and testing images
    img = cv2.imread(train_path, cv2.IMREAD_GRAYSCALE)
    labeled_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if img is None or labeled_img is None:
        print(f"Skipping frame {frame_num:03d}: File not found.")
        continue

    # Initialize dataframe for image data and filters
    df = pd.DataFrame()
    img2 = img.reshape(-1)  # Flatten image
    df['Original Image'] = img2

    #%% Adding filters
    num = 1  # To count numbers for labeling Gabor features
    kernels = []
    for theta in range(2):  # Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  # Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
                for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5
                    gabor_label = f'Gabor{num}'  # Label Gabor columns
                    ksize = 9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)

                    # Filter the image and add values to a new column
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img
                    num += 1

    # CANNY EDGE
    edges = cv2.Canny(img, 100, 200)
    df['Canny Edge'] = edges.reshape(-1)

    # ROBERTS EDGE
    df['Roberts'] = roberts(img).reshape(-1)

    # SOBEL
    df['Sobel'] = sobel(img).reshape(-1)

    # SCHARR
    df['Scharr'] = scharr(img).reshape(-1)

    # PREWITT
    df['Prewitt'] = prewitt(img).reshape(-1)

    # GAUSSIAN with sigma=3
    df['Gaussian s3'] = nd.gaussian_filter(img, sigma=3).reshape(-1)

    # GAUSSIAN with sigma=7
    df['Gaussian s7'] = nd.gaussian_filter(img, sigma=7).reshape(-1)

    # MEDIAN with size=3
    df['Median s3'] = nd.median_filter(img, size=3).reshape(-1)

    # VARIANCE with size=3
    df['Variance s3'] = nd.generic_filter(img, np.var, size=3).reshape(-1)

    #%% Add labels
    df['Label'] = labeled_img.reshape(-1)

    #%% Prepare data for machine learning
    Y = df['Label'].values  # Dependent variable
    X = df.drop(labels=['Label'], axis=1)  # Independent variables

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

    # Train the model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, Y_train)

    # Make predictions and calculate accuracy
    prediction_test = model.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, prediction_test)

    print(f"Frame {frame_num:03d}: Accuracy = {accuracy:.4f}")
    accuracy_results.append((frame_num, accuracy))

#%% Summary of results
print("\nSummary of Accuracy Results:")
for frame_num, accuracy in accuracy_results:
    print(f"Frame {frame_num:03d}: Accuracy = {accuracy:.4f}")
