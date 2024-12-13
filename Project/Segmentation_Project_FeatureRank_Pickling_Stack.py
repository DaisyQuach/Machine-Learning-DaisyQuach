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
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
import matplotlib.pyplot as plt

#%%############# Upload TIFF Stack ##############

# (Train / Original Images)
input_fileTrain = r'C:\Users\daisy\Desktop\UofU\2024 FALL\CS_6350_MachineLearning\Project\sandstone_data_for_ML\partial_labels_for_traditional_ML\sandstone_train_images.tif'  
output_directoryTrain = r'C:\Users\daisy\Desktop\UofU\2024 FALL\CS_6350_MachineLearning\Project\Data\Train'

# Create the output directory if it doesn't exist
os.makedirs(output_directoryTrain, exist_ok=True)

# Open the TIFF stack
with Image.open(input_fileTrain) as img:
    frame = 0
    while True:
        output_path = os.path.join(output_directoryTrain, f"frame_{frame:03d}.tiff")
        img.save(output_path)
        print(f"Saved {output_path}")
        frame += 1
        try:
            img.seek(frame)
        except EOFError:
            break


# (Test / Masked Images)
input_fileTest = r'C:\Users\daisy\Desktop\UofU\2024 FALL\CS_6350_MachineLearning\Project\sandstone_data_for_ML\partial_labels_for_traditional_ML\sandstone_partial_labels_from_APEER_ML.tif'
output_directoryTest = r'C:\Users\daisy\Desktop\UofU\2024 FALL\CS_6350_MachineLearning\Project\Data\Test'

# Create the output directory if it doesn't exist
os.makedirs(output_directoryTest, exist_ok=True)

# Open the TIFF stack
with Image.open(input_fileTest) as img:
    frame = 0
    while True:
        output_path = os.path.join(output_directoryTest, f"frame_{frame:03d}.tiff")
        img.save(output_path)
        print(f"Saved {output_path}")
        frame += 1
        try:
            img.seek(frame)
        except EOFError:
            break


#%%############# Importing Images and Feature Extraction for Each Frame ##############
for frame_num in range(9):  # frames 000 to 008
    df = pd.DataFrame()

    # Load train image
    train_img_path = os.path.join(output_directoryTrain, f"frame_{frame_num:03d}.tiff")
    img_train = cv2.imread(train_img_path, cv2.IMREAD_GRAYSCALE)
    
    # Flatten image for use in dataframe
    img2_train = img_train.reshape(-1)
    df[f'Original Image {frame_num}'] = img2_train

    # Apply Gabor filters and other features
    num = 1
    kernels = []
    for theta in range(2):  # Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  # Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
                for gamma in (0.05, 0.5):  # Gamma values of 0.05 and 0.5
                    gabor_label = f'Gabor{num}'
                    ksize = 9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    
                    # Filter the image and add values to a new column
                    fimg = cv2.filter2D(img_train, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img
                    num += 1  # Increment for Gabor column label

    # Apply edge filters (Canny, Roberts, Sobel, etc.)
    edges = cv2.Canny(img_train, 100, 200)  
    df[f'Canny Edge {frame_num}'] = edges.reshape(-1)
    
    df[f'Roberts {frame_num}'] = roberts(img_train).reshape(-1)
    df[f'Sobel {frame_num}'] = sobel(img_train).reshape(-1)
    df[f'Scharr {frame_num}'] = scharr(img_train).reshape(-1)
    df[f'Prewitt {frame_num}'] = prewitt(img_train).reshape(-1)
    
    df[f'Gaussian s3 {frame_num}'] = nd.gaussian_filter(img_train, sigma=3).reshape(-1)
    df[f'Gaussian s7 {frame_num}'] = nd.gaussian_filter(img_train, sigma=7).reshape(-1)
    df[f'Median s3 {frame_num}'] = nd.median_filter(img_train, size=3).reshape(-1)
    df[f'Variance s3 {frame_num}'] = nd.generic_filter(img_train, np.var, size=3).reshape(-1)

    # Load corresponding test (masked) image for labeling
    test_img_path = os.path.join(output_directoryTest, f"frame_{frame_num:03d}.tiff")
    img_test = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    df[f'Label {frame_num}'] = img_test.reshape(-1)

    # Dependent variable
    Y = df.filter(like='Label').values.flatten()
    # Independent variables (drop label columns)
    X = df.drop(columns=[col for col in df.columns if 'Label' in col])

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

    # Import ML algorithm and train the model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, Y_train)

    # Predictions
    prediction_test = model.predict(X_test)

    # Accuracy
    print(f"Accuracy for frame_{frame_num:03d}: ", metrics.accuracy_score(Y_test, prediction_test))

    # Feature ranking
    importances = list(model.feature_importances_)
    features_list = list(X.columns)
    feature_imp = pd.Series(model.feature_importances_, index=features_list).sort_values(ascending=False)
    print(f"Feature importance for frame_{frame_num:03d}:")
    print(feature_imp)

    # Pickling Model
    pickle_filename = f'sandstone_model_{frame_num:03d}.pkl'
    pickle.dump(model, open(pickle_filename, 'wb'))
    
    # Load model and make prediction
    load_model = pickle.load(open(pickle_filename, 'rb'))
    result = load_model.predict(X)

    # Reshape result for visualization
    segmented = result.reshape((img_train.shape))

    # Save segmented image as JPEG
    segmented_filename = f'segmented_rock{frame_num:03d}.jpeg'
    plt.imshow(segmented, cmap='jet')
    plt.imsave(segmented_filename, segmented, cmap='jet')

    print(f"Saved segmented image as {segmented_filename}")
