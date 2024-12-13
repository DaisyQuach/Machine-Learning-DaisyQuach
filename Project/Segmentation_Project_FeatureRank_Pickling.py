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


#%%############# Upload TIFF Stack ##############

#(Train / Original Images)
# Load the TIFF stack
input_fileTrain = r'C:\Users\daisy\Desktop\UofU\2024 FALL\CS_6350_MachineLearning\Project\sandstone_data_for_ML\partial_labels_for_traditional_ML\sandstone_train_images.tif'  # Replace with the path to your TIFF stack
output_directoryTrain = r'C:\Users\daisy\Desktop\UofU\2024 FALL\CS_6350_MachineLearning\Project\Data\Train'

# Create the output directory if it doesn't exist
os.makedirs(output_directoryTrain, exist_ok=True)

# Open the TIFF stack
with Image.open(input_fileTrain) as img:
    frame = 0
    while True:
        # Save each frame as an individual TIFF
        output_path = os.path.join(output_directoryTrain, f"frame_{frame:03d}.tiff")
        img.save(output_path)
        print(f"Saved {output_path}")
        frame += 1
        try:
            img.seek(frame)  # Go to the next frame
        except EOFError:
            break  # No more frames in the stack


#(Test / Masked Images)
# Load the TIFF stack
input_fileTest = r'C:\Users\daisy\Desktop\UofU\2024 FALL\CS_6350_MachineLearning\Project\sandstone_data_for_ML\partial_labels_for_traditional_ML\sandstone_partial_labels_from_APEER_ML.tif'
output_directoryTest = r'C:\Users\daisy\Desktop\UofU\2024 FALL\CS_6350_MachineLearning\Project\Data\Test'

# Create the output directory if it doesn't exist
os.makedirs(output_directoryTest, exist_ok=True)

# Open the TIFF stack
with Image.open(input_fileTest) as img:
    frame = 0
    while True:
        # Save each frame as an individual TIFF
        output_path = os.path.join(output_directoryTest, f"frame_{frame:03d}.tiff")
        img.save(output_path)
        print(f"Saved {output_path}")
        frame += 1
        try:
            img.seek(frame)  # Go to the next frame
        except EOFError:
            break  # No more frames in the stack
            
            
print("All frames have been saved.")


#%%############# Import single image ##############
img = cv2.imread(r'C:\Users\daisy\Desktop\UofU\2024 FALL\CS_6350_MachineLearning\Project\Data\Train\frame_000.tiff')                              # As a 2D array
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # In color scale


# Initialize dataframe for image data and filters
df = pd.DataFrame()
img2 = img.reshape(-1)                           # As 1D column array
df['Original Image'] = img2


#%%############# Adding filters ##############
num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
kernels = []
for theta in range(2):   #Define number of thetas
    theta = theta / 4. * np.pi
    for sigma in (1, 3):  #Sigma with 1 and 3
        for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
            for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
            
                
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
#                print(gabor_label)
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
#                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
                    
        
########################################
#Gerate OTHER FEATURES and add them to the data frame
                
    #CANNY EDGE
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe
    
    from skimage.filters import roberts, sobel, scharr, prewitt
    
    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    
    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    
    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1
    
    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    
    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1
    
    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    
    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    
    #VARIANCE with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1  #Add column to original dataframe


labeled_img = cv2.imread(r'C:\Users\daisy\Desktop\UofU\2024 FALL\CS_6350_MachineLearning\Project\Data\Test\frame_000.tiff')
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
labeled_img1 = labeled_img.reshape(-1)
df['Label'] = labeled_img1


#%%########### Training, Predictions, and Accuracy############################
# Dependent variable
Y = df['Label'].values
# Independent variables
X = df.drop(labels = ['Label'], axis=1)

# Split data into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.4, random_state=20)

# Import ML algorithm and train the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, random_state=42)

model.fit(X_train, Y_train)

prediction_test = model.predict(X_test)

from sklearn import metrics
print("Accuracy =", metrics.accuracy_score(Y_test, prediction_test))


#%%########### Feature Ranking ############################
importances = list(model.feature_importances_)

features_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=features_list).sort_values(ascending=False)
print(feature_imp)

#%%########### Pickling Model ############################
import pickle

filename = 'sandstone_model'
pickle.dump(model, open(filename,'wb'))

load_model = pickle.load(open(filename, 'rb'))
result = load_model.predict(X)

segmented = result.reshape((img.shape))

from matplotlib import pyplot as plt
plt.imshow(segmented, cmap='jet')
plt.imsave('segmented_rock.jpg',segmented, cmap='jet')
