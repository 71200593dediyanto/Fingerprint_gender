# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:22:55 2024

@author: DEDI
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
import PIL
import fingerprint_enhancer
import numpy as np
import os
import multiprocessing
import cv2
from localbinarypatterns import LocalBinaryPatterns
from ridgedensity import RidgeDensityCalculator
from sklearn.svm import LinearSVC
from skimage import io, color, filters
from skimage.morphology import skeletonize
from skimage.color import rgba2rgb
import pandas as pd
from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import utils



cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

def minutiae_at(pixels, i, j):
    values = [pixels[i + k][j + l] for k, l in cells]

    crossings = 0
    for k in range(0, 8):
        crossings += abs(values[k] - values[k + 1])
    crossings /= 2

    if pixels[i][j] == 1:
        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"
    return "none"

def calculate_minutiaes(im):
    pixels = utils.load_image(im)
    utils.apply_to_each_pixel(pixels, lambda x: 0.0 if x > 10 else 1.0)

    (x, y) = im.size
    result = im.convert("RGB")

    draw = ImageDraw.Draw(result)
    minutea_feature = {"ending":0,"bifurcation":0}
    colors = {"ending" : (150, 0, 0), "bifurcation" : (0, 150, 0)}

    ellipse_size = 2
    for i in range(1, x - 1):
        for j in range(1, y - 1):
            minutiae = minutiae_at(pixels, i, j)
            if minutiae != "none":
               if minutiae == 'bifurcation':
                    minutea_feature['bifurcation'] += 1
               else:
                    minutea_feature['ending'] += 1
                #result[i - ellipse_size:i + ellipse_size, j - ellipse_size:j + ellipse_size] = colors[minutiae]

    del draw

    return minutea_feature



def convert_to_grayscale_and_enchance_image(image_path):

    img = io.imread(image_path)
    
    if img.ndim == 3 and img.shape[2] == 3:  # RGB image
        gray_img = color.rgb2gray(img)
    elif img.ndim == 3 and img.shape[2] == 4:  # RGBA image
        gray_img = color.rgb2gray(color.rgba2rgb(img))
    elif img.ndim == 2:  # Grayscale image
        gray_img = img
    else:
        raise ValueError("Unsupported image format")
  
    ## Resize the image
    scaling_factor = 2
    resized_img = cv2.resize(gray_img, None, fx=scaling_factor, fy=scaling_factor)

    # Apply local thresholding
    local_binary = filters.threshold_local(resized_img, block_size=55, method='gaussian')

    # Binarize the image using local threshold
    binarized_img = resized_img > local_binary

    # Enhance image
    out = fingerprint_enhancer.enhance_Fingerprint(binarized_img)
    out[out == 255] = 1  # Replace 255 with 1
    skeleton = skeletonize(out)
    inverted_skeleton = np.invert(skeleton)
    
    
    return inverted_skeleton


def image_preprocesing_for_LBP(image_path):
    img = io.imread(image_path)
    
    if img.ndim == 3 and img.shape[2] == 3:  # RGB image
        gray_img = color.rgb2gray(img)
    elif img.ndim == 3 and img.shape[2] == 4:  # RGBA image
        gray_img = color.rgb2gray(color.rgba2rgb(img))
    elif img.ndim == 2:  # Grayscale image
        gray_img = img
    else:
        raise ValueError("Unsupported image format")
    
    ## Resize the image
    scaling_factor = 6
    resized_img = cv2.resize(gray_img, None, fx=scaling_factor, fy=scaling_factor)
    
    # Apply noise reduction using Gaussian filter
    blurred_img = filters.gaussian(resized_img, sigma=1)

    # Apply histogram equalization for contrast enhancement
    equalized_img = exposure.equalize_hist(blurred_img)
    
    # Apply local thresholding
    local_binary = filters.threshold_local(equalized_img, block_size=55, method='gaussian')

    # Binarize the image using local threshold
    binarized_img = resized_img > local_binary
   

    
    return binarized_img

def get_label_from_filename(file_path):
    # Split the file path based on the directory separator
    parts = file_path.split('\\')

    # Get the last part containing the filename
    filename = parts[-1]

    # Split the filename based on double underscore
    label = filename.split('__')[1][0]
    return 0 if label == "F" else 1


def store_LBP_features(arr_file):
    array_lbp = []
    
    for i in range(0,len(arr_file)):
        desc = LocalBinaryPatterns(24,8)
        histogram = desc.describe(arr_file[i])
        array_lbp.append(histogram)
        
    return array_lbp

def store_minutae_features(arr_file):
    array_minutae = []
    
    for i in range(0,len(arr_file)):
        image_path = arr_file[i].replace("\\", "/")
        im = Image.open(image_path)
        im_gray = im.convert("L")
        minutae_feature = calculate_minutiaes(im_gray)
        temp = []
        temp.append(minutae_feature['ending'])
        temp.append(minutae_feature['bifurcation'])
        array_minutae.append(temp)
    return array_minutae

def store_ridge_density_features(arr_file):
    array_ridge_density = []
    
    
    for i in range(0,len(arr_file)):
        convert_to_int = (1 - arr_file[i]).astype(np.uint8) * 255
        area = np.prod(arr_file[i].shape)
        ridge_count = sum(255 for row in convert_to_int for pixel in row if pixel == 255)
        ridge_density = RidgeDensityCalculator(ridge_count, area)
        count_ridge_density = ridge_density.calculate_ridge_density()
        array_ridge_density.append(count_ridge_density)
    return array_ridge_density

def store_hand(arr_file):
    temp_array = []
    for i in arr_file:
        path = os.path.basename(i)
        file = path.split("_")
        array =[]
        location = file[3]
        
        array.append(file[0])

        if location == "thumb":
            array.append(0)
            if file[2] == "Left":
                array.append(0)
            elif file[2] == "Right":
                array.append(1)
        elif location == "middle":
            array.append(1)
            if file[2] == "Left":
                array.append(0)
            elif file[2] == "Right":
                array.append(1)
        elif location == "ring":
            array.append(2)
            if file[2] == "Left":
                array.append(0)
            elif file[2] == "Right":
                array.append(1)
        elif location == "index":
            array.append(3)
            if file[2] == "Left":
                array.append(0)
            elif file[2] == "Right":
                array.append(1)
        else:
            array.append(4)
            if file[2] == "Left":
                array.append(0)
            elif file[2] == "Right":
                array.append(1)
        temp_array.append(array)
    return temp_array


if __name__ == '__main__':
    
    # Directory containing your fingerprint images
   #images_directory = r'D:\SEMESETER 8\ngoding\thumb'
   images_directory = r'D:\SEMESETER 8\SKRIPSI\dataset\Real'
   enhance_directory = r'D:\SEMESETER 8\ngoding\enhance_image'
   enhance_directory_test = r'D:\SEMESETER 8\ngoding\enhance_image_test'
   

   # List all image files in the directory
   image_files = sorted([os.path.join(images_directory, f) for f in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, f))])
   
   
   images_test_directory = r'D:\SEMESETER 8\ngoding\test'
   
   image_test = [os.path.join(images_test_directory, f) for f in os.listdir(images_test_directory) if os.path.isfile(os.path.join(images_test_directory, f))]
   
   enhance_files = [os.path.join(enhance_directory, f) for f in os.listdir(enhance_directory) if os.path.isfile(os.path.join(enhance_directory, f))]
   enhance_files_test =[os.path.join(enhance_directory_test, f) for f in os.listdir(enhance_directory_test) if os.path.isfile(os.path.join(enhance_directory_test, f))]

   # List all image files in the directory
   image_labels = {filename: get_label_from_filename(filename) for filename in image_files}
   labels = [label for label in image_labels.values()]
   
   test_label = {filename: get_label_from_filename(filename) for filename in image_test}
   test_labels = [label for label in test_label.values()]
   
   
   num_processes = multiprocessing.cpu_count()

   
   pool = multiprocessing.Pool(processes=num_processes)
   
   images_for_LBP = pool.map(image_preprocesing_for_LBP,image_files)

   
   images_for_ridge_density = pool.map(convert_to_grayscale_and_enchance_image, image_files)
   
   
   images_for_LBP_test = pool.map(image_preprocesing_for_LBP, image_test)
   images_for_ridge_density_test = pool.map(convert_to_grayscale_and_enchance_image, image_test)

   # Close the pool to release resources
   pool.close()
   pool.join()
   
   arr_lbp_features = store_LBP_features(images_for_LBP)
   arr_ridge_density_features = store_ridge_density_features(images_for_ridge_density)
   arr_minutae_features = store_minutae_features(enhance_files)
   arr_hand_location = store_hand(image_files)
   
   
   arr_lbp_features_test = store_LBP_features(images_for_LBP_test)
   arr_ridge_density_features_test = store_ridge_density_features(images_for_ridge_density_test)
   arr_minutae_features_test = store_minutae_features(enhance_files_test)
   
   
   
   #training
   # Convert lists to numpy arrays
   X_lbp = np.array(arr_lbp_features)
   X_ridge_density = np.array(arr_ridge_density_features)
   X_minutae_feature = np.array(arr_minutae_features)
   X_array_hand_location = np.array(arr_hand_location)
   y = np.array(labels)
   
   vector_features = []
   
   for i in range(0,len(X_lbp)):
       #temp_array = np.concatenate((X_lbp[i], [X_ridge_density[i]]))
       temp_array = np.concatenate((X_lbp[i], [X_ridge_density[i]],X_array_hand_location[i]))
       vector_features.append(temp_array)
       
       


   # Concatenate features
   vector_features = np.array(vector_features)
   X_train, X_test, y_train, y_test = train_test_split(vector_features, labels, test_size=0.1, random_state=100)
   X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.1, random_state=100)
   # Initialize the SVM classifier
   # Train SVM models with different kernels
# Training SVM with linear kernel
   print("Training SVM with linear kernel...")
   svm_linear = SVC(kernel='linear',C=100,gamma=40)
   svm_linear.fit(X_train, y_train)

    # Evaluate the linear SVM model
   train_accuracy_linear = svm_linear.score(X_train, y_train) * 100
   val_accuracy_linear = svm_linear.score(X_val, y_val) * 100
   test_accuracy_linear = svm_linear.score(X_test, y_test) * 100

   print(f"Train Accuracy (linear): {train_accuracy_linear:.2f}%")
   print(f"Validation Accuracy (linear): {val_accuracy_linear:.2f}%")
   print(f"Test Accuracy (linear): {test_accuracy_linear:.2f}%")
   print()

# Training SVM with polynomial kernel
   print("Training SVM with polynomial kernel...")
   svm_poly = SVC(kernel='poly')
   svm_poly.fit(X_train, y_train)

# Evaluate the polynomial SVM model
   train_accuracy_poly = svm_poly.score(X_train, y_train) * 100
   val_accuracy_poly = svm_poly.score(X_val, y_val) * 100
   test_accuracy_poly = svm_poly.score(X_test, y_test) * 100

   print(f"Train Accuracy (polynomial): {train_accuracy_poly:.2f}%")
   print(f"Validation Accuracy (polynomial): {val_accuracy_poly:.2f}%")
   print(f"Test Accuracy (polynomial): {test_accuracy_poly:.2f}%")
   print()

# Training SVM with RBF kernel
   print("Training SVM with RBF kernel...")
   svm_rbf = SVC(kernel='rbf')
   svm_rbf.fit(X_train, y_train)

# Evaluate the RBF SVM model
   train_accuracy_rbf = svm_rbf.score(X_train, y_train) * 100
   val_accuracy_rbf = svm_rbf.score(X_val, y_val) * 100
   test_accuracy_rbf = svm_rbf.score(X_test, y_test) * 100

   print(f"Train Accuracy (RBF): {train_accuracy_rbf:.2f}%")
   print(f"Validation Accuracy (RBF): {val_accuracy_rbf:.2f}%")
   print(f"Test Accuracy (RBF): {test_accuracy_rbf:.2f}%")
   print()
       
   X_lbp_test = np.array(arr_lbp_features_test)
   X_ridge_density_test = np.array(arr_ridge_density_features_test)
   X_minutae_feature_test = np.array(arr_minutae_features_test)
   y_test = np.array(test_labels)
   
  
    
   
   
   
   
    

        