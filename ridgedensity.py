# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:00:34 2024

@author: DEDI
"""



#ROI do the center fingerprint
#extract the ridge density


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
import fingerprint_enhancer
import numpy as np
import os
import multiprocessing
import cv2
from skimage import io, color, filters
from skimage.morphology import skeletonize




class RidgeDensityCalculator:
    def __init__(self,ridge_count,area):
        self.ridge_count = ridge_count
        self.area = area

    def calculate_ridge_density(self):
        return self.ridge_count / self.area