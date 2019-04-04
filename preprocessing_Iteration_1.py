#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 09:28:58 2019

@author: moealani
"""


import glob
import cv2
import os
import numpy as np

#### PART 1: reading images from a folder ##### function to load an image from a folder


## the stored_images_path parameter should be changed to wherever your images are stored

stored_images_path = [img for img in glob.glob("/Users/moealani/PycharmProjects/MSR/data/*.png")]

#preprocessed_images_path is where the images will be written
preprocessed_images_path ="/Users/moealani/PycharmProjects/MSR/data/test2/"

stored_images_path.sort() 


def load_images_from_folder(folder):
    images = []

    for img in folder:

        img = cv2.imread(img,0)

        images.append(img)


    return images # return the collection of images in the folder

#### PART 2: performing histogram equalization ##### 

def contrast_eq(images):
    im_list = []
    for img in images:
        #print(img)
        
        img = img.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        cl1 = clahe.apply(img) 

        
        im_list.append(cl1)
    return im_list




####  PART 3: function to write images back in the same folder ##### 
def write_images_to_folder(path,images):
    
    for i in range(len(images)):
        cv2.imwrite(path+str(i)+'_Preprocessed'+'.png',images[i])


images= load_images_from_folder(stored_images_path) # example of loading images


y = contrast_eq(images)

write_images_to_folder(preprocessed_images_path,y)



