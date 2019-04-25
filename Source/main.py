import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from preprocessing_Iteration_1 import preprocess_images
from parseAnnotations import parse_annotation_csv
from parseAnnotations import generate_training_images
from feature_extraction import generate_training_feature_vectors
from classification import GridSearch
from classification import SVM
from classification import seq
from classification import process


######Script Entry Point##########
if __name__ == '__main__':

    ###########  Define Directory/File and paths ##########
    # Define where input data resides
    data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')), "Vehicules1024"))

    # Define where input images reside
    input_image_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),
                                                                                        "Vehicules1024"+os.sep))

    # Define where images output from pre-processing algorithm will be written
    preprocessed_images_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),
                                                                                        "../Data/Preprocessed"))
    # Define filepath to image annotations
    annotation_filepath = os.path.join(data_folder, "annotation.txt")
    # Define where training samples will be written
    training_dir = os.path.join(data_folder, "Training")

    training_features = os.path.join(training_dir, "training_features.csv")

    ########## Image PreProcessing ##########
    user_input = input('Run Image Preprocessing (y/n)?')
    if(user_input == "y"):
        # Generate pre-processed images
        preprocess_images(input_image_dir, "*.png", preprocessed_images_path)

    annotations = parse_annotation_csv( annotation_filepath, ' ' )

    ########## Training Sample Generation ##########
    user_input = input('Generate Training Samples (y/n)?')
    if (user_input == "y"):
        generate_training_images(annotations, input_image_dir, training_dir)



    # Define the dimensions of the training images, and window used during vehicle detection
    window_width = 64
    window_height = 64

    ########## Feature Extraction ##########
    user_input = input('Perform Feature Extraction (y/n)?')
    if (user_input == "y"):
        feature_df = generate_training_feature_vectors( training_dir )
    else:
        feature_df = pd.read_csv(training_features)

    user_input = input('Perform Classification (y/n)?')
    if (user_input == "y"):
        data = process()
        c0 =data[0]
        c1=data[1]
        X_train_0, X_test_0, y_train_0, y_test_0 = c0[0],c0[1],c0[2],c0[3]
        X_train_1, X_test_1, y_train_1, y_test_1 = c1[0], c1[1], c1[2], c1[3]



        user_input = input('Perform SVM Classification (y/n)?')

        if (user_input == "y"):
            SVM(X_train_0, X_test_0, y_train_0, y_test_0)


        user_input = input('Perform Grid Search Classification (this can take a while) (y/n)? ')

        if (user_input == "y"):
            GridSearch(X_train_0, X_test_0, y_train_0, y_test_0)

        user_input = input('Perform Sequential Classification (y/n)?')

        if (user_input == "y"):
            seq(X_train_1, X_test_1, y_train_0, y_test_0)



    # For each image in training set
        ##### Image Pre-processing #####
        ##### Feature Extraction #####

    ##### SVM Training #####

    ########## Model Testing ##########
    ##### Image Pre-processing #####
    ##### Feature Extraction #####
    ##### Sliding Window algorithm #####
    # For each window
        ##### Feature Extraction #####
        ##### Classification #####
    ##### Non-maximal Suppression (bounding box generation) #####
    ##### Performance Metric Calculation #####
    print('Complete')