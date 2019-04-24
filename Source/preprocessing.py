import glob
import cv2
import os
import numpy as np


def get_image_filepaths(input_image_dir, glob_pattern ):
    input_images_filepaths = [img for img in glob.glob(os.path.join(input_image_dir, glob_pattern))]

    input_images_filepaths.sort()

    return input_images_filepaths

#### PART 2: performing histogram equalization ##### 

def contrast_eq(img):

    img = img.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    img = clahe.apply(img)
    img = cv2.fastNlMeansDenoising(img, None,h =7,templateWindowSize=21 ,searchWindowSize=7)# remove noise from image 

    return img


####  PART 3: function to write images back in the same folder ##### 
def write_image_to_folder(output_dir,filename,image):
        check = cv2.imwrite(os.path.join(output_dir, filename+'_preprocessed.png'), image)
        return

def preprocess_images( input_image_dir, glob_pattern, output_dir  ):
    image_filepaths = get_image_filepaths( input_image_dir, glob_pattern )
    for image_filepath in image_filepaths:
        image = cv2.imread(image_filepath,0)
        pre_processed_image = contrast_eq(image)
        write_image_to_folder( output_dir, os.path.splitext(os.path.basename(image_filepath))[0],
                              pre_processed_image)

######Script Entry Point##########
if __name__ == '__main__':
    input_image_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),
                                                                                        "../Data/Vehicules1024"+os.sep))

    # preprocessed_images_path is where the images will be written
    preprocessed_images_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),
                                                                                        "../Data/Preprocessed"))

    # load images from the specified directory matching the glob pattern
    images = preprocess_images(input_image_dir, "*.png", preprocessed_images_path)



