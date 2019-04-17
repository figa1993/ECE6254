from skimage.feature import hog
from scipy.misc import imread,imresize
import matplotlib.pyplot as plt

######Script Entry Point##########
if __name__ == '__main__':

    ########## Model Training ##########
    ##### Image Loading #####
    ##### Training set generation #####

    # For each image in training set
        ##### Image Pre-processing #####
        ##### Feature Extraction #####
    image_path = '../Data/Vehicules1024/00000001_ir.png'
    image = imread(image_path )
    image_window = image[99:199,99:199 ] # random block
    fd, hog_image = hog(image_window, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(10, 10), visualize=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(image_window, cmap=plt.cm.gray)
    ax2.imshow(hog_image, cmap=plt.cm.gray)
    plt.show()

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