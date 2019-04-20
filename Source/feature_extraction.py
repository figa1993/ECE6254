import os
import glob
from scipy.misc import imread,imresize
import glob
from skimage.feature import hog
import pandas as pd
import numpy as np
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from skimage import feature

def generate_training_feature_vectors( input_dir ):
    # Get a list of all the training image filepaths for images containing a vehicle
    vehicle_im_filepaths = [img for img in glob.glob(os.path.join(input_dir, "*.v.png"))]
    # Get a list of all the training image filepaths for images NOT containing a vehicle
    background_im_filepaths = [img for img in glob.glob(os.path.join(input_dir, "*.bg.png"))]
    # Create a list of all the image filepaths
    filepaths = vehicle_im_filepaths + background_im_filepaths
    # Create a corresponding list of labels
    labels = ['vehicle'] * len(vehicle_im_filepaths) + ['background'] * len(background_im_filepaths)

    num_images = len(filepaths)

    # Define tuning parameters for the HOG algorithm
    num_cells = 8 # number of cells to divide the block (window of overall image) along each direction (assumes square image input)
    num_orientations = 8 #number of orientations into which to bucket the gradient

    # Define parameters for the Gabor feature generation
    # Spatial frequencies in pixels for the Gabor filters
    frequencies = [0.125, 0.25, 0.5, 1, 2, 4, 8] # Guesses
    # Widths of the Gaussian Kernels in the Gabor Filter
    sigmas = [1,2,3] # Guesses

    hog_headers = generate_HOG_feature_headers(num_cells, num_orientations)

    df = pd.DataFrame(
        {
            'filepath': filepaths,
            'label': labels,
        }
    )

    hog_feature_mtx = np.array([])
    gabor_feature_mtx = np.array([])
    lbp_feature_mtx = np.array([])

    for i, image_path in enumerate(filepaths):
        image = imread(image_path)
        hog_feature_vec  = generate_HOG_features(image, num_cells, num_orientations)
        gabor_feature_vec = generate_gabor_features(image, num_orientations, frequencies, sigmas )
        lbp_feature_vec = generate_lbp_features(image, 24 ,8) # 24 and 8 values taken from the cited example
        # Append the HOG features for this image to HOG feature matrix
        if hog_feature_mtx.shape[0]>0:
            hog_feature_mtx = np.vstack((hog_feature_mtx,hog_feature_vec))
        else:
            hog_feature_mtx = hog_feature_vec
        # Append the gabor features for this image to gabor feature matrix
        if gabor_feature_mtx.shape[0]>0:
            gabor_feature_mtx = np.vstack((gabor_feature_mtx, gabor_feature_vec))
        else:
            gabor_feature_mtx = gabor_feature_vec
        # Append the lbp features for this image to lbp feature matrix
        if lbp_feature_mtx.shape[0]>0:
            lbp_feature_mtx = np.vstack((lbp_feature_mtx, lbp_feature_vec))
        else:
            lbp_feature_mtx = lbp_feature_vec
        print(i, 'features out of ', num_images,' generated')

    hog_headers = generate_HOG_feature_headers(num_cells, num_orientations)
    gabor_headers = generate_gabor_headers(num_orientations,frequencies, sigmas)
    lbp_headers = generate_lbp_headers(24, 8)

    for i,header in enumerate(hog_headers):
        df[header] = pd.Series( hog_feature_mtx[:,i], index=df.index )

    for i,header in enumerate(gabor_headers):
        df[header] = pd.Series( gabor_feature_mtx[:,i], index=df.index )

    for i,header in enumerate(lbp_headers):
        df[header] = pd.Series( lbp_feature_mtx[:,i], index=df.index )

    # Write the feature vectors to file, so they do not have to be recalculated each time
    output_filepath = os.path.join(input_dir, "training_features.csv")
    with open(output_filepath, 'w') as output_file:
        df.to_csv(output_file)

    return df # return the dataframe


def generate_HOG_features( image, num_cells, num_orientations ):
    pixels_per_block_dim = int(image.shape[0] / num_cells)
    fd = hog(image, orientations=num_orientations, pixels_per_cell=(pixels_per_block_dim, pixels_per_block_dim),
                     cells_per_block=(num_cells, num_cells), feature_vector=True)

    return fd #return the vector of HOG features

def generate_HOG_feature_headers( num_cells, num_orientations):
    feature_headers = [] #initialize an empty list into which string descriptor for each feature will go
    grad_step_size_degrees = 180 / num_orientations # defines step size from 0 to 180 of angle at which grad is binned
    for i in range(0,num_cells):
        row_str = 'block_'+i.__str__()
        for j in range(0,num_cells):
            col_str = row_str + '_' + j.__str__()
            for k in range(0,num_orientations):
                feature_headers.append(col_str + '_' + (k*grad_step_size_degrees).__str__() )

    return feature_headers

def generate_gabor_features( image, num_orientations, frequencies, sigmas ):
    # define number of orientations for which gabor filters are generated (since symmetric only through pi)
    theta_step_size_radians = np.pi / num_orientations
    feature_vec = np.empty(num_orientations*len(frequencies)*len(sigmas)*2) #multiplied by 2 since 2 features per kernel
    iter = 0
    for i in range(num_orientations):
        theta = i*theta_step_size_radians
        for sigma in sigmas:
            for frequency in frequencies:
                # Generate the Gabor Kernel
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                # convolve image with the kernel
                filtered_image = ndi.convolve(image, kernel, mode = 'wrap')
                feature_vec[iter]= filtered_image.mean()
                iter = iter + 1
                feature_vec[iter] = filtered_image.var()
                iter = iter + 1
    return feature_vec

def generate_gabor_headers(num_orientations, frequencies, sigmas):
    # define number of orientations for which gabor filters are generated (since symmetric only through pi)
    theta_step_size_radians = np.pi / num_orientations
    gabor_headers = []
    for i in range(num_orientations):
        theta = i*theta_step_size_radians
        for sigma in sigmas:
            for frequency in frequencies:
                gabor_headers.append('gabor_'+theta.__str__()+'_'+sigma.__str__() + '_' + frequency.__str__()+'_mean' )
                gabor_headers.append('gabor_' + theta.__str__() + '_' + sigma.__str__() + '_' +
                                     frequency.__str__()+'_var')

    return gabor_headers

# LBP feature generation adapted from https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
def generate_lbp_features(image, num_points, radius):
    lbp = feature.local_binary_pattern(image, num_points, radius, method = 'uniform')
    (feature_vec, unused )= np.histogram(lbp.ravel(), bins = np.arange(0,num_points+3), range=(0, num_points+2) )
    #normalize feature vector
    feature_vec = feature_vec.astype("float")
    feature_vec /= feature_vec.sum()

    return feature_vec

def generate_lbp_headers(num_points, radius):
    lbp_headers = []
    for i in range(0, num_points+2):
        lbp_headers.append('num_lbp_pattern_'+i.__str__())
    return lbp_headers