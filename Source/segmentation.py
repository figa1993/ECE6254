# Segmentation Module
# Provides sliding window and image pyramid with gaussian blur

from skimage.util import view_as_windows
from skimage.transform import pyramid_gaussian
import numpy as np
import cv2
import os
from skimage import data
import time

def run_sliding_window(input_image, num_pix_slide, window_height_px, window_width_px):
    """
    Run sliding window on imput image with desired settings
    :param input_image: Image to be partitioned
    :param num_pix_slide: Number of pixels to slide the window
    :param window_height_px: Number of pixels used in window height (rows in image)
    :param window_width_px: Number of pixels used in window width (columns in image)
    :return: An array of image partitions
    """
    num_channels = input_image.shape[2]
    window_size = [window_height_px, window_width_px, num_channels]
    partitioned_image = np.squeeze(view_as_windows(input_image, window_size, num_pix_slide))
    return partitioned_image


def do_sliding_window_partition(input_image, window_height=4, window_width=4, num_pix_slide=1, num_downscales=3):
    """
    Main function to run segmentation algorithm
    :param input_image: Image to be partitioned
    :param window_height: Number of pixels used in window width (columns in image)
    :param window_width:  Number of pixels used in window height (rows in image)
    :param num_pix_slide: Number of pixels to slide the window
    :param num_downscales: Number of times to downscale the image (will get num_downscales + 1 images returned)
    :return: list of all partitioned images.
    """
    list_of_partitioned_images = []
    if num_downscales > 0:
        image_pyramid = pyramid_gaussian(input_image, downscale=2, multichannel=True)
        while num_downscales >= 0:
            downscaled_image = next(image_pyramid)
            list_of_partitioned_images.append(run_sliding_window(downscaled_image, num_pix_slide, window_height,
                                                                 window_width))
            num_downscales -= 1
    else:
        list_of_partitioned_images.append(run_sliding_window(input_image, num_pix_slide, window_height,
                                                             window_width))
    return list_of_partitioned_images


# For testing
if __name__ == "__main__":
    im_path =  os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),
                                                                            "../Data/Vehicules1024/00000004_co.png"))

    image = cv2.imread(im_path)
    t1 = time.time()
    out = do_sliding_window_partition(image, window_height=64, window_width=64, num_pix_slide=8, num_downscales=0)
    cv2.imshow('orig', image)
    cv2.imshow('image', out[0][0,0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    t2 = time.time()
    print('time = {0}'.format(str(t2-t1)))
