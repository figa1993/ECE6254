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

    window_size = [window_height_px, window_width_px]
    partitioned_image = np.squeeze(view_as_windows(input_image, window_size, num_pix_slide))
    bboxes_vertices = []

    # Iterate over rows/cols and get idx of top left corner and bottom right corner of bounding box.  Lists of lists are used for speed over numpy array
    for row_idx, row in enumerate(partitioned_image):
        this_col = []
        for column_idx, element in enumerate(row):
            top_left_x = column_idx * num_pix_slide
            top_left_y = row_idx * num_pix_slide

            bottom_right_x = column_idx * num_pix_slide + window_width_px - 1
            bottom_right_y = row_idx * num_pix_slide + window_height_px - 1
            this_col.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
        bboxes_vertices.append(this_col)

    # Check that number of bounding boxes is consistent with number of partitions
    assert (len(bboxes_vertices), len(bboxes_vertices[0])) == partitioned_image.shape[:2], "Number of bounding boxes does not equal the number of image partitions!"

    return partitioned_image, bboxes_vertices


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
    list_of_bbox_verticies = []
    if num_downscales > 0:
        image_pyramid = pyramid_gaussian(input_image, downscale=2, multichannel=True)
        while num_downscales >= 0:
            downscaled_image = next(image_pyramid)
            partitioned_image, bboxes_vertices = run_sliding_window(downscaled_image, num_pix_slide, window_height,
                                                                    window_width)
            list_of_partitioned_images.append(partitioned_image)
            list_of_bbox_verticies.append(bboxes_vertices)
            num_downscales -= 1
    else:
        partitioned_image, bboxes_vertices = run_sliding_window(input_image, num_pix_slide, window_height,
                                                             window_width)
        list_of_partitioned_images.append(partitioned_image)
        list_of_bbox_verticies.append(bboxes_vertices)
    return list_of_partitioned_images, list_of_bbox_verticies


# For testing
if __name__ == "__main__":
    im_path =  os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),
                                                                            "../Data/Vehicules1024/00000006_ir.png"))

    image = cv2.imread(im_path, 0)
    t1 = time.time()
    list_of_partitioned_images, list_of_bbox_verticies = \
        do_sliding_window_partition(image, window_height=64, window_width=64, num_pix_slide=8, num_downscales=0)
    # cv2.imshow('orig', image)
    # cv2.imshow('image', out[0][0,0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    t2 = time.time()
    print('time = {0}'.format(str(t2-t1)))
