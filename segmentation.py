# Segmentation Module
# Provides sliding window and image pyramid with gaussian blur

from skimage import data
from skimage.util import view_as_windows
from skimage.transform import pyramid_gaussian


def run_sliding_window(input_image, num_pix_slide, window_height_px, window_width_px):
    """
    Run sliding window on imput image with desired settings
    :param input_image: Image to be partitioned
    :param num_pix_slide: Number of pixels to slide the window
    :param window_height_px: Number of pixels used in window height (rows in image)
    :param window_width_px: Number of pixels used in window width (columns in image)
    :return: An array of image partitions
    """
    window_size = [window_height_px, window_width_px, 3]
    partitioned_image = view_as_windows(input_image, window_size, num_pix_slide)
    return partitioned_image


def run(input_image, window_height=4, window_width=4, num_pix_slide=1, num_downscales=3):
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
    image_pyramid = pyramid_gaussian(input_image, downscale=2, multichannel=True)
    while num_downscales >= 0:
        downscaled_image = next(image_pyramid)
        out.append(run_sliding_window(downscaled_image, num_pix_slide, window_height, window_width))
        num_downscales -= 1
    return list_of_partitioned_images


# For testing
if __name__ == "__main__":
    import time
    t1 = time.time()
    image = data.astronaut()
    out = run(image, 4, 4, 1, 3)

    t2 = time.time()
    print('time = {0}'.format(str(t2-t1)))
