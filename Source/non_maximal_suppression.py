# import the necessary packages
import numpy as np


def non_max_suppression(bboxes, confidences=None, overlap_thresh=0.5):
    # if there are no bboxes, return an empty list
    if len(bboxes) == 0:
        return []

    if type(bboxes) is list:
        bboxes = np.array(bboxes)
    # if the bounding bboxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if bboxes.dtype.kind == "i":
        bboxes = bboxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-right y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if confidences is not None:
        idxs = confidences

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

    # return only the bounding bboxes that were picked
    return bboxes[pick].astype("int")

# For testing
if __name__ == "__main__":
    import cv2
    import os
    import time
    from segmentation import do_sliding_window_partition
    im_path =  os.path.abspath(os.path.join(os.path.dirname(os.path.realpath('__file__')),
                                                                            "../Data/Vehicules1024/00000006_ir.png"))

    image = cv2.imread(im_path, 0)
    list_of_partitioned_images, list_of_bbox_verticies = \
        do_sliding_window_partition(image, window_height=64, window_width=64, num_pix_slide=8, num_downscales=0)
    bbox_verticies = list_of_bbox_verticies[0]
    test_detected_list_of_bbox_verticies = [bbox_verticies[0][0], bbox_verticies[0][4], bbox_verticies[0][5], bbox_verticies[1][4]]
    test_scores = [0.5, 0.3, 0.4, 0.3]

    t1 = time.time()
    picked_bbox_vertices = non_max_suppression(test_detected_list_of_bbox_verticies, test_scores)
    t2 = time.time()
    print('time = {0}'.format(str(t2-t1)))
    pass