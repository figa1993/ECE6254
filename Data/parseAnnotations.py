# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:14:00 2019

@author: tkrot
"""
import csv
from pathlib import Path
import numpy as np
from PIL import Image

def check_overlap(box1, box2):
    if (box1[0] > box2[2]) or (box2[0] > box1[2]) or (box1[1] > box2[3]) or (box2[1] > box1[3]):
        return False
    else:
        return True
    

data_folder = Path("Data/Vehicules1024/")
output_folder = Path("Data/Output")
annotation_file = data_folder / ("annotation.txt")

# Need (x,y,x,y) to be top left and bottom right, so - +, + - for xy xy
box_dim = [-32, -32, 32, 32]
total_images = 0 
inbound_imageID = []
image_class =[]
image_string = []
selected_v = np.array([])
with open(annotation_file) as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        corners = np.round(np.column_stack((row[1], row[2], row[1], row[2])) + box_dim)
        if corners.min() >= 0 and corners.max() <= 1023:
            print("Object in bounds ", corners)
            inbound_imageID.append(row[0])
            image_string.append(str(int(row[0])).zfill(8))
            image_class.append(row[12])
            if selected_v.shape[0] == 0:
                print("First object")
                selected_v = corners
            else:
                selected_v = np.append(selected_v, corners, axis=0)
        else:
            print("Object out of bounds - ", corners)

        total_images = total_images + 1        
f.close()
prev_imageID = inbound_imageID[0]
j = 1
num_image_per_ID = []
truncated_imageID = [1]
for i in range(1,len(inbound_imageID)):
    if inbound_imageID[i] == prev_imageID:
        j = j+1
    else:
        num_image_per_ID.append(j)
        truncated_imageID.append(inbound_imageID[i])
        j = 1
        prev_imageID = inbound_imageID[i]



# Create bg and remove overlapping images
cumulative_ID = 0
bg_chip = np.array([])
selected_v_post_overlap = np.array([])
inbound_imageID_post_overlap = np.array([])
for i in range(len(num_image_per_ID)):
    print("Starting i", truncated_imageID[i], num_image_per_ID[i], i)
    for j in range(num_image_per_ID[i]):        
        # Create random background for each vehicle
        clean_bg = False
        while (not clean_bg):
            corner = np.random.uniform(65, 900, 2)
            corners = np.round(np.column_stack((corner[0], corner[1], corner[0], corner[1]))+box_dim)
            is_overlap = False
            for vv in range(num_image_per_ID[i]):
                if check_overlap(selected_v[cumulative_ID + vv,:], corners[0,:]):
                    is_overlap = True
            if ( not is_overlap):
                clean_bg = True
        if bg_chip.shape[0] == 0:
            bg_chip = corners
        else:    
            bg_chip = np.append(bg_chip, corners, axis=0)
        
    # Remove overlap vehicles
    overlap_idx = []
    # All of the vehicles in a particular image
    
    intermediary_v = selected_v[cumulative_ID:cumulative_ID+num_image_per_ID[i]]
    intermediary_imageID = inbound_imageID[cumulative_ID:cumulative_ID+num_image_per_ID[i]]
    intermediary_Class = image_class[cumulative_ID:cumulative_ID+num_image_per_ID[i]]
    
    
    if num_image_per_ID[i]>1:
        for v in range(intermediary_v.shape[0]):
            # Remove this vehicle from our list of all vehicles
            other_v = np.delete(intermediary_v, v, axis=0) 
            
            # Compare selected vehicle v to all other in list.
            v_overlap = False
            for vo in range(0, other_v.shape[0]):
                # If overlap, delete selected v. This will happen again when 
                # Selected v = overlapped v so both will be deleted
                if check_overlap(selected_v[cumulative_ID + v,:], other_v[vo,:]):
                    v_overlap = True
            if v_overlap:
                overlap_idx.append(v)
                
    
    # Add happy non overlapping vehicles and continue
    nonOverlapping = np.delete(intermediary_v, overlap_idx, axis=0)
    nonOverlappingID = np.delete(intermediary_imageID, overlap_idx, axis=0)
    nonOverlappingClass = np.delete(intermediary_Class,overlap_idx, axis = 0)
    if selected_v_post_overlap.shape[0] == 0:
        selected_v_post_overlap = nonOverlapping
        inbound_imageID_post_overlap = nonOverlappingID
        image_class_post_overlap = nonOverlappingClass
    else:
        selected_v_post_overlap = np.append(selected_v_post_overlap, nonOverlapping, axis=0)
        inbound_imageID_post_overlap = np.append(inbound_imageID_post_overlap, nonOverlappingID,axis = 0)
        image_class_post_overlap = np.append(image_class_post_overlap, nonOverlappingClass,axis = 0)
    cumulative_ID = cumulative_ID + num_image_per_ID[i]
    
# Just some fun metrics
print("Total Images:" ,total_images)
print("Inbound Images:", len(inbound_imageID))
print("Background Images:", len(bg_chip))
print("Vehicles after overlap deletions:", len(selected_v_post_overlap))

# Plot non overlapping Vehicles
prev_imageID = -1
multi_image_counter = 0
for i in range(len(inbound_imageID_post_overlap)):
    # If multiple images from same original image, give different name to chip
    if inbound_imageID_post_overlap[i] == prev_imageID:
        multi_image_counter = multi_image_counter + 1
    else:
        multi_image_counter = 0
    prev_imageID = inbound_imageID_post_overlap[i]
    
    # Use path to get the input/output folders
    # That string int thing is just turning 59 into 00000059 for the file name
    image_file = data_folder / ((str(int(inbound_imageID_post_overlap[i])).zfill(8)) +"_ir.png")
    output_file = output_folder / ((str(int(inbound_imageID_post_overlap[i])).zfill(8)) + "." + str(multi_image_counter) + "."+ str(int(image_class_post_overlap[i])) + ".png")
    print("Processing:",(str(int(inbound_imageID_post_overlap[i])).zfill(8)),".",multi_image_counter)
    
    # Crop image. CROP fct requires top left and bottom right of the image. 
    # Note that (0,0) is top left, but our annotations have (0,0) as bottom left
    img = Image.open(image_file)
    img2 = img.crop((selected_v_post_overlap[i,:]))
    img2.save(output_file)
    img.close()
    
# Plot backgrounds
for i in range(len(bg_chip)):
    # Note that background was created using the vehicles pre deletion, so 
    # can use the image_string and old variables to parse.
    if inbound_imageID[i] == prev_imageID:
        multi_image_counter = multi_image_counter + 1
    else:
        multi_image_counter = 0
    prev_imageID = inbound_imageID[i]
    
    # Use path to get the input/output folders
    image_file = data_folder / (image_string[i] +"_ir.png")
    output_file = output_folder / (image_string[i] + "." + str(multi_image_counter) + ".bg.png")
    print("Processing Background:",image_string[i],".",multi_image_counter)
    
    # Crop image. CROP fct requires top left and bottom right of the image. 
    # Note that (0,0) is top left, but our annotations have (0,0) as bottom left
    img = Image.open(image_file)
    img2 = img.crop((bg_chip[i,:]))
    img2.save(output_file)