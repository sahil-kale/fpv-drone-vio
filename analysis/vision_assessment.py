from interface import VisionInputFrame, VisionRelativeOdometry
import computer_vision as mycv
import numpy as np
import os

with open(r'dataset/vio_dataset_1/left_images.txt') as f:
    left_images = []
    for line in f:
        if line.strip() and not line.startswith("#"):  # Skip empty lines and comments
            parts = line.split()
            if len(parts) == 3:  # Ensure the line has the correct format
                left_images.append(os.path.join('dataset/vio_dataset_1', parts[2]))

with open(r'dataset/vio_dataset_1/right_images.txt') as f:
    right_images = []
    for line in f:
        if line.strip() and not line.startswith("#"):  # Skip empty lines and comments
            parts = line.split()
            if len(parts) == 3:  # Ensure the line has the correct format
                right_images.append(os.path.join('dataset/vio_dataset_1', parts[2]))

#Check that the arrays are the same length
if len(left_images) != len(right_images):
    raise ValueError("The number of left and right images do not match.")


#Initialize a VisionOdometryCalculator using the first pair of images
#this will be used to track the camera pose in the world frame
initial_frame = VisionInputFrame(left_images[0], right_images[0])
frame = mycv.VisionRelativeOdometryCalculator(initial_camera_input=initial_frame,
                                              feature_extractor=mycv.SIFTFeatureExtractor(),
                                              feature_matcher=mycv.FLANNMatcher(),
                                              feature_match_filter=mycv.RANSACFilter())

for left_image, right_image in zip(left_images[1:], right_images[1:]):
    input_