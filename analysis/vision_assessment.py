from interface import VisionInputFrame, VisionRelativeOdometry
import computer_vision as mycv
import numpy as np
import os
import matplotlib.pyplot as plt


with open(r'dataset/vio_dataset_1/left_images.txt') as f:
    image_timestamps = []
    left_images = []
    initial_timestamp = 0
    for i, line in enumerate(f):
        if line.strip() and not line.startswith("#"):  # Skip empty lines and comments
            parts = line.split()
            if len(parts) == 3:  # Ensure the line has the correct format
                left_images.append(os.path.join('dataset/vio_dataset_1', parts[2]))
                if i == 1:
                    initial_timestamp = float(parts[1])  # Store the initial timestamp
                image_timestamps.append(float(parts[1]) - initial_timestamp + 4908.793704125)  # Store the timestamp

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

# Load homogenous ground truth coordinates of the prism
with open(r'dataset/vio_dataset_1/homogenous_ground_truth_converted_by_us.txt') as f:
    ground_truth_transformations = []
    ground_truth_lines = f.readlines()[1:]  # Skip the first line
    idx = 0
    for timestamp in image_timestamps:
        closest_line = None
        closest_time_diff = float('inf')
        for i, line in enumerate(ground_truth_lines[idx:], start=idx):
            if line.strip():  # Skip empty lines
                parts = line.split()
                ground_truth_time = float(parts[0])
                time_diff = abs(ground_truth_time - timestamp)
                if time_diff < closest_time_diff:  # Find the closest timestamp
                    closest_time_diff = time_diff
                    closest_line = parts[1:]
                    idx = i  # Update the index to continue from here
                else:
                    # If time_diff starts increasing, stop searching
                    break
        ground_truth_transformations.append(np.reshape([float(x) for x in closest_line], (4, 4)))



def plot_trajectory(trajectory, ax, color):
    ax.plot([[t[0, 3] for t in trajectory]],
            [[t[1, 3] for t in trajectory]],
            [[t[2, 3] for t in trajectory]],
            label='Estimated trajectory',
            color=color,
            marker='o',)

#Initialize a VisionOdometryCalculator using the first pair of images
#this will be used to track the camera pose in the world frame
initial_frame = VisionInputFrame(left_images[0], right_images[0])

odometry_calculator = mycv.VisionRelativeOdometryCalculator(initial_camera_input=initial_frame,
                                              feature_extractor=mycv.SIFTFeatureExtractor(),
                                              feature_matcher=mycv.FLANNMatcher(),
                                              feature_match_filter=mycv.RANSACFilter())

#Set up arrays to track pose
estimated_transformations = []
estimated_transformations.append(ground_truth_transformations[0])

for i, (left_image, right_image) in enumerate(zip(left_images, right_images)):
    print(i)
    input_frame = VisionInputFrame(left_image, right_image)
    # Calculate the relative odometry
    relative_transformation=odometry_calculator.calculate_relative_odometry_homogenous(input_frame)
    estimated_transformations.append(estimated_transformations[-1] @ relative_transformation)
    

#Plot the estimated v.s. ground truth trajectory
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Camera trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plot_trajectory(ground_truth_transformations, ax, 'k')
plot_trajectory(estimated_transformations, ax, 'r')
ax.legend()
plt.show()