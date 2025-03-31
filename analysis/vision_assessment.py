from interface import VisionInputFrame, VisionRelativeOdometry
import computer_vision as mycv
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp


# Read ground truth file and get the first timestamp
with open(r'dataset/vio_dataset_1/homogenous_ground_truth_converted_by_us.txt') as f:
    ground_truth_lines = f.readlines()[1:]  # Skip header line
if not ground_truth_lines or not ground_truth_lines[0].strip():
    raise ValueError("Ground truth file is empty or invalid.")
first_gt_time = float(ground_truth_lines[0].split()[0])


with open(r'dataset/vio_dataset_1/left_images.txt') as f:
    image_timestamps = []
    left_images = []
    for i, line in enumerate(f):
        if line.strip() and not line.startswith("#"):  # Skip empty lines and comments
            parts = line.split()
            if len(parts) == 3:  # Ensure the line has the correct format
                left_images.append(os.path.join('dataset/vio_dataset_1', parts[2]))
                image_timestamps.append(float(parts[1]))  # Store the timestamp

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

first_valid_index = next(i for i, ts in enumerate(image_timestamps) if ts >= first_gt_time)
image_timestamps = image_timestamps[first_valid_index:]
left_images = left_images[first_valid_index:]
right_images = right_images[first_valid_index:]

# Load homogenous ground truth coordinates of the prism
ground_truth_transformations = []
idx = 0
for timestamp in image_timestamps:
    closest_line = None
    closest_time_diff = float('inf')
    for i, line in enumerate(ground_truth_lines[idx:], start=idx):
        if line.strip():
            parts = line.split()
            ground_truth_time = float(parts[0])
            time_diff = abs(ground_truth_time - timestamp)
            if time_diff < closest_time_diff:
                closest_time_diff = time_diff
                closest_line = parts[1:]
                idx = i  # update starting index
            else:
                # When the time difference increases, assume we've passed the closest match.
                break
    if closest_line is None:
        raise ValueError(f"No ground truth transformation found for timestamp {timestamp}")
    ground_truth_transformations.append(np.reshape([float(x) for x in closest_line], (4, 4)))

def plot_trajectory(estimate, ground, ax):
    ax.plot([t[0, 3] for t in ground],  # X coordinates
            [t[1, 3] for t in ground],  # Y coordinates
            [t[2, 3] for t in ground],  # Z coordinates
            label='Ground',
            color='k',
            marker='o')
    ax.plot([t[0, 3] for t in estimate],  # X coordinates
            [t[1, 3] for t in estimate],  # Y coordinates
            [t[2, 3] for t in estimate],  # Z coordinates
            label='Estimated',
            color='r',
            marker='o')

#Initialize a VisionOdometryCalculator using the first pair of images
#this will be used to track the camera pose in the world frame
initial_frame = VisionInputFrame(left_images[0], right_images[0])

odometry_calculator = mycv.VisionRelativeOdometryCalculator(initial_camera_input=initial_frame,
                                              feature_extractor=mycv.SIFTFeatureExtractor(),
                                              feature_matcher=mycv.FLANNMatcher(),
                                              feature_match_filter=mycv.RANSACFilter(min_matches=12, reproj_thresh=1))

#Set up arrays to track pose
estimated_transformations = []
estimated_transformations.append(ground_truth_transformations[0])

counter = 0
max = len(ground_truth_transformations)
limit = 500
visualize = False


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

T_cam02imu0=np.array([[-0.02822879, 0.01440125, 0.99949774, 0.00110212],
            [-0.99960149, -0.00041887, -0.02822568, 0.02170142],
            [ 0.00001218, -0.99989621, 0.01440734, -0.00005928],
            [ 0., 0., 0., 1. ]])

T_imu2refl = np.diag([1, -1, -1, 1]) #invert the transformation to get the camera to imu transformation
T_cam2drone = np.array([[0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]]) @ T_cam02imu0 @ np.diag([1, -1, -1, 1]) #invert the transformation to get the camera to imu transformation


# Filter parameter (0 < alpha <= 1; lower values smooth more)
alpha_t = 0.2
alpha_R = 0.2
filtered_R = np.eye(3)
filtered_t = np.zeros(3)
for i, (left_image, right_image) in enumerate(zip(left_images, right_images)):
    if counter > limit:
        break
    counter += 1
    print(i)
    
    input_frame = VisionInputFrame(left_image, right_image)

    # Calculate the relative odometry between the previous and new input frame
    relative_transformation, points_prev, points_cur, points_left, points_right = \
        odometry_calculator.calculate_relative_odometry_homogenous(input_frame)
    
    #Switch the basis of the relative transformation to the drone frame
    relative_transformation = (
        T_cam2drone @
        relative_transformation @ 
        np.linalg.inv(T_cam2drone)
    )

    # Decompose the relative transformation into rotation (R_new) and translation (t_new)
    R_new = relative_transformation[:3, :3]
    t_new = relative_transformation[:3, 3]

    if i == 0:
        filtered_R = R_new.copy()
        filtered_t = t_new.copy()
    else:        
        # Filter the translation
        filtered_t = alpha_t * t_new + (1 - alpha_t) * filtered_t
        
        # Filter the rotation using quaternions
        key_times = [0, 1]
        key_rots = Rotation.from_matrix([filtered_R, R_new])
        slerp = Slerp(key_times, key_rots)
        rot_filtered = slerp(alpha_R)
        filtered_R = rot_filtered.as_matrix()

    # Recompose the filtered relative transformation
    filtered_relative_transformation = np.eye(4)
    filtered_relative_transformation[:3, :3] = filtered_R
    filtered_relative_transformation[:3, 3] = filtered_t

    # Apply the new transformation to the previous one to get the new world position
    estimated_transformations.append(estimated_transformations[-1] @ filtered_relative_transformation)
    #note that the above is still not quite correct I think, but it is a start               

    # Plot the difference between the point clouds for this step
    # odometry_calculator.plot_point_clouds(points_prev, points_cur)
    if visualize:
        image_l, image_r = mycv.load_images(input_frame)
        odometry_calculator.StereoPair.plot_undistorted_points(points_left, points_right, image_l, image_r, axes, animate=True)
if visualize:
    plt.show()  

#Plot the estimated v.s. ground truth trajectory
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Camera trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#Extract just translation
est_positions = np.array([t[:3, 3] for t in estimated_transformations])

# Compute the min, max, midpoint, and range along each axis
mins = est_positions.min(axis=0)
maxs = est_positions.max(axis=0)
mid = (mins + maxs) / 2
max_range = (maxs - mins).max()

# Set the axis limits uniformly so the plot is centered and scaled equally
ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

# Plot the trajectories (assuming plot_trajectory is defined)
plot_trajectory(estimated_transformations, ground_truth_transformations[:limit], ax)
ax.legend()
plt.show()