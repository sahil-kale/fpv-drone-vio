from interface import VisionInputFrame, VisionRelativeOdometry
import computer_vision as mycv
import cv2
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

def plot_trajectory(trajectory, ax, color='r', triad_scale=0.1, show_triads=False):
    # Plot the trajectory line
    ax.plot([t[0, 3] for t in trajectory],  # X coordinates
            [t[1, 3] for t in trajectory],  # Y coordinates
            [t[2, 3] for t in trajectory],  # Z coordinates
            color=color,
            marker='o')
    
    if show_triads:
        # Plot orientation triads at each position
        for T in trajectory:
            # Extract position
            pos = T[0:3, 3]
            
            # Extract rotation matrix (each column is a direction vector)
            R = T[0:3, 0:3]
            
            # Draw x-axis (red)
            ax.quiver(pos[0], pos[1], pos[2],
                    R[0, 0], R[1, 0], R[2, 0],
                    color='r', length=triad_scale)
            
            # Draw y-axis (blue)
            ax.quiver(pos[0], pos[1], pos[2],
                    R[0, 1], R[1, 1], R[2, 1],
                    color='b', length=triad_scale)
            
            # Draw z-axis (black)
            ax.quiver(pos[0], pos[1], pos[2],
                    R[0, 2], R[1, 2], R[2, 2],
                    color='k', length=triad_scale)

#Function to evaluate the error in the delta estimations
def get_delta_residuals(estimated_trajectory, ground_truth_trajectory):
    estimated_deltas = []
    ground_truth_deltas = []
    for i in range(1, len(estimated_trajectory)):
        estimated_deltas.append(np.linalg.inv(estimated_trajectory[i-1]) @ estimated_trajectory[i])
        ground_truth_deltas.append(np.linalg.inv(ground_truth_trajectory[i-1]) @ ground_truth_trajectory[i])
    
    # The deltas give a position change vector and a rotation matrix 
    # We can find the difference between estimated and ground truth deltas
    # Error in the position change is given by a vector
    # Error in the rotation is given by a rotation matrix which we will convert to a rodrigues vector
    estimated_deltas = np.array(estimated_deltas)
    ground_truth_deltas = np.array(ground_truth_deltas)
    position_residuals = estimated_deltas[:, :3, 3] - ground_truth_deltas[:, :3, 3] #Subtract t - t_gt
    rotation_residuals = np.zeros((len(estimated_deltas), 3))
    for i in range(len(estimated_deltas)):
        estimated_rotation = estimated_deltas[i][:3, :3]
        ground_truth_rotation = ground_truth_deltas[i][:3, :3]
        # Find the difference between the two rotations
        rotation_diff = ground_truth_rotation.T @ estimated_rotation
        # Convert the rotation matrix to a rodrigues vector
        rotation_vector, _ = cv2.Rodrigues(rotation_diff)
        rotation_residuals[i] = rotation_vector.flatten()

    return position_residuals, rotation_residuals

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
limit = 100
visualize = False

if visualize:
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
    if i >= limit - 1:
        break
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
plot_trajectory(estimated_transformations, ax, color='purple', show_triads=True)
plot_trajectory(ground_truth_transformations[:limit], ax, color='g', show_triads=True)
ax.legend(['Estimated', 'Ground Truth'])
ax.set_title('Camera trajectory')
plt.show(block=False)  # Use block=False to prevent blocking

# Calculate the residuals
position_residuals, rotation_residuals = get_delta_residuals(estimated_transformations, ground_truth_transformations[:limit])

mag_position_residuals = np.linalg.norm(position_residuals, axis=1)
mag_rotation_residuals = np.linalg.norm(rotation_residuals, axis=1)

# Calculate RMS errors
pos_rms_x = np.sqrt(np.mean(position_residuals[:, 0]**2))
pos_rms_y = np.sqrt(np.mean(position_residuals[:, 1]**2))
pos_rms_z = np.sqrt(np.mean(position_residuals[:, 2]**2))
pos_rms_total = np.sqrt(np.mean(mag_position_residuals**2))

rot_rms_x = np.sqrt(np.mean(rotation_residuals[:, 0]**2))
rot_rms_y = np.sqrt(np.mean(rotation_residuals[:, 1]**2))
rot_rms_z = np.sqrt(np.mean(rotation_residuals[:, 2]**2))
rot_rms_total = np.sqrt(np.mean(mag_rotation_residuals**2))

print(f"Position RMS errors - X: {pos_rms_x:.4f}, Y: {pos_rms_y:.4f}, Z: {pos_rms_z:.4f}, Total: {pos_rms_total:.4f}")
print(f"Rotation RMS errors - X: {rot_rms_x:.4f}, Y: {rot_rms_y:.4f}, Z: {rot_rms_z:.4f}, Total: {rot_rms_total:.4f}")

fig2 = plt.figure(figsize=(1, 5))
ax1 = fig2.add_subplot(2, 2, 1, projection='3d')
ax2 = fig2.add_subplot(2, 2, 3, projection='3d')
ax3 = fig2.add_subplot(2, 2, (2, 4))

ax1.plot([p[0] for p in position_residuals], 
         [p[1] for p in position_residuals], 
         [p[2] for p in position_residuals], color='r')
ax1.set_title('Position Residuals')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2.plot([r[0] for r in rotation_residuals],
         [r[1] for r in rotation_residuals], 
         [r[2] for r in rotation_residuals], color='b')
ax2.set_title('Rotation Residuals')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

ax3.plot(mag_position_residuals, color='r', label='Position Residuals')
ax3.plot(mag_rotation_residuals, color='b', label='Rotation Residuals')
ax3.set_title('Magnitude of Residuals')
ax3.set_xlabel('Frame')
ax3.set_ylabel('Magnitude')
ax3.legend()

plt.tight_layout()
plt.show()