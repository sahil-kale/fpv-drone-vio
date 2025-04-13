import os
from interface import VisionInputFrame, VisionRelativeOdometry
import computer_vision as mycv
import cv2

import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=["training", "testing"], default="testing", help='Path to the dataset directory [training, testing]')

args = parser.parse_args()

dataset_path = r'dataset/vio_dataset_1'
if args.dataset == "testing":
    dataset_path = r'dataset/indoor_forward_6_snapdragon_with_gt'

# Read ground truth file and get the first timestamp
with open(os.path.join(dataset_path, 'homogenous_ground_truth_converted_by_us.txt')) as f:
    ground_truth_lines = f.readlines()[1:]  # skip header 
first_gt_time = float(ground_truth_lines[0].split()[0])

#load left images
with open(os.path.join(dataset_path,'left_images.txt')) as f:
    image_timestamps = []
    left_images = []
    for i, line in enumerate(f):
        if line.strip() and not line.startswith("#"):
            parts = line.split()
            if len(parts) == 3:
                left_images.append(os.path.join(dataset_path, parts[2]))
                image_timestamps.append(float(parts[1]))

#load right imagrs
with open(os.path.join(dataset_path,'right_images.txt')) as f:
    right_images = []
    for line in f:
        if line.strip() and not line.startswith("#"):
            parts = line.split()
            if len(parts) == 3:
                right_images.append(os.path.join(dataset_path, parts[2]))

#Check that the arrays are the same length
if len(left_images) != len(right_images):
    raise ValueError("The number of left and right images do not match.")

START_FRAME = 500

first_valid_index = next(i for i, ts in enumerate(image_timestamps) if ts >= first_gt_time)
image_timestamps = image_timestamps[first_valid_index+START_FRAME:]
left_images = left_images[first_valid_index+START_FRAME:]
right_images = right_images[first_valid_index+START_FRAME:]

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
                #
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
        for i, T in enumerate(trajectory):
            if i % 10 == 0:  # Only plot every tenth triad
                pos = T[0:3, 3]
                
                R = T[0:3, 0:3]
                
                reduced_scale = triad_scale / 3
                
                ax.quiver(pos[0], pos[1], pos[2],
                        R[0, 0], R[1, 0], R[2, 0],
                        color='r', length=reduced_scale)
                
                ax.quiver(pos[0], pos[1], pos[2],
                        R[0, 1], R[1, 1], R[2, 1],
                        color='b', length=reduced_scale)
                
                ax.quiver(pos[0], pos[1], pos[2],
                        R[0, 2], R[1, 2], R[2, 2],
                        color='k', length=reduced_scale)

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

    return position_residuals, rotation_residuals, ground_truth_deltas

def get_scaled_residuals(position_residuals, rotation_residuals, ground_truth_deltas, eps=1e-3):
    mag_position_residuals = np.linalg.norm(position_residuals, axis=1)
    mag_rotation_residuals = np.linalg.norm(rotation_residuals, axis=1)
    
    scaled_position_residuals = []
    scaled_rotation_residuals = []
    
    for i in range(len(ground_truth_deltas)):
        # use the norm of the ground truth translation vector
        gt_trans = ground_truth_deltas[i][:3, 3]
        translation_scale = np.linalg.norm(gt_trans)
        if translation_scale < eps:
            translation_scale = eps
        scaled_position_residuals.append(position_residuals[i] / translation_scale)
        
        # convert the ground truth rotation to a rodrigues vector
        gt_rot = ground_truth_deltas[i][:3, :3]
        gt_rodrigues, _ = cv2.Rodrigues(gt_rot)
        rotation_scale = np.linalg.norm(gt_rodrigues)
        if rotation_scale < eps:
            rotation_scale = eps
        scaled_rotation_residuals.append(rotation_residuals[i] / rotation_scale)
        
    return np.array(scaled_position_residuals), np.array(scaled_rotation_residuals)



#Initialize a VisionOdometryCalculator using the first pair of images
#this will be used to track the camera pose in the world frame
initial_frame = VisionInputFrame(left_images[0], right_images[0], timestamp=image_timestamps[0])

odometry_calculator = mycv.VisionRelativeOdometryCalculator(initial_camera_input=initial_frame,
                                              feature_extractor=mycv.AKAZEFeatureExtractor(n_features=1000),
                                              feature_matcher=mycv.FLANNMatcher(trees=10, checks=100),
                                              feature_match_filter=mycv.RANSACFilter(min_matches=12, reproj_thresh=1),
                                              transformation_threshold=0.1)

#Set up arrays to track pose
estimated_transformations = []
estimated_transformations.append(ground_truth_transformations[0])

counter = 0
maximum = len(ground_truth_transformations)
limit = 500

for i, (left_image, right_image) in enumerate(zip(left_images, right_images), start=1):
    if i >= limit - 1:
        break
    print(i)
    
    input_frame = VisionInputFrame(left_image, right_image, timestamp=image_timestamps[i])

    # Calculate the relative odometry between the previous and new input frame
    relative_transformation = odometry_calculator.calculate_relative_odometry_homogenous(input_frame, camera_frame=False)

    # Decompose  relative transformation into rotation and translation
    R_new = relative_transformation[:3, :3]
    t_new = relative_transformation[:3, 3]

    filtered_R = R_new.copy()
    filtered_t = t_new.copy()

    filtered_relative_transformation = np.eye(4)
    filtered_relative_transformation[:3, :3] = filtered_R
    filtered_relative_transformation[:3, 3] = filtered_t

    # apply the new transformation to the previous one to get the new world position
    estimated_transformations.append(estimated_transformations[-1] @ filtered_relative_transformation)

#Plot the estimated vs ground truth trajectory
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Camera trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#Extract just translation
est_positions = np.array([t[:3, 3] for t in estimated_transformations])

mins = est_positions.min(axis=0)
maxs = est_positions.max(axis=0)
mid = est_positions[0]  # Use the first position as the midpoint
max_range = (maxs - mins).max()

ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

plot_trajectory(estimated_transformations, ax, color='purple', show_triads=True)
plot_trajectory(ground_truth_transformations[:limit], ax, color='g', show_triads=True)
ax.legend(['Estimated', 'Ground Truth'])
ax.set_title('Camera trajectory')
plt.show(block=False)

# Calculate the residuals
position_residuals, rotation_residuals, ground_truth_deltas = \
    get_delta_residuals(estimated_transformations, ground_truth_transformations[:limit])
scaled_position_residuals, scaled_rotation_residuals = \
    get_scaled_residuals(position_residuals, rotation_residuals, ground_truth_deltas)

mag_position_residuals = np.linalg.norm(position_residuals, axis=1)
mag_rotation_residuals = np.linalg.norm(rotation_residuals, axis=1)

# Calculate RMS errors
pos_mean_x = np.mean(position_residuals[:, 0])
pos_mean_y = np.mean(position_residuals[:, 1])
pos_mean_z = np.mean(position_residuals[:, 2])

pos_var_x = np.var(position_residuals[:, 0])
pos_var_y = np.var(position_residuals[:, 1])
pos_var_z = np.var(position_residuals[:, 2])

pos_rms_x = np.sqrt(np.mean(position_residuals[:, 0]**2))
pos_rms_y = np.sqrt(np.mean(position_residuals[:, 1]**2))
pos_rms_z = np.sqrt(np.mean(position_residuals[:, 2]**2))
pos_rms_total = np.sqrt(np.mean(mag_position_residuals**2))

rot_mean_x = np.mean(rotation_residuals[:, 0])
rot_mean_y = np.mean(rotation_residuals[:, 1])
rot_mean_z = np.mean(rotation_residuals[:, 2])

rot_var_x = np.var(rotation_residuals[:, 0])
rot_var_y = np.var(rotation_residuals[:, 1])
rot_var_z = np.var(rotation_residuals[:, 2])

rot_rms_x = np.sqrt(np.mean(rotation_residuals[:, 0]**2))
rot_rms_y = np.sqrt(np.mean(rotation_residuals[:, 1]**2))
rot_rms_z = np.sqrt(np.mean(rotation_residuals[:, 2]**2))
rot_rms_total = np.sqrt(np.mean(mag_rotation_residuals**2))

print(f"Position Mean - X: {pos_mean_x:.4f}, Y: {pos_mean_y:.4f}, Z: {pos_mean_z:.4f}")
print(f"Position RMS errors - X: {pos_rms_x:.4f}, Y: {pos_rms_y:.4f}, Z: {pos_rms_z:.4f}, Total: {pos_rms_total:.4f}\n")
print(f"Position Variance - X: {pos_var_x:.4f}, Y: {pos_var_y:.4f}, Z: {pos_var_z:.4f}")

print(f"Rotation Mean - X: {rot_mean_x:.4f}, Y: {rot_mean_y:.4f}, Z: {rot_mean_z:.4f}")
print(f"Rotation RMS errors - X: {rot_rms_x:.4f}, Y: {rot_rms_y:.4f}, Z: {rot_rms_z:.4f}, Total: {rot_rms_total:.4f}")
print(f"Rotation Variance - X: {rot_var_x:.4f}, Y: {rot_var_y:.4f}, Z: {rot_var_z:.4f}")

fig2 = plt.figure(figsize=(10, 10))
ax1 = fig2.add_subplot(2, 2, 1, projection='3d')
ax2 = fig2.add_subplot(2, 2, 3, projection='3d')
ax3 = fig2.add_subplot(2, 2, (2, 4))

ax1.plot([p[0] for p in position_residuals], 
         [p[1] for p in position_residuals], 
         [p[2] for p in position_residuals], color='r')
# Position Residuals plot
ax1.set_title('Position Residuals')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Mark the origin
ax1.scatter([0], [0], [0], color='black', s=50, marker='o')

# Center around origin
pos_max_range = max(
    abs(position_residuals[:, 0]).max(),
    abs(position_residuals[:, 1]).max(),
    abs(position_residuals[:, 2]).max()
)
ax1.set_xlim(-pos_max_range, pos_max_range)
ax1.set_ylim(-pos_max_range, pos_max_range)
ax1.set_zlim(-pos_max_range, pos_max_range)

# Rotation Residuals plot
ax2.plot([r[0] for r in rotation_residuals],
         [r[1] for r in rotation_residuals], 
         [r[2] for r in rotation_residuals], color='b')
ax2.set_title('Rotation Residuals')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')


# Mark the origin
ax2.scatter([0], [0], [0], color='black', s=50, marker='o')

# Center around origin
rot_max_range = max(
    abs(rotation_residuals[:, 0]).max(),
    abs(rotation_residuals[:, 1]).max(),
    abs(rotation_residuals[:, 2]).max()
)
ax2.set_xlim(-rot_max_range, rot_max_range)
ax2.set_ylim(-rot_max_range, rot_max_range)
ax2.set_zlim(-rot_max_range, rot_max_range)

ax3.plot(position_residuals[:, 0], color='r', label='Position Residuals X')
ax3.plot(position_residuals[:, 1], color='g', label='Position Residuals Y')
ax3.plot(position_residuals[:, 2], color='k', label='Position Residuals Z')
ax3.plot(rotation_residuals[:, 0], color='magenta', label='Rotation Residuals X')
ax3.plot(rotation_residuals[:, 1], color='cyan', label='Rotation Residuals Y')
ax3.plot(rotation_residuals[:, 2], color='purple', label='Rotation Residuals Z')
ax3.set_title('Magnitude of Residuals')
ax3.set_xlabel('Frame')
ax3.set_ylabel('Magnitude')
ax3.legend()

plt.tight_layout()
plt.show(block=False)

# Calculate RMS errors
pos_mean_x = np.mean(scaled_position_residuals[:, 0])
pos_mean_y = np.mean(scaled_position_residuals[:, 1])
pos_mean_z = np.mean(scaled_position_residuals[:, 2])

pos_rms_x = np.sqrt(np.mean(scaled_position_residuals[:, 0]**2))
pos_rms_y = np.sqrt(np.mean(scaled_position_residuals[:, 1]**2))
pos_rms_z = np.sqrt(np.mean(scaled_position_residuals[:, 2]**2))

rot_mean_x = np.mean(scaled_rotation_residuals[:, 0])
rot_mean_y = np.mean(scaled_rotation_residuals[:, 1])
rot_mean_z = np.mean(scaled_rotation_residuals[:, 2])

rot_rms_x = np.sqrt(np.mean(scaled_rotation_residuals[:, 0]**2))
rot_rms_y = np.sqrt(np.mean(scaled_rotation_residuals[:, 1]**2))
rot_rms_z = np.sqrt(np.mean(scaled_rotation_residuals[:, 2]**2))

print(f"\n\nScaled Position Mean - X: {pos_mean_x:.4f}, Y: {pos_mean_y:.4f}, Z: {pos_mean_z:.4f}")
print(f"Scaled Position RMS errors - X: {pos_rms_x:.4f}, Y: {pos_rms_y:.4f}, Z: {pos_rms_z:.4f}\n")
print(f"Scaled Rotation Mean - X: {rot_mean_x:.4f}, Y: {rot_mean_y:.4f}, Z: {rot_mean_z:.4f}")
print(f"Scaled Rotation RMS errors - X: {rot_rms_x:.4f}, Y: {rot_rms_y:.4f}, Z: {rot_rms_z:.4f}")


fig3 = plt.figure(figsize=(10, 10))
ax4 = fig3.add_subplot(2, 2, 1, projection='3d')
ax5 = fig3.add_subplot(2, 2, 3, projection='3d')
ax6 = fig3.add_subplot(2, 2, (2, 4))

# Scaled Position Residuals plot
ax4.plot([sp[0] for sp in scaled_position_residuals], 
         [sp[1] for sp in scaled_position_residuals], 
         [sp[2] for sp in scaled_position_residuals], color='r')
ax4.set_title('Scaled Position Residuals')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
# Mark the origin
ax4.scatter([0], [0], [0], color='black', s=50, marker='o')
# Center around origin
pos_max_range = max(
    abs(scaled_position_residuals[:, 0]).max(),
    abs(scaled_position_residuals[:, 1]).max(),
    abs(scaled_position_residuals[:, 2]).max()
)
ax4.set_xlim(-pos_max_range, pos_max_range)
ax4.set_ylim(-pos_max_range, pos_max_range)
ax4.set_zlim(-pos_max_range, pos_max_range)

# Scaled Rotation Residuals plot
ax5.plot([sr[0] for sr in scaled_rotation_residuals],
         [sr[1] for sr in scaled_rotation_residuals], 
         [sr[2] for sr in scaled_rotation_residuals], color='b')
ax5.set_title('Scaled Rotation Residuals')
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')
# Mark the origin
ax5.scatter([0], [0], [0], color='black', s=50, marker='o')
# Center around origin
rot_max_range = max(
    abs(scaled_rotation_residuals[:, 0]).max(),
    abs(scaled_rotation_residuals[:, 1]).max(),
    abs(scaled_rotation_residuals[:, 2]).max()
)
ax5.set_xlim(-rot_max_range, rot_max_range)
ax5.set_ylim(-rot_max_range, rot_max_range)
ax5.set_zlim(-rot_max_range, rot_max_range)

ax6.plot(np.linalg.norm(scaled_position_residuals,axis=1), color='r', label='Scaled Position Residuals')
ax6.plot(np.linalg.norm(scaled_rotation_residuals,axis=1), color='b', label='Scaled Rotation Residuals')
ax6.set_title('Magnitude of Scaled Residuals')
ax6.set_xlabel('Frame')
ax6.set_ylabel('Magnitude')
ax6.legend()

plt.show(block=False)

#Plot distributions of residuals x, y, z, rotx, roty, rotz
labels = ['$x$', '$y$', '$z$', '$\\theta_x$', '$\\theta_y$', '$\\theta_z$']
colors = ['r', 'g', 'b']
data = [position_residuals, rotation_residuals]
units = ['[m]', '[mrad]']

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})


fig4, axs = plt.subplots(2, 3, figsize=(6, 4), dpi=300)

for i in range(2):
    for j in range(3):
        axs[i, j].hist(data[i][:, j], bins=50, color=colors[j], alpha=0.7)
        axs[i, j].set_title(labels[i * 3 + j])
        axs[i, j].set_xlabel(f'Error {units[i]}')
        axs[i, j].set_ylabel('Frequency')
        if i == 1:
            axs[i, j].set_xticklabels([f'{tick * 1000:.0f}' for tick in axs[i, j].get_xticks()])
fig4.tight_layout(pad=1.5)  # Increase padding between plots
plt.show()

def calculate_covariance(position_res, rotation_res):
    cov_pr = np.cov(np.vstack([position_res.T, rotation_res.T]))
    return cov_pr


covariance_matrix = calculate_covariance(position_residuals, rotation_residuals)
print("\nCovariance Matrix:")   
print(covariance_matrix)

for i in range(6):
    for j in range(6):
        print(f"{covariance_matrix[i, j]:.5f},", end="\t")
    print("\n")