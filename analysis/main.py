import pandas as pd
import numpy as np
import os
from interface import IMUInputFrame, VisionInputFrame, VisionRelativeOdometry, VisionAbsoluteOdometry, EKFDroneState
from imu_ekf import IMUKalmanFilter
from converting_quaternion import *
from visualizer import Visualizer
from util import conditional_breakpoint
import computer_vision as mycv
from vio_update_bridge import VIOTranslator
import argparse
from madgwick import MadgwickFilter


# Need to implement data ingestion and data processing here

def load_data(path):
    # Load the dataset/vio_dataset_1/imu.txt into a pandas dataframe
    # put together os.getcwd() and the path to the dataset
    df = pd.read_csv(path, sep=' ', header=None)
    return df

def develop_array_of_imu_input_frame(timestamp, acceleration, angular_velocity):
    # Create an array of IMUInputFrame objects
    imu_input_frames = []
    for i in range(len(timestamp)):
        imu_input_frames.append(IMUInputFrame(angular_velocity[i], acceleration[i], timestamp[i]))
    return imu_input_frames

def develop_array_of_ground_truth(timestamp, position, orientation) -> list[EKFDroneState]:
    # Create an array of EKFDroneState objects
    gt_states = []
    for i in range(len(timestamp)):
        position_vec = np.array([position[i][0], position[i][1], position[i][2]])
        quaternion_vec = np.array([orientation[i][0], orientation[i][1], orientation[i][2], orientation[i][3]])
        # Convert quaternion to Euler angles
        euler_vec = quaternion_xyzw_to_euler(quaternion_vec[0], quaternion_vec[1], quaternion_vec[2], quaternion_vec[3])
        velocity_vec = np.array([0,0,0])
        combined_state_vec = np.concatenate((position_vec, velocity_vec, euler_vec)).reshape(-1)
        # Create EKFDroneState object
        gt_states.append(EKFDroneState(combined_state_vec))

    return gt_states

def develop_array_of_vision_input_frame(dataset_dir, timestamp, image_path_left, image_path_right):
    # Create an array of VisionInputFrame objects
    vision_input_frames = []
    for i in range(len(timestamp)):
        vision_input_frames.append(VisionInputFrame(dataset_dir + image_path_left[i], dataset_dir + image_path_right[i], timestamp[i]))
    return vision_input_frames

def estimate_gyro_bias(imu_input_frames):
    gyro_bias = np.zeros(3)
    for imu_input_frame in imu_input_frames:
        gyro_data = imu_input_frame.get_gyro_data()
        gyro_bias += gyro_data
    return gyro_bias / len(imu_input_frames)


def align_trajectories_ekf(state_estimates, state_ground_truth):
    """
    Aligns estimated drone trajectory to ground truth using Kabsch’s algorithm.
    
    Args:
        state_estimates (list of EKFDroneState): Estimated drone states.
        state_ground_truth (list of EKFDroneState): Ground truth states.

    Returns:
        aligned_estimates (np.ndarray): Transformed estimated positions.
        R_opt (np.ndarray): Optimal rotation matrix (3x3).
        t_opt (np.ndarray): Optimal translation vector (3x1).
    """
    # Extract world positions from the EKFDroneState objects
    est_positions = np.array([state.get_world_position() for state in state_estimates])
    gt_positions = np.array([state.get_world_position() for state in state_ground_truth])

    # Compute centroids
    est_centroid = np.mean(est_positions, axis=0)
    gt_centroid = np.mean(gt_positions, axis=0)

    # Center the data
    est_centered = est_positions - est_centroid
    gt_centered = gt_positions - gt_centroid

    # Compute optimal rotation using SVD (Kabsch algorithm)
    H = est_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T

    # Ensure a proper rotation (det(R) should be +1)
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T

    # Compute optimal translation
    t_opt = gt_centroid - R_opt @ est_centroid

    # Apply transformation to align estimated positions
    aligned_estimates = (R_opt @ est_positions.T).T + t_opt

    return aligned_estimates, R_opt, t_opt

# Main
if __name__ == '__main__':

    # Add arguments for using gyro and accelerometer ground truth data

    # Code to parse arguments
    parser = argparse.ArgumentParser(description='Arguments for whether ground truth should be used for gyro or accelerometer data')
    parser.add_argument('--use-gyro-ground-truth', action='store_true', default=False, help='Whether gyro ground truth should be used')
    parser.add_argument('--use-accel-ground-truth', action='store_true', default=False, help='Whether accelerometer ground truth should be used')
    parser.add_argument('--steps', type=int, default=15, help='Number of steps to downsample the data for visualization')
    parser.add_argument('--end-stamp', type=int, default=2500, help='Number of samples to use for simulation')

    args = parser.parse_args()


    DATASET_DIR = os.getcwd() + '/dataset/vio_dataset_1/'

    df_ground_truth = pd.read_csv(os.getcwd() + '/dataset/vio_dataset_1/groundtruth.txt', sep=' ', header=None)
    # Extract into arrays for timestamp (column 1), position (columns 2-4), and orientation quaternion (columns 5-8)
    gt_timestamp = df_ground_truth[0]
    gt_position = df_ground_truth.iloc[:, 1:4]
    gt_orientation = df_ground_truth.iloc[:, 4:8]

    # Convert into numpy arrays
    gt_timestamp = gt_timestamp.to_numpy()[1:].astype(float)
    gt_position = gt_position.to_numpy()[1:].astype(float)
    gt_orientation = gt_orientation.to_numpy()[1:].astype(float)

    # Convert into array of EKFDroneState objects
    gt_states = develop_array_of_ground_truth(gt_timestamp, gt_position, gt_orientation)

    imu_path = DATASET_DIR + 'imu.txt'
    df_imu = load_data(imu_path)
    
    # Extract into arrays for timestamp (column 1), acceleration (columns 2-4), and angular velocity (columns 5-7)
    imu_timestamp = df_imu[1]
    angular_velocity = df_imu.iloc[:, 2:5]
    acceleration = df_imu.iloc[:, 5:8]

    # Convert into numpy arrays
    imu_timestamp = imu_timestamp.to_numpy()[1:].astype(float)
    acceleration = acceleration.to_numpy()[1:].astype(float)
    angular_velocity = angular_velocity.to_numpy()[1:].astype(float)

    imu_input_frames = develop_array_of_imu_input_frame(imu_timestamp, acceleration, angular_velocity)

#   # Load the left and right image paths
    img_paths_left_txt_file = DATASET_DIR + 'left_images.txt'
    img_paths_right_txt_file = DATASET_DIR + 'right_images.txt'
    df_img_paths_left = load_data(img_paths_left_txt_file)
    df_img_paths_right = load_data(img_paths_right_txt_file)
    image_timestamps = df_img_paths_left[1].to_numpy()[1:].astype(float)
    image_paths_left = df_img_paths_left.iloc[:, 2].tolist()[1:]
    image_paths_right = df_img_paths_right.iloc[:, 2].tolist()[1:]
    vision_input_frames = develop_array_of_vision_input_frame(DATASET_DIR, image_timestamps, image_paths_left, image_paths_right)
    
    # IMU and CAM data start recording earlier than GT data
    index_at_which_imu_data_is_synced = 0
    for i, timestamp in enumerate(imu_timestamp):
        if timestamp >= gt_timestamp[0]:
            index_at_which_imu_data_is_synced = i
            break
    
    imu_input_frames = imu_input_frames[index_at_which_imu_data_is_synced:]
    imu_timestamp = imu_timestamp[index_at_which_imu_data_is_synced:]

    NUM_FRAMES_TO_IGNORE = 500
    END_STAMP = args.end_stamp
    gyro_bias = estimate_gyro_bias(imu_input_frames[0:NUM_FRAMES_TO_IGNORE])

    imu_input_frames = imu_input_frames[NUM_FRAMES_TO_IGNORE:END_STAMP]
    imu_timestamp = imu_timestamp[NUM_FRAMES_TO_IGNORE:END_STAMP]
    gt_states = gt_states[NUM_FRAMES_TO_IGNORE:END_STAMP]

    index_at_which_vision_data_is_synced = 0
    for i, timestamp in enumerate(image_timestamps):
        if timestamp >= imu_timestamp[0]:
            index_at_which_vision_data_is_synced = i
            break
    vision_input_frames = vision_input_frames[index_at_which_vision_data_is_synced:]
    image_timestamps = image_timestamps[index_at_which_vision_data_is_synced:]

    # Pass into the EKF
    initial_state = gt_states[0].state
    starting_quaternion_xyzw = gt_orientation[NUM_FRAMES_TO_IGNORE]
    starting_quaternion = np.array([starting_quaternion_xyzw[3], starting_quaternion_xyzw[0], starting_quaternion_xyzw[1], starting_quaternion_xyzw[2]])
    initial_covariance = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1])
    process_noise = np.array([0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1])
    measurement_noise = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1])

    NUM_STATES = 9
    dt = 0.002
    ekf = IMUKalmanFilter(dt, initial_state, initial_covariance, process_noise, measurement_noise, NUM_STATES, gyro_bias)
    ekf_states = []

    #Initialize the computer vision relative odometry calculator
    vision_system = mycv.VisionRelativeOdometryCalculator(
        initial_camera_input= vision_input_frames[0],
        feature_extractor= mycv.SIFTFeatureExtractor(),
        feature_matcher= mycv.FLANNMatcher(),
        feature_match_filter= mycv.RANSACFilter()
    )


    current_image_index = 0 #variable to keep track of vision data index

    madgwick_filter = MadgwickFilter(initial_quaternion=starting_quaternion, gyro_bias=gyro_bias, mu=0.01)
    MADGWICK_FILTER_MAX_ROTATION_RATE_RAD_PER_SEC = 5
    madgwick_filter.compute_optimal_mu(max_qdot=MADGWICK_FILTER_MAX_ROTATION_RATE_RAD_PER_SEC, dt=dt) 

    for i, imu_input_frame in enumerate(imu_input_frames):
        madgwick_filter.update(imu_input_frame, dt)
        if (args.use_gyro_ground_truth):
            # Use gyro ground truth data
            imu_input_frame.gyro_data = gt_states[i].state[6:9]
        else:
            imu_input_frame.gyro_data = madgwick_filter.get_euler_angles()

        ekf.predict(dt, imu_input_frame)

        #Integrate the vision data

        if imu_timestamp[i] >= image_timestamps[current_image_index]:
            if current_image_index < len(vision_input_frames):
                if current_image_index == 0:
                    # Use the first image to initialize the VIOTranslator
                    vio_translator = VIOTranslator(initial_state=ekf.get_state())
                else:
                    vision_relative_odometry = vision_system.calculate_relative_odometry(vision_input_frames[current_image_index])
                    current_image_index += 1

                    vio_translator.integrate_predicted_state_estimate(vision_relative_odometry)
                
                    #Update the EKF with the vision absolute odometry
                    # abs_cv_state = vio_translator.get_current_state_vector()
                    # absolute_translation_vector = abs_cv_state[:3]

                    # vision_absolute_odometry = VisionAbsoluteOdometry(absolute_translation_vector, np.zeros((3,)))

            # -- Update the EKF using the vision absolute odometry --
            ekf.update(vio_translator.get_current_state_vector())

            # We only update this right after using the vision to update the ekf,
            # because the relative transformation is between camera frame k, k-1 not imu frame n, n-1
            vio_translator.update_state_estimate(ekf.get_state())


        ekf_states.append(ekf.get_state())

    visualizer = Visualizer(ekf_states, gt_states, imu_timestamp, vision_input_frames, image_timestamps, downsample=True, step=args.steps)
    visualizer.plot_3d_trajectory_animation(plot_ground_truth=True)
