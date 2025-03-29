import pandas as pd
import numpy as np
import os
from interface import IMUInputFrame, VisionInputFrame, VisionRelativeOdometry, VisionAbsoluteOdometry, EKFDroneState
from imu_ekf import IMUKalmanFilter
from converting_quaternion import *
from visualizer import Visualizer
from util import conditional_breakpoint

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
        euler_vec = quaternion_to_euler(quaternion_vec[0], quaternion_vec[1], quaternion_vec[2], quaternion_vec[3])
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

# Main
if __name__ == '__main__':
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

    index_at_which_vision_data_is_synced = 0
    for i, timestamp in enumerate(image_timestamps):
        if timestamp >= gt_timestamp[0]:
            index_at_which_vision_data_is_synced = i
            break
    vision_input_frames = vision_input_frames[index_at_which_vision_data_is_synced:]
    image_timestamps = image_timestamps[index_at_which_vision_data_is_synced:]

    NUM_FRAMES_TO_IGNORE = 500
    NUM_FRAMES_TO_PLOT = 3000
    gyro_bias = estimate_gyro_bias(imu_input_frames[0:NUM_FRAMES_TO_IGNORE])
    imu_input_frames = imu_input_frames[NUM_FRAMES_TO_IGNORE:NUM_FRAMES_TO_PLOT]
    imu_timestamp = imu_timestamp[NUM_FRAMES_TO_IGNORE:NUM_FRAMES_TO_PLOT]
    gt_states = gt_states[NUM_FRAMES_TO_IGNORE:NUM_FRAMES_TO_PLOT]

    # Pass into the EKF
    initial_state = gt_states[0].state

    initial_covariance = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1])
    process_noise = np.array([0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1])
    measurement_noise = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1])

    NUM_STATES = 9
    dt = 0.002
    ekf = IMUKalmanFilter(dt, initial_state, initial_covariance, process_noise, measurement_noise, NUM_STATES, gyro_bias)
    ekf_states = []

    for i, imu_input_frame in enumerate(imu_input_frames):
        ekf.predict(dt, imu_input_frame)
        ekf_states.append(ekf.get_state())
    
    visualizer = Visualizer(ekf_states, gt_states, imu_timestamp, vision_input_frames, image_timestamps, downsample=True)
    visualizer.plot_3d_trajectory_animation(plot_ground_truth=True)