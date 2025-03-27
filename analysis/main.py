import pandas as pd
import numpy as np
import os
from interface import IMUInputFrame, VisionInputFrame, VisionRelativeOdometry, VisionAbsoluteOdometry, EKFDroneState
from imu_ekf import IMUKalmanFilter
from converting_quaternion import *
from visualizer import Visualizer

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
        imu_input_frames.append(IMUInputFrame(acceleration[i], angular_velocity[i]))
    return imu_input_frames

def develop_array_of_ground_truth(timestamp, position, orientation) -> list[EKFDroneState]:
    # Create an array of EKFDroneState objects
    gt_states = []
    for i in range(len(timestamp)):
        position_vec = np.array([position[i][0], position[i][1], position[i][2]])
        quaternion_vec = np.array([orientation[i][0], orientation[i][1], orientation[i][2], orientation[i][3]])
        # Convert quaternion to Euler angles
        euler_vec = quaternion_to_euler(quaternion_vec[0], quaternion_vec[1], quaternion_vec[2], quaternion_vec[3])
        combined_state_vec = np.concatenate((position_vec, euler_vec)).reshape(-1)
        # Create EKFDroneState object
        gt_states.append(EKFDroneState(combined_state_vec))

    return gt_states

# Main
if __name__ == '__main__':
    imu_path = os.getcwd() + '/dataset/vio_dataset_1/imu.txt'
    df_imu = load_data(imu_path)

    NUM_SAMPLES_TO_IGNORE = 500
    
    # Extract into arrays for timestamp (column 1), acceleration (columns 2-4), and angular velocity (columns 5-7)
    imu_timestamp = df_imu[1]
    acceleration = df_imu.iloc[:, 5:8]
    angular_velocity = df_imu.iloc[:, 2:5]

    # Convert into numpy arrays
    imu_timestamp = imu_timestamp.to_numpy()[NUM_SAMPLES_TO_IGNORE:].astype(float)
    acceleration = acceleration.to_numpy()[NUM_SAMPLES_TO_IGNORE:].astype(float)
    angular_velocity = angular_velocity.to_numpy()[NUM_SAMPLES_TO_IGNORE:].astype(float)

    imu_input_frames = develop_array_of_imu_input_frame(imu_timestamp, acceleration, angular_velocity)

    df_ground_truth = pd.read_csv(os.getcwd() + '/dataset/vio_dataset_1/groundtruth.txt', sep=' ', header=None)
    # Extract into arrays for timestamp (column 1), position (columns 2-4), and orientation quaternion (columns 5-8)
    gt_timestamp = df_ground_truth[0]
    gt_position = df_ground_truth.iloc[:, 1:4]
    gt_orientation = df_ground_truth.iloc[:, 4:8]

    # Convert into numpy arrays
    gt_timestamp = gt_timestamp.to_numpy()[NUM_SAMPLES_TO_IGNORE:].astype(float)
    gt_position = gt_position.to_numpy()[NUM_SAMPLES_TO_IGNORE:].astype(float)
    gt_orientation = gt_orientation.to_numpy()[NUM_SAMPLES_TO_IGNORE:].astype(float)

    # Convert into array of EKFDroneState objects
    gt_states = develop_array_of_ground_truth(gt_timestamp, gt_position, gt_orientation)

    # Pass into the EKF
    initial_state = gt_states[0].state
    initial_covariance = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    process_noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    measurement_noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    dt = 0.002

    ekf = IMUKalmanFilter(dt, initial_state, initial_covariance, process_noise, measurement_noise)

    ekf_states = []

    for imu_input_frame in imu_input_frames:
        ekf.predict(dt, imu_input_frame)
        ekf_states.append(ekf.get_state())
    
    visualizer = Visualizer(ekf_states, None, gt_states, None)
    visualizer.plot_3d_trajectory_animation(plot_ground_truth=True)



