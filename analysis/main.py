import pandas as pd
import numpy as np
import os
from interface import IMUInputFrame, VisionInputFrame, VisionRelativeOdometry, VisionAbsoluteOdometry
from imu_ekf import IMUKalmanFilter

# Need to implement data ingestion and data processing here

def load_data():
    # Load the dataset/vio_dataset_1/imu.txt into a pandas dataframe
    # put together os.getcwd() and the path to the dataset
    df = pd.read_csv(os.getcwd() + '/dataset/vio_dataset_1/imu.txt', sep=' ', header=None)
    return df

def develop_array_of_imu_input_frame(timestamp, acceleration, angular_velocity):
    # Create an array of IMUInputFrame objects
    imu_input_frames = []
    for i in range(len(timestamp)):
        imu_input_frames.append(IMUInputFrame(acceleration[i], angular_velocity[i]))
    return imu_input_frames

# Main
if __name__ == '__main__':
    df = load_data()
    
    # Extract into arrays for timestamp (column 1), acceleration (columns 2-4), and angular velocity (columns 5-7)
    imu_timestamp = df[1]
    acceleration = df.iloc[:, 5:8]
    angular_velocity = df.iloc[:, 2:5]

    # Convert into numpy arrays
    imu_timestamp = imu_timestamp.to_numpy()[1:].astype(float)
    acceleration = acceleration.to_numpy()[1:].astype(float)
    angular_velocity = angular_velocity.to_numpy()[1:].astype(float)

    imu_input_frames = develop_array_of_imu_input_frame(imu_timestamp, acceleration, angular_velocity)

    # Pass into the EKF
    initial_state = np.array([0, 0, 0, 0, 0, 0])
    initial_covariance = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    process_noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    measurement_noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    dt = 0.002

    ekf = IMUKalmanFilter(dt, initial_state, initial_covariance, process_noise, measurement_noise)
    
    for imu_input_frame in imu_input_frames:
        ekf.predict(dt, imu_input_frame)
    
    print(ekf.state)



