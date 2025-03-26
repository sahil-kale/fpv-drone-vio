import numpy as np
from interface import VisionAbsoluteOdometry, IMUInputFrame

GRAVITY_M_PER_S_SQUARED = 9.81

#   Goal: 
#   Take in IMU Data for system dynamics prediction
#   Steps:
#       - Take in IMU data from dataset
#       - Use changes in ang and lin for system dynamics
#       - Spit out new states for each time step

#   States:
#   [x_w, y_w, z_w, t_w_x, t_w_y, t_w_z]^T

class IMUKalmanFilter:
    def __init__(self, dt, initial_state, initial_covariance, process_noise, measurement_noise):
        self.dt = dt
        self.state = initial_state

        self.num_states = 6
        self.P = np.eye(self.num_states) * initial_covariance
        self.Q = np.eye(self.num_states) * process_noise
        self.R = np.eye(self.num_states) * measurement_noise
        
        self.K = 1
        self.C = np.eye(self.num_states)

        self.x_k = np.zeros(self.num_states) # [x, y, z, t_x, t_y, t_z]

        self.gravity = np.array([0, 0, -GRAVITY_M_PER_S_SQUARED])

    def predict(self, dt, imu_input_frame: IMUInputFrame):
        # Extract IMU data
        ang_vel = imu_input_frame[:3]
        lin_acc = imu_input_frame[3:]
        # Update dt
        self.dt = dt

        # Extract state variables
        x, y, z, t_x, t_y, t_z = self.state
        gyro_x = ang_vel[0]
        gyro_y = ang_vel[1]
        gyro_z = ang_vel[2]
        acc_x = lin_acc[0]
        acc_y = lin_acc[1]
        acc_z = lin_acc[2]

        drone_to_world_frame_matrix = self.euler_to_rotation_matrix(t_x, t_y, t_z)

        # Transfer the gyro vector to the world frame
        gyro_world = drone_to_world_frame_matrix @ np.array([gyro_x, gyro_y, gyro_z])

        # Integrate the gyro vector to get the new orientation
        t_x += gyro_world[0] * self.dt
        t_y += gyro_world[1] * self.dt
        t_z += gyro_world[2] * self.dt

        # Transfer the acceleration vector to the world frame
        acc_world = drone_to_world_frame_matrix @ np.array([acc_x, acc_y, acc_z]) - self.gravity

        # Integrate the acceleration vector to get the new position
        x += acc_world[0] * self.dt * self.dt / 2
        y += acc_world[1] * self.dt * self.dt / 2
        z += acc_world[2] * self.dt * self.dt / 2

        # Update the state
        self.state = np.array([x, y, z, t_x, t_y, t_z]).reshape(self.num_states, 1)
    
    def update(self, camera_measurments: VisionAbsoluteOdometry):
        
        #assume camera_masurements is the real x,y,z pos

        # Kalman Gain: K = P_k|k-1 * C^T (C*P_k|k-1*C^T + R_k )^-1
        # X_k|k = x_k|k-1 + K * (z_k - z_k|k-1)
        # P_k,k = (I - KC)P_k|k-1

        measurement_vector = camera_measurments.get_measurement_vector().reshape(self.num_states, 1)

        self.K = self.P @ np.transpose(self.C) @ np.linalg.inv((self.C @ self.P @ np.transpose(self.C) + self.R))
        self.x_k = self.x_k + self.K * ( measurement_vector - self.state) # Not sure how to get the real output, z
        self.P = (np.eye(self.num_states) - self.K @ self.C) @ self.P

    def euler_to_rotation_matrix(self, t_x, t_y, t_z):
        # Convert euler angles to rotation matrix
        # t_x is roll, t_y is pitch, t_z is yaw
        cos_t_x = np.cos(t_x)
        sin_t_x = np.sin(t_x)
        cos_t_y = np.cos(t_y)
        sin_t_y = np.sin(t_y)
        cos_t_z = np.cos(t_z)
        sin_t_z = np.sin(t_z)

        return np.array([
            [cos_t_y * cos_t_z, sin_t_x * sin_t_y * cos_t_z - cos_t_x * sin_t_z, cos_t_x * sin_t_y * cos_t_z + sin_t_x * sin_t_z],
            [cos_t_y * sin_t_z, sin_t_x * sin_t_y * sin_t_z + cos_t_x * cos_t_z, cos_t_x * sin_t_y * sin_t_z - sin_t_x * cos_t_z],
            [-sin_t_y, sin_t_x * cos_t_y, cos_t_x * cos_t_y]])
