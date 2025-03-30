import numpy as np
from interface import VisionAbsoluteOdometry, IMUInputFrame, EKFDroneState
from converting_quaternion import *
from util import conditional_breakpoint

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
    def __init__(self, dt, initial_state, initial_covariance: np.ndarray, process_noise: np.ndarray, measurement_noise: np.ndarray, num_states, gyro_bias: np.ndarray):
        self.dt = dt
        self.state = initial_state

        self.num_states = num_states
        self.P = np.eye(self.num_states) * initial_covariance
        self.Q = np.eye(self.num_states) * process_noise
        self.R = np.eye(self.num_states) * measurement_noise
        
        self.K = np.eye(self.num_states)
        self.C = np.eye(self.num_states) #Measurement Model, or H Matrix
        self.C[2:5][:] = 0 #Remove Velocity Components from output vector   

        self.A = np.eye(9)
        self.A[0, 3] = dt  # x += v_x * dt
        self.A[1, 4] = dt
        self.A[2, 5] = dt
        self.counter = 0

        self.gravity = np.array([0, 0, -GRAVITY_M_PER_S_SQUARED]).reshape(3, 1)

        self.imu_to_drone_rotation_matrix = elementary_rotation_matrix_x(np.pi)
        self.gyro_bias = gyro_bias.reshape(3, 1)

    def predict(self, dt, imu_input_frame: IMUInputFrame):
        # Extract IMU data
        ang_vel = imu_input_frame.get_gyro_data()
        lin_acc = imu_input_frame.get_accel_data()

        self.dt = dt
        x, y, z, v_x, v_y, v_z, t_x, t_y, t_z = self.state

        # Bias-corrected angular velocity
        gyro = np.array([
            ang_vel[0] - self.gyro_bias[0],
            ang_vel[1] - self.gyro_bias[1],
            ang_vel[2] - self.gyro_bias[2]
        ]).reshape(3, 1)

        # Raw linear acceleration (IMU frame)
        acc = np.array([lin_acc[0], lin_acc[1], lin_acc[2]]).reshape(3, 1)

        # Apply IMU-to-drone rotation matrix (handle axis flips)
        gyro_drone = self.imu_to_drone_rotation_matrix @ gyro
        acc_drone = self.imu_to_drone_rotation_matrix @ acc

        # Construct rotation matrix from Euler angles
        R_world_to_drone = euler_to_rotation_matrix(t_x.item(), t_y.item(), t_z.item())
        R_drone_to_world = R_world_to_drone.T  # Inverse

        # === Orientation Update ===
        t_x += gyro_drone[0] * dt
        t_y += gyro_drone[1] * dt
        t_z += gyro_drone[2] * dt

        # Normalize angles to [-π, π]
        t_x = (t_x + np.pi) % (2 * np.pi) - np.pi
        t_y = (t_y + np.pi) % (2 * np.pi) - np.pi
        t_z = (t_z + np.pi) % (2 * np.pi) - np.pi

        # === Acceleration → World Frame ===
        acc_world = R_drone_to_world @ acc_drone - self.gravity  # gravity should be [0, 0, 9.81] for Z-down IMU

        # === Position & Velocity Update ===
        x += v_x * dt
        y += v_y * dt
        z += v_z * dt

        v_x += acc_world[0] * dt
        v_y += acc_world[1] * dt
        v_z += acc_world[2] * dt

        # Debug output
        if self.counter < 5:
            print("acc_body (drone frame)", acc_drone.ravel())
            print("acc_world", acc_world.ravel())
            self.counter += 1

        self.state = np.array([
            x.item(), y.item(), z.item(),
            v_x.item(), v_y.item(), v_z.item(),
            t_x.item(), t_y.item(), t_z.item()
        ]).reshape(self.num_states, 1)

        self.P = self.A @ self.P @ self.A.T + self.Q
    
    def update(self, camera_measurments: VisionAbsoluteOdometry):
        
        #assume camera_masurements is the real x,y,z pos

        # Kalman Gain: K = P_k|k-1 * C^T (C*P_k|k-1*C^T + R_k )^-1
        # X_k|k = x_k|k-1 + K * (z_k - z_k|k-1)
        # P_k,k = (I - KC)P_k|k-1

        measurement_vector = camera_measurments.get_measurement_vector().reshape(self.num_states, 1)

        self.K = self.P @ np.transpose(self.C) @ np.linalg.inv((self.C @ self.P @ np.transpose(self.C) + self.R))
        self.state = self.state + self.K * ( measurement_vector - self.C @ self.state) # Not sure how to get the real output, z
        self.P = (np.eye(self.num_states) - self.K @ self.C) @ self.P

    def get_state(self) -> EKFDroneState:
        return EKFDroneState(self.state.reshape(-1))
