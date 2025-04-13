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
        self.C[6:][:] = 0 #Remove Velocity Components from output vector   

        self.A = np.eye(self.num_states)
        self.A[0, 3] = dt  # x += v_x * dt
        self.A[1, 4] = dt
        self.A[2, 5] = dt
        self.max_error = 0

        self.gravity = np.array([0, 0, -GRAVITY_M_PER_S_SQUARED]).reshape(3, 1)

        self.imu_to_drone_rotation_matrix = elementary_rotation_matrix_x(np.pi)
        self.gyro_bias = gyro_bias.reshape(3, 1)

    def predict(self, dt, imu_input_frame: IMUInputFrame):
        # Extract IMU data
        angle = imu_input_frame.get_gyro_data()
        lin_acc = imu_input_frame.get_accel_data()
        
        # Update dt
        self.dt = dt

        # Extract state variables
        x, y, z, v_x, v_y, v_z, t_x, t_y, t_z = self.state

        t_x = angle[0] - self.gyro_bias[0]
        t_y = angle[1] - self.gyro_bias[1]
        t_z = angle[2] - self.gyro_bias[2]

        acc_x = lin_acc[0]
        acc_y = lin_acc[1]
        acc_z = lin_acc[2]

        # Transfer the acceleration vector to the world frame
        world_to_drone_rotation = euler_to_rotation_matrix(t_x.item(), t_y.item(), t_z.item())
        drone_to_world_rotation = world_to_drone_rotation.T

        world_accel = (world_to_drone_rotation @ np.array([acc_x,acc_y,acc_z]).reshape(3,1)) + self.gravity


        # Integrate the acceleration vector to get the new position
        x = x +  v_x*dt  
        y = y +  v_y*dt 
        z = z +  v_z*dt

        v_x = v_x + world_accel[0]*dt
        v_y = v_y + world_accel[1]*dt
        v_z = v_z + world_accel[2]*dt
        
        self.state = np.array([x.item(), y.item(), z.item(), v_x.item(), v_y.item(), v_z.item(), t_x.item(), t_y.item(), t_z.item()]).reshape(self.num_states, 1)

        self.P = self.A @ self.P @ np.transpose(self.A) + self.Q
    
    def update(self, camera_measurements: VisionAbsoluteOdometry):
        cam_meas = camera_measurements
        measurement_vector = np.concatenate((cam_meas, np.zeros(self.num_states - len(cam_meas))))
        measurement_vector = measurement_vector.reshape(self.num_states, 1)

        self.K = self.P @ np.transpose(self.C) @ np.linalg.inv((self.C @ self.P @ np.transpose(self.C) + self.R))
        self.state = self.state + self.K @ ( measurement_vector - self.C @ self.state) # Not sure how to get the real output, z
        self.P = (np.eye(self.num_states) - self.K @ self.C) @ self.P

    def get_state(self) -> EKFDroneState:
        return EKFDroneState(self.state.reshape(-1))
