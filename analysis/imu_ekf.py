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
        self.max_error = 0

        self.gravity = np.array([0, 0, -GRAVITY_M_PER_S_SQUARED]).reshape(3, 1)

        self.imu_to_drone_rotation_matrix = elementary_rotation_matrix_x(np.pi)
        self.gyro_bias = gyro_bias.reshape(3, 1)

    def predict(self, dt, imu_input_frame: IMUInputFrame):
        # Extract IMU data
        ang_vel = imu_input_frame.get_gyro_data()
        lin_acc = imu_input_frame.get_accel_data()
        # Update dt
        self.dt = dt



        # Extract state variables
        x, y, z, v_x, v_y, v_z, t_x, t_y, t_z = self.state


        gyro_x = ang_vel[0] - self.gyro_bias[0]
        gyro_y = ang_vel[1] - self.gyro_bias[1]
        gyro_z = ang_vel[2] - self.gyro_bias[2]


        acc_x = lin_acc[0]
        acc_y = lin_acc[1]
        acc_z = lin_acc[2]

        world_to_drone_rotation_matrix = euler_to_rotation_matrix(t_x.item(), t_y.item(), t_z.item()).reshape(3, 3)
        drone_to_world_rotation_matrix = world_to_drone_rotation_matrix.T
       

        gyro_in_drone_frame = self.imu_to_drone_rotation_matrix @ np.array([gyro_x, gyro_y, gyro_z]).reshape(3, 1)

        # Transfer the gyro vector to the world frame
        gyro_world = drone_to_world_rotation_matrix @ gyro_in_drone_frame


        # Integrate the gyro vector to get the new orientation
        
        # t_x += gyro_world[0] * self.dt
        # t_y += gyro_world[1] * self.dt
        # t_z += gyro_world[2] * self.dt

        # Transfer the acceleration vector to the world frame
        acc_in_drone_frame = self.imu_to_drone_rotation_matrix @ np.array([acc_x, acc_y, acc_z]).reshape(3, 1)

        acc_world = drone_to_world_rotation_matrix @ acc_in_drone_frame - self.gravity


        # Integrate the acceleration vector to get the new position
        x = x +  v_x*dt + 0.5*acc_world[0]*dt*dt
        y = y +  v_y*dt + 0.5*acc_world[1]*dt*dt
        z = z +  v_z*dt + 0.5*acc_world[2]*dt*dt


        v_x = v_x + acc_world[0]*dt
        v_y = v_y + acc_world[1]*dt
        v_z = v_z + acc_world[2]*dt




        # if self.counter < 5:
        #     print("acc_body", lin_acc.ravel())
        #     print("acc_world", acc_world.ravel())
        #     self.counter+=1
        
        self.state = np.array([x.item(), y.item(), z.item(), v_x.item(), v_y.item(), v_z.item(), t_x.item(), t_y.item(), t_z.item()]).reshape(self.num_states, 1)

        self.P = self.A @ self.P @ np.transpose(self.A) + self.Q
    
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
