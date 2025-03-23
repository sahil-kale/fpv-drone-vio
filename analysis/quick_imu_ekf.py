import numpy as np

class IMUKalmanFilter:
    def __init__(self, dt):
        self.dt = dt  # Time step
        self.state = np.zeros(9)  # [roll, pitch, yaw, x, y, z, v_x, v_y, v_z]
        self.P = np.eye(9) * 0.1  # Initial covariance
        self.Q = np.eye(9) * 0.01  # Process noise covariance
        self.R = np.eye(3) * 0.1  # Measurement noise covariance

    def predict(self, ang_vel, lin_acc):
        roll, pitch, yaw, x, y, z, v_x, v_y, v_z = self.state
        
        # Angular velocity integration (Euler angles)
        roll += ang_vel[0] * self.dt
        pitch += ang_vel[1] * self.dt
        yaw += ang_vel[2] * self.dt
        
        # Acceleration integration for velocity and position
        v_x += lin_acc[0] * self.dt
        v_y += lin_acc[1] * self.dt
        v_z += (lin_acc[2] - 9.81) * self.dt  # Subtract gravity
        
        x += v_x * self.dt
        y += v_y * self.dt
        z += v_z * self.dt
        
        self.state = np.array([roll, pitch, yaw, x, y, z, v_x, v_y, v_z])
        
        # State transition matrix (simple integration model)
        F = np.eye(9)
        F[3, 6] = self.dt  # x depends on v_x
        F[4, 7] = self.dt  # y depends on v_y
        F[5, 8] = self.dt  # z depends on v_z
        
        self.P = F @ self.P @ F.T + self.Q

    def update(self, lin_acc):
        # Measurement matrix (we measure acceleration in x, y, z)
        H = np.zeros((3, 9))
        H[0, 6] = 1  # Measure v_x
        H[1, 7] = 1  # Measure v_y
        H[2, 8] = 1  # Measure v_z
        
        z = np.array(lin_acc)  # Measurement
        y = z - H @ self.state  # Innovation
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.state += K @ y
        self.P = (np.eye(9) - K @ H) @ self.P

    def get_state(self):
        return self.state
