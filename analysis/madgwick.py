import numpy as np
from interface import *
from converting_quaternion import quaternion_xyzw_to_euler

class MadgwickFilter:
    def __init__(self, initial_quaternion: np.ndarray, gyro_bias: np.ndarray, mu):
        self.q = initial_quaternion.reshape(4, 1)
        self.gyro_bias = gyro_bias
        self.mu = mu
        self.assert_quaternion()

    def assert_quaternion(self):
        assert np.isclose(np.linalg.norm(self.q), 1.0), "Quaternion is not normalized"
        assert self.q.shape == (4, 1), "Quaternion is not of shape (4,)"
        assert self.mu > 0, "Mu must be positive"
        assert self.gyro_bias.shape == (3,), "Gyro bias must be of shape (3,)"

    def compute_optimal_mu(self, max_qdot, dt, alpha=1.0):
        self.mu = alpha * max_qdot * dt
        return self.mu
    
    def determine_mu_from_accel(self, accel_data):
        mu_lookup_y = np.array([0.0, self.mu, 0.0])
        x_mu_lookup_const = 0.5 # how far away we allow the norm to be
        GRAVITY_M_PER_S2 = 9.81
        mu_lookup_x = np.array([GRAVITY_M_PER_S2 - GRAVITY_M_PER_S2*x_mu_lookup_const, GRAVITY_M_PER_S2, GRAVITY_M_PER_S2 + GRAVITY_M_PER_S2*x_mu_lookup_const])

        accel_norm = np.linalg.norm(accel_data)
        return np.interp(accel_norm, mu_lookup_x, mu_lookup_y)

    def quaternion_multiplication(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        # https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
        a1, b1, c1, d1 = q1.reshape(-1)
        a2, b2, c2, d2 = q2.reshape(-1)
        return np.array([
            a1*a2 - b1*b2 - c1*c2 - d1*d2,
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2
        ]).reshape(4, 1)
    
    def determine_qdot_from_gyro(self, gyro_measurements):
        gyro_data_bias_free = gyro_measurements - self.gyro_bias
        omega = np.array([0.0, *gyro_data_bias_free]).reshape(4, 1)
        q_dot = 0.5 * self.quaternion_multiplication(self.q, omega)
        assert q_dot.shape == (4, 1), "q_dot is not of shape (4, 1)"
        return q_dot
    
    def update(self, imu_input_frame: IMUInputFrame, dt: float):
        gyro_measurements = imu_input_frame.get_gyro_data()
        q_dot_from_gyro = self.determine_qdot_from_gyro(gyro_measurements)
        accel_data = imu_input_frame.get_accel_data()
        acc_normalized = accel_data / np.linalg.norm(accel_data)
        # assume the d vector is [0, 0, 1], which corrosponds to a predefined reference in the earth frame aligning with gravity
        qw, qx, qy, qz = self.q.reshape(-1)
        # equation 25: https://courses.cs.washington.edu/courses/cse466/14au/labs/l4/madgwick_internal_report.pdf, use the simplification of d
        f = np.array([
            [2*(qx*qz - qw*qy) - acc_normalized[0]],
            [2*(qw*qx + qy*qz) - acc_normalized[1]],
            [2*(0.5 - qx**2 - qy**2) - acc_normalized[2]]
        ]
        )
        J = np.array([[-2.0*qy,  2.0*qz, -2.0*qw, 2.0*qx],
                      [ 2.0*qx,  2.0*qw,  2.0*qz, 2.0*qy],
                      [ 0.0,    -4.0*qx, -4.0*qy, 0.0   ]])
        
        gradient = J.T @ f
        gradient_norm = np.linalg.norm(gradient)
        correction_qdot_from_accel = self.determine_mu_from_accel(accel_data) * gradient/gradient_norm
        q_dot = q_dot_from_gyro - correction_qdot_from_accel
        q_new = self.q + q_dot * dt
        q_new_norm = np.linalg.norm(q_new)
        self.q = q_new / q_new_norm
        self.assert_quaternion()
        return self.q

    def get_quaternion(self):
        return self.q.reshape(-1)
    
    def get_euler_angles(self):
        qw, qx, qy, qz = self.q.reshape(-1)
        return quaternion_xyzw_to_euler(qx, qy, qz, qw)