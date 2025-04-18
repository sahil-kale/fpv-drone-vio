import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_matrix(qx, qy, qz, qw):
    """Convert quaternion (qx, qy, qz, qw) to a 3x3 rotation matrix."""
    rot = R.from_quat([qx, qy, qz, qw])
    return rot.as_matrix()

def convert_to_homogeneous(tx, ty, tz, qx, qy, qz, qw):
    """Convert translation (tx, ty, tz) and quaternion (qx, qy, qz, qw) to a 4x4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, :3] = quaternion_to_matrix(qx, qy, qz, qw)  # Rotation part
    T[:3, 3] = [tx, ty, tz]  # Translation part
    return T

def quaternion_xyzw_to_euler(qx, qy, qz, qw):
    """Convert quaternion (qx, qy, qz, qw) to Euler angles (roll, pitch, yaw)."""
    rot = R.from_quat([qx, qy, qz, qw])
    return rot.as_euler('xyz', degrees=False)

def elementary_rotation_matrix_x(theta):
    """Generate a rotation matrix for a rotation around the x-axis by theta radians."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def elementary_rotation_matrix_y(theta):
    """Generate a rotation matrix for a rotation around the y-axis by theta radians."""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def elementary_rotation_matrix_z(theta):
    """Generate a rotation matrix for a rotation around the z-axis by theta radians."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def euler_to_rotation_matrix(t_x, t_y, t_z):
    rot_z = elementary_rotation_matrix_z(t_z)
    rot_y = elementary_rotation_matrix_y(t_y)
    rot_x = elementary_rotation_matrix_x(t_x)

    combined = rot_z @ rot_y @ rot_x
    assert combined.shape == (3, 3), "Combined rotation matrix must be of shape (3, 3)"
    return combined

