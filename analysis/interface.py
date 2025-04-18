import numpy as np
from converting_quaternion import quaternion_xyzw_to_euler
class EKFDroneState:
    def __init__(self, initial_state: np.ndarray):
        self.state = initial_state
        assert self.state.shape == (9,), f"State vector must be of shape (9,), got {self.state.shape}"

    def get_state(self) -> np.ndarray:
        return self.state
    
    def get_world_position(self) -> np.ndarray:
        return self.state[:3]
    
    def get_world_orientation(self) -> np.ndarray:
        return self.state[6:]
    
    def __str__(self):
        repr = f"""
        EKFDroneState:
        Position: 
        x: {self.get_world_position()[0]}
        y: {self.get_world_position()[1]}
        z: {self.get_world_position()[2]}
        Orientation:
        roll: {self.get_world_orientation()[0]}
        pitch: {self.get_world_orientation()[1]}
        yaw: {self.get_world_orientation()[2]}
        """
        return repr

class IMUInputFrame:
    def __init__(self, gyro_data: np.ndarray, accel_data: np.ndarray, timestamp):
        self.gyro_data = gyro_data
        self.accel_data = accel_data
        self.timestamp = timestamp
    
    def get_gyro_data(self) -> np.ndarray:
        return self.gyro_data
    
    def get_accel_data(self) -> np.ndarray:
        return self.accel_data
    
    def __str__(self):
        repr = f"""
        IMUInputFrame:
        Gyro data: 
        x: {self.get_gyro_data()[0]}
        y: {self.get_gyro_data()[1]}
        z: {self.get_gyro_data()[2]}
        Accel data:
        x: {self.get_accel_data()[0]}
        y: {self.get_accel_data()[1]}
        z: {self.get_accel_data()[2]}
        """
        return repr
    
class VisionInputFrame:
    def __init__(self, image_path_left: str, image_path_right: str, timestamp):
        self.image_left = image_path_left
        self.image_right = image_path_right
        self.timestamp = timestamp
    
    def get_image_left_path(self) -> str:
        return self.image_left
    
    def get_image_right_path(self) -> str:
        return self.image_right
    
class VisionRelativeOdometry:
    def __init__(self, relative_translation_vector: np.ndarray, relative_rotation_vector: np.ndarray):
        self.relative_translation_vector = relative_translation_vector
        self.relative_rotation_vector = relative_rotation_vector # theta x, theta y, theta z
        assert self.relative_rotation_vector.shape == (3,), "Rotation vector must be of shape (3,)"
        assert self.relative_translation_vector.shape == (3,), "Translation vector must be of shape (3,)"
    
    def get_relative_translation_vector(self) -> np.ndarray:
        return self.relative_translation_vector
    
    def get_relative_rotation_vector(self) -> np.ndarray:
        return self.relative_rotation_vector

def create_VisionRelativeOdometry_from_homogeneous_matrix(T: np.ndarray):
    assert T.shape == (4, 4), "Homogeneous matrix must be of shape (4, 4)"
    relative_translation_vector = T[:3, 3]
    # -- For now we are using a zero vector and neglecting rotation --
    relative_rotation_vector = np.zeros((3,))
    return VisionRelativeOdometry(relative_translation_vector, relative_rotation_vector)
    
class VisionAbsoluteOdometry:
    def __init__(self, absolute_translation_vector: np.ndarray, absolute_rotation_vector: np.ndarray):
        self.absolute_translation_vector = absolute_translation_vector
        self.absolute_rotation_vector = absolute_rotation_vector # theta x, theta y, theta z
        assert self.absolute_rotation_vector.shape == (3,), "Rotation vector must be of shape (3,)"
        assert self.absolute_translation_vector.shape == (3,), "Translation vector must be of shape (3,)"
    
    def get_absolute_translation_vector(self) -> np.ndarray:
        return self.absolute_translation_vector
    
    def get_absolute_rotation_vector(self) -> np.ndarray:
        return self.absolute_rotation_vector
    
    def get_measurement_vector(self) -> np.ndarray:
        measurement = np.concatenate((self.absolute_translation_vector, self.absolute_rotation_vector))
        assert measurement.shape == (6,), "Measurement vector must be of shape (6,)"
        return measurement

class OrientationState:
    def __init__(self, qw, qx, qy, qz):
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz
    
    def get_quaternion(self) -> np.ndarray:
        return np.array([self.qw, self.qx, self.qy, self.qz])
    
    def get_quaternion_as_euler(self):
        # Convert quaternion to euler angles
        q = self.get_quaternion()
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        return quaternion_xyzw_to_euler(qx, qy, qz, qw)
        
