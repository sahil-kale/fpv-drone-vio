import numpy as np

class EKFDroneState:
    def __init__(self, initial_state: np.ndarray):
        self.state = initial_state
        assert self.state.shape == (6,), "State vector must be of shape (6,)"

    def get_state(self) -> np.ndarray:
        return self.state
    
    def get_world_position(self) -> np.ndarray:
        return self.state[:3]
    
    def get_world_orientation(self) -> np.ndarray:
        return self.state[3:6]
    
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
    def __init__(self, gyro_data: np.ndarray, accel_data: np.ndarray):
        self.gyro_data = gyro_data
        self.accel_data = accel_data
    
    def get_gyro_data(self) -> np.ndarray:
        return self.gyro_data
    
    def get_accel_data(self) -> np.ndarray:
        return self.accel_data
    
class VisionInputFrame:
    def __init__(self, image_path_left: str, image_path_right: str):
        self.image_left = image_path_left
        self.image_right = image_path_right
    
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


def euler_to_rotation_matrix(t_x, t_y, t_z):
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