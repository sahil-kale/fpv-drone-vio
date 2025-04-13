import numpy as np
from interface import *
from scipy.spatial.transform import Rotation as R

def euler_to_rotation_matrix(euler_vec, order='xyz'): 
    """
    Converts Euler angles to a rotation matrix.
    
    Parameters:
      euler_vec (array-like): [roll, pitch, yaw] or [phi, theta, psi] in radians.
      order (str): Specifies the axis sequence, e.g. 'zyx' means first rotate about x, then y, then z.
                   Adjust this depending on how your quaternion_to_euler function orders angles.
                   
    Returns:
      A 3x3 rotation matrix.
    """
    r = R.from_euler(order, euler_vec)
    return r.as_matrix()

class VIOTranslator:
    def __init__(self, initial_state: EKFDroneState):
        self.initial_state = initial_state
    
    def integrate_predicted_state_estimate(self, vision_relative_odometery: VisionRelativeOdometry):
        translation_vector = vision_relative_odometery.get_relative_translation_vector() 

        rotation_matrix = self.get_prev_state_rotation_matrix()

        # Rotate the relative translation into the world frame
        rotated_translation = rotation_matrix @ translation_vector
        
        self.initial_state.state[:2] += rotated_translation[:2]
        self.initial_state.state[2] += rotated_translation[2]*0.3
        assert self.initial_state.state.shape == (9,), "State vector must be of shape (9,)"

    def get_prev_state_rotation_matrix(self):
        prev_state = self.initial_state.get_state()
        euler_vec = prev_state[6:]
        rotation_matrix = euler_to_rotation_matrix(euler_vec)
        return rotation_matrix
    
    def get_current_state_vector(self):
        return self.initial_state.get_state()
    
    def update_state_estimate(self, state: EKFDroneState):
        self.initial_state = state