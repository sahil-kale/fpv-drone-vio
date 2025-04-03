import numpy as np
from interface import *
from scipy.spatial.transform import Rotation as R

def euler_to_rotation_matrix(euler_vec, order='zyx'): #TODO: This order needs to be checked.
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
        rotation_vector = vision_relative_odometery.get_relative_rotation_vector() # theta x, theta y, theta z

        rotation_matrix = euler_to_rotation_matrix(rotation_vector)

        # Rotate the relative translation into the world frame
        rotated_translation = np.dot(rotation_matrix, translation_vector) #TODO: Check if this should be inverted
        
        delta_state = np.concatenate((rotated_translation, rotation_vector, np.zeros(3,)))
        assert delta_state.shape == (9,), "State vector must be of shape (9,)" #I changed it to 9 because it is 9 now? not sure
        # Rotate the relative translation into the world frame

        self.initial_state.state += delta_state
        assert self.initial_state.state.shape == (9,), "State vector must be of shape (9,)"

    def get_prev_state_rotation_matrix(self):
        prev_state = self.initial_state.get_state()
        euler_vec = prev_state[3:]
        rotation_matrix = euler_to_rotation_matrix(euler_vec)
        return rotation_matrix
    
    def get_current_state_vector(self):
        return self.initial_state.get_state()
    
    def update_state_estimate(self, state: EKFDroneState):
        self.initial_state.state = state.get_state()

    # 1) after getting vision odom rel output, call integrate predicted state estimate
    # 2) Need to feed in vision abs odom to ekf class
    # 3) Call update state estimate to update translator state estimate with ekf estimate
