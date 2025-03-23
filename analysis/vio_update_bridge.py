import numpy as np
from interface import *

class VIOTranslator:
    def __init__(self, initial_state: EKFDroneState):
        self.initial_state = initial_state
    
    def integrate_predicted_state_estimate(self, vision_relative_odometery: VisionRelativeOdometry):
        delta_state = np.concatenate(vision_relative_odometery.get_relative_translation_vector(), vision_relative_odometery.get_relative_rotation_vector())
        assert delta_state.shape == (6,), "State vector must be of shape (6,)"
        self.initial_state.state += delta_state
        assert self.initial_state.state.shape == (6,), "State vector must be of shape (6,)"

    def update_state_estimate(self, state: EKFDroneState):
        self.initial_state.state = state.get_state()
