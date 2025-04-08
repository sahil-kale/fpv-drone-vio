import numpy as np
import matplotlib.pyplot as plt
from interface import *

class VisualizerOrientationOnly:
    def __init__(self, gt_orientation_list, est_orientation_list, dt):
        """
        gt_orientation_list and est_orientation_list are lists of OrientationState objects
        """
        self.gt_orientation_list = gt_orientation_list
        self.est_orientation_list = est_orientation_list
        self.min_len = min(len(gt_orientation_list), len(est_orientation_list))
        self.dt = dt

    def plot_and_compute_stats(self):
        num_samples = self.min_len
        errors = np.zeros((num_samples, 3))
        gt_eulers = np.zeros((num_samples, 3))
        est_eulers = np.zeros((num_samples, 3))
        for i in range(num_samples):
            gt_euler = self.gt_orientation_list[i].get_quaternion_as_euler()
            est_euler = self.est_orientation_list[i].get_quaternion_as_euler()
            x_error_wrapped = np.arctan2(np.sin(est_euler[0] - gt_euler[0]), np.cos(est_euler[0] - gt_euler[0]))
            y_error_wrapped = np.arctan2(np.sin(est_euler[1] - gt_euler[1]), np.cos(est_euler[1] - gt_euler[1]))
            z_error_wrapped = np.arctan2(np.sin(est_euler[2] - gt_euler[2]), np.cos(est_euler[2] - gt_euler[2]))
            errors[i, :] = np.array([x_error_wrapped, y_error_wrapped, z_error_wrapped])
            gt_eulers[i] = gt_euler
            est_eulers[i] = est_euler

        rmse = np.sqrt(np.mean(errors**2, axis=0))
        # Plotting
        time = np.arange(num_samples) * self.dt

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ['Roll', 'Pitch', 'Yaw']
        for i in range(3):
            axes[i].plot(time, gt_eulers[:, i], label='Ground Truth', linestyle='--')
            axes[i].plot(time, est_eulers[:, i], label='Estimate')
            axes[i].set_ylabel(f'{labels[i]} (rad)')
            axes[i].legend()
            axes[i].grid(True)

        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('Orientation Comparison (Euler Angles)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        print("RMSE (Roll, Pitch, Yaw) - rad:", rmse)
        plt.show()

        return rmse
