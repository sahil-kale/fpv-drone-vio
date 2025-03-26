import numpy as np
import matplotlib.pyplot as plt
from interface import *
import matplotlib.animation as animation

class Visualizer:
    def __init__(self, states: EKFDroneState, state_timeseries: np.ndarray, ground_truth: EKFDroneState, ground_truth_timeseries: np.ndarray):
        self.states = states
        self.state_timeseries = state_timeseries
        self.ground_truth = ground_truth
        self.ground_truth_timeseries = ground_truth_timeseries
    
    def plot_3d_coordinate_axes(self, ax, state: EKFDroneState, color_strs: list = ['r', 'g', 'b']):
        origin = state.get_world_position()
        quiver_length = 0.1
        quiver_matrix = np.eye(3) * quiver_length
        tx, ty, tz = state.get_world_orientation()
        quiver_matrix = euler_to_rotation_matrix(tx, ty, tz) @ quiver_matrix
        x_quiver = quiver_matrix[:, 0]
        y_quiver = quiver_matrix[:, 1]
        z_quiver = quiver_matrix[:, 2]

        qx = ax.quiver(*origin, *x_quiver, color=color_strs[0], length=quiver_length, normalize=True)
        qy = ax.quiver(*origin, *y_quiver, color=color_strs[1], length=quiver_length, normalize=True)
        qz = ax.quiver(*origin, *z_quiver, color=color_strs[2], length=quiver_length, normalize=True)

        return [qx, qy, qz]

    def plot_3d_trajectory_animation(self, plot_ground_truth: bool = True):
        """
        Plot the 3D trajectory of the drone over time. A timeseries for both the state and ground truth is available. 
        The states from the EKF and the ground truth are plotted in the same 3D space, using the plot_3d_coordinate_axes function to show the states orientation.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Drone 3D Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        est_positions = np.array([state.get_world_position() for state in self.states])
        if plot_ground_truth:
            gt_positions = np.array([state.get_world_position() for state in self.ground_truth])

        # Plot full trajectories as background
        ax.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], label='Estimated', color='blue')
        if plot_ground_truth:
            ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label='Ground Truth', color='orange')

        # Set axis limits for better visibility
        all_positions = est_positions
        if plot_ground_truth:
            all_positions = np.vstack((est_positions, gt_positions))
        min_vals = np.min(all_positions, axis=0) - 0.1
        max_vals = np.max(all_positions, axis=0) + 0.1

        global_min = np.min(min_vals)
        global_max = np.max(max_vals)
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.set_zlim(global_min, global_max)


        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        
        # Plot initialization
        est_quiver_artists = []
        gt_quiver_artists = []

        def update(frame):
            # Remove previous quivers
            for artist in est_quiver_artists:
                artist.remove()
            est_quiver_artists.clear()

            for artist in gt_quiver_artists:
                artist.remove()
            gt_quiver_artists.clear()

            # Plot estimated quivers
            est_state = self.states[frame]
            est_quivers = self.plot_3d_coordinate_axes(ax, est_state, color_strs=['b', 'c', 'm'])
            est_quiver_artists.extend(est_quivers)

            # Plot ground truth quivers
            if plot_ground_truth:
                gt_state = self.ground_truth[frame]
                gt_quivers = self.plot_3d_coordinate_axes(ax, gt_state, color_strs=['r', 'g', 'y'])
                gt_quiver_artists.extend(gt_quivers)

            print(f"Frame: {frame}, Estimated Position: {est_state.get_world_position()}, Ground Truth Position: {gt_state.get_world_position() if plot_ground_truth else 'N/A'}")

            return est_quiver_artists + gt_quiver_artists

        ani = animation.FuncAnimation(fig, update, frames=len(self.states), interval=100, blit=False)
        ax.legend()
        plt.show()

if __name__ == "__main__":
    # Example usage
    initial_state = EKFDroneState(np.array([0, 0, 0, 0, 0, 0], dtype=np.float64))
    
    num_test_states = 100

    ground_truth = [initial_state] * num_test_states
    state_timeseries = np.linspace(0, 1, num_test_states)
    ground_truth_timeseries = np.linspace(0, 1, num_test_states)

    states = [initial_state]
    for i in range(num_test_states):
        random_state = np.random.rand(6) * 0.01
        new_state = EKFDroneState(states[-1].state + random_state)
        states.append(new_state)

    visualizer = Visualizer(states, state_timeseries, ground_truth, ground_truth_timeseries)
    visualizer.plot_3d_trajectory_animation()