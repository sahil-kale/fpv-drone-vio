import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg  # For reading image paths
from converting_quaternion import euler_to_rotation_matrix
from interface import VisionInputFrame, EKFDroneState


class Visualizer:
    def __init__(self, states, ground_truth, state_timeseries, vision_input_frames, vision_state_timeseries):
        self.states = states
        self.ground_truth = ground_truth
        self.state_timeseries = state_timeseries
        self.vision_input_frames = vision_input_frames
        self.vision_state_timeseries = vision_state_timeseries
        self.vision_timeseries_index = 0

    def plot_3d_coordinate_axes(self, ax, state: EKFDroneState, color_strs=['r', 'g', 'b']):
        origin = state.get_world_position()
        quiver_length = 0.5
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

    def plot_3d_trajectory_animation(self, plot_ground_truth=True):
        fig = plt.figure(figsize=(12, 6))
        ax_traj = fig.add_subplot(121, projection='3d')
        ax_left_img = fig.add_subplot(222)
        ax_right_img = fig.add_subplot(224)

        ax_traj.set_title("Drone 3D Trajectory")
        ax_traj.set_xlabel("X")
        ax_traj.set_ylabel("Y")
        ax_traj.set_zlabel("Z")

        est_positions = np.array([state.get_world_position() for state in self.states])
        if plot_ground_truth:
            gt_positions = np.array([state.get_world_position() for state in self.ground_truth])

        ax_traj.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], label='Estimated', color='blue')
        if plot_ground_truth:
            ax_traj.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label='Ground Truth', color='orange')

        all_positions = est_positions
        if plot_ground_truth:
            all_positions = np.vstack((est_positions, gt_positions))
        starting_point = est_positions[0]
        min_vals = np.min(all_positions, axis=0) - 0.1 - np.array(starting_point).reshape(-1)
        max_vals = np.max(all_positions, axis=0) + 0.1 - np.array(starting_point).reshape(-1)
        global_min = np.min(min_vals)
        global_max = np.max(max_vals)
        ax_traj.set_xlim(global_min + starting_point[0], global_max + starting_point[0])
        ax_traj.set_ylim(global_min + starting_point[1], global_max + starting_point[1])
        ax_traj.set_zlim(global_min + starting_point[2], global_max + starting_point[2])
        ax_traj.set_box_aspect([1, 1, 1])
        ax_traj.legend()

        est_quiver_artists = []
        gt_quiver_artists = []

        # Initialize blank images
        dummy_img = np.zeros((480, 640))  # Change dimensions if needed
        left_image_display = ax_left_img.imshow(dummy_img, cmap='gray')
        left_image_display.set_clim(0.0, 1.0)
        right_image_display = ax_right_img.imshow(dummy_img, cmap='gray')
        right_image_display.set_clim(0.0, 1.0)
        ax_left_img.set_title("Left Stereo Image")
        ax_right_img.set_title("Right Stereo Image")
        ax_left_img.axis('off')
        ax_right_img.axis('off')

        def update(frame):
            for artist in est_quiver_artists:
                artist.remove()
            est_quiver_artists.clear()

            for artist in gt_quiver_artists:
                artist.remove()
            gt_quiver_artists.clear()

            est_state = self.states[frame]
            est_quivers = self.plot_3d_coordinate_axes(ax_traj, est_state, color_strs=['b', 'c', 'm'])
            est_quiver_artists.extend(est_quivers)

            if plot_ground_truth:
                gt_state = self.ground_truth[frame]
                gt_quivers = self.plot_3d_coordinate_axes(ax_traj, gt_state, color_strs=['r', 'g', 'y'])
                gt_quiver_artists.extend(gt_quivers)

            current_time = self.state_timeseries[frame]
            if self.vision_timeseries_index < len(self.vision_state_timeseries):
                next_vision_time = self.vision_state_timeseries[self.vision_timeseries_index]
                if current_time >= next_vision_time:
                    vision_frame = self.vision_input_frames[self.vision_timeseries_index]

                    left_path = vision_frame.get_image_left_path()
                    right_path = vision_frame.get_image_right_path()
                    left_image = mpimg.imread(left_path)
                    right_image = mpimg.imread(right_path)

                    print(f"Loaded images for time: {next_vision_time}: {left_path}, {right_path}")

                    left_image_display.set_data(left_image)
                    right_image_display.set_data(right_image)

                    self.vision_timeseries_index += 1

            #print(f"Frame: {frame}, Estimated Position: {est_state.get_world_position()}, Ground Truth Position: {gt_state.get_world_position() if plot_ground_truth else 'N/A'}")
            return est_quiver_artists + gt_quiver_artists + [left_image_display, right_image_display]

        ani = animation.FuncAnimation(fig, update, frames=len(self.states), interval=0.1, blit=False)
        plt.tight_layout()
        plt.show()
