import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from quick_imu_ekf import IMUKalmanFilter

def load_imu_data(file_path):
    return pd.read_csv(file_path, delim_whitespace=True, comment='#',
                       names=['timestamp', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z'])

def load_ground_truth(file_path):
    return pd.read_csv(file_path, delim_whitespace=True, comment='#',
                       names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

def run_kalman_filter(imu_data, ground_truth_data, dt):
    kf = IMUKalmanFilter(dt)
    estimated_positions = []
    # cheat a little and set the initial state to the first ground truth position
    kf.state[0:3] = ground_truth_data.iloc[0][['tx', 'ty', 'tz']].to_numpy()
    
    for _, row in imu_data.iterrows():
        ang_vel = np.array([row['ang_vel_x'], row['ang_vel_y'], row['ang_vel_z']])
        lin_acc = np.array([row['lin_acc_x'], row['lin_acc_y'], row['lin_acc_z']])
        
        kf.predict(ang_vel, lin_acc)
        kf.update(lin_acc)
        estimated_positions.append(kf.state[:3])

    return np.array(estimated_positions)

def plot_3d_animation(imu_positions, gt_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim([np.min(gt_positions[:, 0]), np.max(gt_positions[:, 0])])
    ax.set_ylim([np.min(gt_positions[:, 1]), np.max(gt_positions[:, 1])])
    ax.set_zlim([np.min(gt_positions[:, 2]), np.max(gt_positions[:, 2])])
    
    imu_scatter, = ax.plot([], [], [], 'bo', markersize=8, label='IMU KF Estimate')
    gt_scatter, = ax.plot([], [], [], 'ro', alpha=0.3, label='Ground Truth')
    
    def update(frame):
        imu_scatter.set_data(imu_positions[:frame, 0], imu_positions[:frame, 1])
        imu_scatter.set_3d_properties(imu_positions[:frame, 2])
        
        gt_scatter.set_data(gt_positions[:frame, 0], gt_positions[:frame, 1])
        gt_scatter.set_3d_properties(gt_positions[:frame, 2])
        
        return imu_scatter, gt_scatter
    
    ani = animation.FuncAnimation(fig, update, frames=len(imu_positions), interval=0.1, blit=False)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def main(imu_file, gt_file):
    imu_data = load_imu_data(imu_file)
    gt_data = load_ground_truth(gt_file)
    
    dt = np.mean(np.diff(imu_data['timestamp']))
    gt_positions = gt_data[['tx', 'ty', 'tz']].to_numpy()
    imu_positions = run_kalman_filter(imu_data, gt_data, dt)
    
    plot_3d_animation(imu_positions, gt_positions)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize IMU Kalman Filter vs Ground Truth')
    parser.add_argument('imu_file', type=str, help='Path to IMU data file')
    parser.add_argument('gt_file', type=str, help='Path to ground truth file')
    args = parser.parse_args()
    
    main(args.imu_file, args.gt_file)
