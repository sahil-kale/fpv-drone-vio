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

#Load the file
data = np.loadtxt('dataset/vio_dataset_1/groundtruth.txt')
#Load timestamp, tx ty tz qx qy qz qw from second row on
timestamp = data[1:, 0]

with open('dataset/vio_dataset_1/homogenous_ground_truth_converted_by_us.txt', 'w') as f:
    f.write('timestamp tx ty tz qx qy qz qw\n')

#Convert the quaternion to homogeneous matrix
for i in range(1, data.shape[0]):
    tx, ty, tz, qx, qy, qz, qw = data[i, 1:]
    T = convert_to_homogeneous(tx, ty, tz, qx, qy, qz, qw)
    #write to new file
    with open('dataset/vio_dataset_1/homogenous_ground_truth_converted_by_us.txt', 'a') as f:
        f.write(str(timestamp[i-1]) + ' ')
        for j in range(4):
            for k in range(4):
                f.write(str(T[j, k]) + ' ')
        f.write('\n')
