from converting_quaternion import convert_to_homogeneous
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Convert VIO dataset to homogeneous format.')
parser.add_argument('--dataset', type=str, choices=["training", "testing"], default="testing", help='Path to the dataset directory [training, testing]')

#Load the file
args = parser.parse_args()

dataset_path = r'dataset/indoor_forward_9_snapdragon_with_gt'
if args.dataset == "training":
    dataset_path = r'dataset/vio_dataset_1'

#load the file
data = np.loadtxt(os.path.join(dataset_path, 'groundtruth.txt'))

timestamp = data[:, 0]
t = data[:, 1:4]
q = data[:, 4:8]

#Find rotation matrix
R = np.zeros((data.shape[0], 3, 3))
for i in range(data.shape[0]):
    R[i] = convert_to_homogeneous(t[i, 0], t[i, 1], t[i, 2], q[i, 0], q[i, 1], q[i, 2], q[i, 3])[:3, :3]

#Create homo matrices
homo = []
for i in range(data.shape[0]):
    homo_matrix = np.eye(4)
    homo_matrix[:3, :3] = R[i]
    homo_matrix[:3, 3] = t[i]
    homo.append(homo_matrix)

#save to file
filepath = os.path.join(dataset_path, "homogenous_ground_truth_converted_by_us.txt")
with open(filepath, 'w') as f:
    f.write("timestamp T00 T01 T02 T03 T10 T11 T12 T13 T20 T21 T22 T23 T30 T31 T32 T33")
    for i in range(len(homo)):
        f.write(f"\n{timestamp[i]:.6f} " + " ".join(map(str, homo[i].flatten())))