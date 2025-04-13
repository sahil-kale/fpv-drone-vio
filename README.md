# FPV Drone VIO
Implementation of a visual-inertial odometry algorithm for MTE 546: Sensor Fusion

## Dataset Link
https://fpv.ifi.uzh.ch/datasets/

## Install instructions
Requirements: Python >3.10
1. `pip install -r requirements.txt`

## Run instructions
`python3 analysis/main.py` (run `--help` for possible arguments)

To run a selected file for longer, use `--end-stamp=10000` to run for a longer time period.

To select datasets, use `--dataset-path` to switch the dataset upon which the algorithm is evaluated on

## Analysis Directory Breakdown
`computer_vision.py`: TODO
`converting_quaternion.py`: Contains utilities to convert between quaternion and euler angle representations of angles, as well as rotation matrix transforms.
`imu_ekf.py`: Contains the position EKF that fuses IMU and visual odometry information to estimate the drone's position.
`interface.py`: Contains common interfaces used to communicate between discrete software modules
`madgwick.py`: Contains the implementation of the Madgwick filter
`main.py`: Runs the VIO, EKF, and Madgwick algorithms
`util.py`: General python debug utilities
`vio_update_bridge.py`: Wrapper class used by `main.py` to integrate relative transformations from the VIO class.
`vision_assessment.py`: TODO
`visualizer_orientation.py`: Visualizer for orientation metrics
`visualizer.py`: 3D pose and track visualizer 