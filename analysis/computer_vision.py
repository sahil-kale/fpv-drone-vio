import interface
import cv2
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import yaml
import matplotlib.pyplot as plt

#Load left and right images
def load_images(input:interface.VisionInputFrame):
    left_image = cv2.imread(input.get_image_left_path())
    right_image = cv2.imread(input.get_image_right_path())
    return left_image, right_image

#Preprocess images
def preprocess_images(left_image, right_image):
    #Convert to grayscale
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    return left_gray, right_gray

#Detect "keypoints" and compute descriptors using a feature detector
class FeatureExtractor(ABC):
    """
    Abstract class for feature extraction
    """
    @abstractmethod
    def extract_features(self, image):
        """
        Take a single image input (greyscale) and return keypoints and descriptors
        """
        pass

class ORBFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using ORB (Oriented FAST and Rotated BRIEF) algorithm
    """
    def __init__(self, n_features=500):
        self.n_features = n_features
        self.orb = cv2.ORB_create(nfeatures=n_features)
    
    def extract_features(self, image):
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors

class SIFTFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using SIFT (Scale-Invariant Feature Transform) algorithm
    """
    def __init__(self, n_features=500):
        self.n_features = n_features
        self.sift = cv2.SIFT_create(nfeatures=n_features)
    
    def extract_features(self, image):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return keypoints, descriptors

class AKAZEFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using AKAZE (Accelerated-KAZE) algorithm
    """
    def __init__(self, n_features=500):
        self.n_features = n_features
        self.akaze = cv2.AKAZE_create()
    
    def extract_features(self, image):
        keypoints, descriptors = self.akaze.detectAndCompute(image, None)
        return keypoints, descriptors

#Match Features between Left and Right Images
class FeatureMatcher(ABC):
    """
    Abstract class for feature matching
    """
    @abstractmethod
    def match_features(self, left_descriptors, right_descriptors):
        """
        Take keypoints and descriptors from left and right images and return matches
        """
        pass

class BFMatcher(FeatureMatcher):
    """
    Feature matcher using Brute-Force algorithm
    """
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def match_features(self, left_descriptors, right_descriptors):
        matches = self.bf.match(left_descriptors, right_descriptors)
        return matches

class FLANNMatcher(FeatureMatcher):
    """
    Feature matcher using FLANN (Fast Library for Approximate Nearest Neighbors) algorithm
    """
    def __init__(self):
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    
    def match_features(self, left_descriptors, right_descriptors):
        matches = self.flann.knnMatch(left_descriptors, right_descriptors, k=2)
        # Convert matches from list of lists to list of DMatch objects
        matches = [m[0] for m in matches if len(m) == 2]
        return matches


#Filter matches to remove outliers
class FeatureMatchFilter(ABC):
    """
    Abstract class for feature match filtering
    """
    @abstractmethod
    def filter_matches(self, matches, keypoints_left=None, keypoints_right=None):
        """
        Take matches and return filtered matches
        """
        pass

class RatioTestFilter(FeatureMatchFilter):
    """
    Filter matches using Lowe's ratio test
    """

    def __init__(self, ratio=0.5):
        self.ratio = ratio
    
    def filter_matches(self, matches, keypoints_left=None, keypoints_right=None):
        filtered_matches = []
        for m, n in matches:
            if m.distance < self.ratio * n.distance:
                filtered_matches.append(m)
        return filtered_matches

class RANSACFilter(FeatureMatchFilter):
    """
    Filter matches using RANSAC (Random Sample Consensus) algorithm
    """
    def __init__(self, min_matches=8, reproj_thresh=4.0):
        self.min_matches = min_matches
        self.reproj_thresh = reproj_thresh
    
    def filter_matches(self, matches, keypoints_left, keypoints_right):
        if len(matches) < self.min_matches:
            return []
        src_pts = np.float32([keypoints_left[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints_right[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        _, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, self.reproj_thresh)
        mask = mask.ravel().tolist()
        filtered_matches = [m for i, m in enumerate(matches) if mask[i]]
        return filtered_matches


class StereoProjection:
    def __init__(self, yaml_file):
        """
        Initializes the StereoProjection class by loading camera intrinsics and extrinsics from a YAML file.
        """
        self.yaml_file = yaml_file
        self.K0 = None  # Left camera intrinsic matrix
        self.K1 = None  # Right camera intrinsic matrix
        self.R = None  # Rotation matrix (left to right camera)
        self.t = None  # Translation vector (left to right camera)
        self.P0 = None  # Left camera projection matrix
        self.P1 = None  # Right camera projection matrix
        self.dist_coeffs0 = None  # Distortion coefficients for left camera
        self.dist_coeffs1 = None  # Distortion coefficients for right camera

        # Load the YAML data and compute projection matrices
        self.load_from_yaml()

    def load_from_yaml(self):
        """
        Reads the YAML calibration file and extracts the necessary camera parameters.
        """
        with open(self.yaml_file, "r") as file:
            data = yaml.safe_load(file)

        # Extract camera intrinsics
        K0_values = data["cam0"]["intrinsics"]
        K1_values = data["cam1"]["intrinsics"]

        self.K0 = np.array([[K0_values[0], 0, K0_values[2]],
                            [0, K0_values[1], K0_values[3]],
                            [0, 0, 1]])

        self.K1 = np.array([[K1_values[0], 0, K1_values[2]],
                            [0, K1_values[1], K1_values[3]],
                            [0, 0, 1]])

        self.dist_coeffs0 = np.array(data["cam0"]["distortion_coeffs"])
        self.dist_coeffs1 = np.array(data["cam1"]["distortion_coeffs"])

        # Extract extrinsic parameters (Rotation & Translation)
        T = np.array(data["cam1"]["T_cn_cnm1"])  # Transformation matrix
        self.R = T[:3, :3]  # First 3x3 block is the rotation matrix
        self.t = T[:3, 3].reshape(3, 1)  # Last column is the translation vector

        # Compute projection matrices
        self.P0 = self.K0 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P0 = K0 * [I | 0]
        Rt = np.hstack((self.R, self.t))  # Combine R and t
        self.P1 = self.K1 @ Rt  # P1 = K1 * [R | t]

    #Function to calculate and visualize distortion correction
    def undistort_image(self, image, camera_matrix, dist_coeffs, use_fisheye=False):
        """
        Undistort an image using the camera matrix and distortion coefficients.

        :param image: The image to undistort.
        :param camera_matrix: The camera matrix (intrinsic parameters).
        :param dist_coeffs: The distortion coefficients.
        :param use_fisheye: Boolean flag to indicate if fisheye model should be used.
        :return: The undistorted image.
        """
        h, w = image.shape[:2]
        if use_fisheye:
            new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                camera_matrix, dist_coeffs, (w, h), np.eye(3), balance=1)
            undistorted_image = cv2.fisheye.undistortImage(image, camera_matrix, dist_coeffs, Knew=new_camera_matrix)
        else:
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
            undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        return undistorted_image


    def triangulate_points(self, points_left, points_right, use_normalized_projection=False):
        """
        Triangulates 3D points from corresponding feature points in left and right images.

        :param points_left: Nx2 array of 2D points in the left image.
        :param points_right: Nx2 array of 2D points in the right image.
        :param use_normalized_projection: If True, use identity intrinsic matrix in projection.
        :return: Nx3 array of triangulated 3D points in the left camera frame.
        """
        points_left = np.array(points_left, dtype=np.float32)
        points_right = np.array(points_right, dtype=np.float32)

        # Convert points to the shape expected by OpenCV functions
        points_left = points_left.reshape(-1, 1, 2)
        points_right = points_right.reshape(-1, 1, 2)

        # Undistort and normalize the points
        points_left_norm = cv2.undistortPoints(points_left, self.K0, self.dist_coeffs0, R=None)
        points_right_norm = cv2.undistortPoints(points_right, self.K1, self.dist_coeffs1, R=None)

        if use_normalized_projection:
            P0 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Use identity matrix for intrinsics
            P1 = np.hstack((self.R, self.t))
        else:
            P0 = self.P0
            P1 = self.P1

        # Perform triangulation
        points_4D = cv2.triangulatePoints(P0, P1, points_left_norm, points_right_norm)

        # Convert from homogeneous to Euclidean coordinates
        points_3D = points_4D[:3] / points_4D[3]

        return points_3D.T  # Return Nx3 array

    def print_matrices(self):
        """
        Prints the intrinsic and projection matrices.
        """
        print("\nIntrinsic Matrix (K0 - Left Camera):\n", self.K0)
        print("\nIntrinsic Matrix (K1 - Right Camera):\n", self.K1)
        print("\nRotation Matrix (R - Left to Right Camera):\n", self.R)
        print("\nTranslation Vector (t - Left to Right Camera):\n", self.t)
        print("\nProjection Matrix for Left Camera (P0):\n", self.P0)
        print("\nProjection Matrix for Right Camera (P1):\n", self.P1)


def show_keypoints(image1, keypoints1, image2, keypoints2, draw_rich_keypoints=False):
    """
    Display two images side by side with keypoints plotted on them.

    :param image1: The first image on which to draw keypoints.
    :param keypoints1: Keypoints detected in the first image.
    :param image2: The second image on which to draw keypoints.
    :param keypoints2: Keypoints detected in the second image.
    :param draw_rich_keypoints: Whether to draw rich keypoints (with size and orientation).
    """
    # Draw keypoints on the first image
    if draw_rich_keypoints:
        image_with_keypoints1 = cv2.drawKeypoints(image1, keypoints1, None,
                                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                                  color=(0, 255, 0))  # Green keypoints
    else:
        image_with_keypoints1 = cv2.drawKeypoints(image1, keypoints1, None,
                                                  color=(0, 255, 0))  # Green keypoints

    # Draw keypoints on the second image
    if draw_rich_keypoints:
        image_with_keypoints2 = cv2.drawKeypoints(image2, keypoints2, None,
                                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                                  color=(0, 255, 0))  # Green keypoints
    else:
        image_with_keypoints2 = cv2.drawKeypoints(image2, keypoints2, None,
                                                  color=(0, 255, 0))  # Green keypoints

    # Create a subplot to show the two images side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot the first image with keypoints
    axes[0].imshow(image_with_keypoints1, cmap="gray")
    axes[0].set_title("Image 1 with Keypoints")
    axes[0].axis("off")

    # Plot the second image with keypoints
    axes[1].imshow(image_with_keypoints2, cmap="gray")
    axes[1].set_title("Image 2 with Keypoints")
    axes[1].axis("off")

    # Display the plot
    plt.tight_layout()
    plt.show()

def show_points(image1, points1, image2, points2, point_cloud):
    """
    Display two images side by side with points plotted on them, and a 3D point cloud.

    :param image1: The first image on which to draw points.
    :param points1: List of points (tuples) for the first image.
    :param image2: The second image on which to draw points.
    :param points2: List of points (tuples) for the second image.
    :param point_cloud: A list or array of 3D points (tuples or np.array).
    """
    # Copy images to avoid modifying the original
    image_with_points1 = image1.copy()
    image_with_points2 = image2.copy()
    
    # Combine points with point cloud and sort by x-coordinate in left image
    combined_data = [(point[0], point, points2[i], point_cloud[i]) 
                    for i, point in enumerate(points1)]
    combined_data.sort(key=lambda x: float(x[0]))  # Sort by x-coordinate as float
    
    # Unpack the sorted data
    sorted_points1 = [data[1] for data in combined_data]
    sorted_points2 = [data[2] for data in combined_data]
    sorted_point_cloud = np.array([data[3] for data in combined_data])
    
    # Generate a colormap with unique colors for each point
    num_points = len(sorted_points1)
    cmap = plt.cm.get_cmap('hsv', num_points)
    colors = [cmap(i)[:3] for i in range(num_points)]
    
    # Convert colors from 0-1 range to 0-255 range for OpenCV
    opencv_colors = [(int(b*255), int(g*255), int(r*255)) for r, g, b in colors]
    
    # Draw points on the first image with unique colors
    for i, point in enumerate(sorted_points1):
        x, y = point
        color = opencv_colors[i]
        cv2.circle(image_with_points1, (int(x), int(y)), 3, color, -1)
    
    # Draw points on the second image with the same colors
    for i, point in enumerate(sorted_points2):
        x, y = point
        color = opencv_colors[i]
        cv2.circle(image_with_points2, (int(x), int(y)), 3, color, -1)
    
    # Create a subplot to show the two images side by side, and a 3D plot for the point cloud
    fig = plt.figure(figsize=(18, 6))
    
    # Plot the first image with points
    ax1 = fig.add_subplot(131)
    ax1.imshow(cv2.cvtColor(image_with_points1, cv2.COLOR_BGR2RGB))
    ax1.set_title("Image 1 with Points")
    ax1.axis("off")
    
    # Plot the second image with points
    ax2 = fig.add_subplot(132)
    ax2.imshow(cv2.cvtColor(image_with_points2, cv2.COLOR_BGR2RGB))
    ax2.set_title("Image 2 with Points")
    ax2.axis("off")
    
    # Plot the 3D point cloud with matching colors
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Extract X, Y, Z coordinates from the sorted point cloud
    X = sorted_point_cloud[:, 0]
    Y = sorted_point_cloud[:, 1]
    Z = sorted_point_cloud[:, 2]

    
    # Plot each point with its corresponding color
    for i in range(len(X)):
        ax3.scatter(X[i], Y[i], Z[i], c=[colors[i]], marker='o', s=10)
    
    # Plot the origin (0, 0, 0) as a unique point
    ax3.scatter(0, 0, 0, c='k', marker='^', s=100, label='Origin')  # Black triangle

    #visualize camera reference frame
    ax3.plot([0, 100], [0, 0], [0, 0], c='r')
    ax3.plot([0, 0], [0, 100], [0, 0], c='g')
    ax3.plot([0, 0], [0, 0], [0, 100], c='k')
    
    # Set labels for the 3D plot
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title("3D Point Cloud")
    
    # Determine axis limits dynamically
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Display the plot
    plt.tight_layout()
    plt.show()


def show_matches(image1, image2, matches, keypoints1, keypoints2):
    """
    Show the matching keypoints between two images using the matches list.

    :param image1: The first image.
    :param image2: The second image.
    :param matches: List of good matches (from FLANN or other matchers).
    :param keypoints1: Keypoints detected in the first image.
    :param keypoints2: Keypoints detected in the second image.
    """
    # Draw matches on the images
    image_with_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the image using Matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(image_with_matches)
    plt.title("Matches between Image 1 and Image 2")
    plt.axis("off")
    plt.show()

def extract_points_from_matches(filtered_matches, keypoints_left, keypoints_right):
    left_points = []
    right_points = []

    for m in filtered_matches:
        left_points.append(keypoints_left[m.queryIdx].pt)
        right_points.append(keypoints_right[m.trainIdx].pt)

    return left_points, right_points


def plot_3d_point_cloud(points_3d):
    """
    Plot the 3D points using matplotlib.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', marker='o')
    # Plot the origin (0, 0, 0) as a unique point
    ax.scatter(0, 0, 0, c='b', marker='^', s=100, label='Origin')  # Blue triangle, size 100

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


#Find the transformation matrix between the two frames using the two point clouds
def find_transformation(src_points, dst_points):
    """
    Find the transformation matrix between two sets of 3D points using SVD.
    
    :param src_points: Source points (Nx3 array)
    :param dst_points: Destination points (Nx3 array)
    :return: 4x4 transformation matrix
    """
    # Calculate centroids
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)
    
    # Center the points
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid
    
    # Calculate rotation using SVD
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation matrix (handle reflection case)
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    
    # Calculate translation
    t = dst_centroid - R @ src_centroid
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    
    return T

def update_camera_pose(T_relative):
    global T_world_to_current
    T_world_to_current = T_world_to_current @ T_relative  # Matrix multiplication
    return T_world_to_current

if __name__ == "__main__":
    # load in a stereo pair and two sequential flames
    frame1 = interface.VisionInputFrame("analysis/image_0_0.png", "analysis/image_1_0.png")
    frame2 = interface.VisionInputFrame("analysis/image_0_1.png", "analysis/image_1_1.png")

    left1, right1 = load_images(frame1)
    left2, right2 = load_images(frame2)

    left1_gs, right1_gs = preprocess_images(left1, right1)
    left2_gs, right2_gs = preprocess_images(left2, right2)

    # ORB doesn't work without transforming the uint8 descriptors to uint32
    # SIFT and AKAZE return uint32s by default
    extractor = SIFTFeatureExtractor()

    l1_keypoints, l1_descriptors = extractor.extract_features(left1_gs)
    r1_keypoints, r1_descriptors = extractor.extract_features(right1_gs)
    l2_keypoints, l2_descriptors = extractor.extract_features(left2_gs)
    r2_keypoints, r2_descriptors = extractor.extract_features(right2_gs)

    matcher = FLANNMatcher()
    matches1 = matcher.match_features(l1_descriptors, r1_descriptors)
    matches2 = matcher.match_features(l2_descriptors, r2_descriptors)

    # First filter matches between left-right pairs using RANSAC
    filter = RANSACFilter(min_matches=12, reproj_thresh=1)
    filtered_matches1 = filter.filter_matches(matches1, l1_keypoints, r1_keypoints)
    filtered_matches2 = filter.filter_matches(matches2, l2_keypoints, r2_keypoints)

    # Now match features between left images of frame 1 and 2
    matches_between_frames = matcher.match_features(l1_descriptors, l2_descriptors)
    filtered_matches_between_frames = filter.filter_matches(matches_between_frames, l1_keypoints, l2_keypoints)

    # Create mapping between frames
    matches_dict = {}
    for m in filtered_matches_between_frames:
        idx1 = m.queryIdx  # Index in frame 1 left image
        idx2 = m.trainIdx  # Index in frame 2 left image
        matches_dict[idx1] = idx2

    # Filter to keep only matches present in all four images
    consistent_matches1 = []
    consistent_matches2 = []
    
    for m1 in filtered_matches1:
        left_idx = m1.queryIdx
        if left_idx in matches_dict:  # If this point has a match in frame 2
            frame2_left_idx = matches_dict[left_idx]
            # Find corresponding match in filtered_matches2
            for m2 in filtered_matches2:
                if m2.queryIdx == frame2_left_idx:
                    consistent_matches1.append(m1)
                    consistent_matches2.append(m2)
                    break

    filtered_matches1 = consistent_matches1
    filtered_matches2 = consistent_matches2

    # show_matches(left1, right1, filtered_matches1, l1_keypoints, r1_keypoints)

    StereoPair = StereoProjection("analysis/camchain-..indoor_forward_calib_snapdragon_cam.yaml")

    #Visualize the undistorted images
    left1_undistorted = StereoPair.undistort_image(left1, StereoPair.K0, StereoPair.dist_coeffs0, use_fisheye=True)
    right1_undistorted = StereoPair.undistort_image(right1, StereoPair.K1, StereoPair.dist_coeffs1, use_fisheye=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(left1_undistorted, cmap="gray")
    ax1.set_title("Undistorted Left Image")
    ax1.axis("off")
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(right1_undistorted, cmap="gray")
    ax2.set_title("Undistorted Right Image")
    ax2.axis("off")

    plt.show()

    pl1, pr1 = extract_points_from_matches(consistent_matches1, l1_keypoints, r1_keypoints)
    points1 = StereoPair.triangulate_points(np.array(pl1), np.array(pr1), use_normalized_projection=True)

    pl2, pr2 = extract_points_from_matches(consistent_matches2, l2_keypoints, r2_keypoints)
    points2 = StereoPair.triangulate_points(np.array(pl2), np.array(pr2), use_normalized_projection=True)

    # Find transformation between frame 1 and frame 2
    transformation = find_transformation(points1, points2)
    print("\nTransformation Matrix between Frame 1 and Frame 2:")
    print(transformation)

    T_cam_imu0 = np.linalg.inv(np.array([[-0.02822879, 0.01440125, 0.99949774, 0.00110212],
                                         [-0.99960149, -0.00041887, -0.02822568, 0.02170142],
                                         [ 0.00001218, -0.99989621, 0.01440734, -0.00005928],
                                         [ 0, 0, 0, 1]]))

    # Initial transformation: Camera's z-axis aligned with world's x-axis, shifted up 2 units in world z-axis
    T_world_to_current = np.array([
        [-0.6308871878503666, 0.775778177185825, 0.012230127084176465, 7.60668919163082],  # Camera z-axis -> World x-axis
        [ -0.7758619249210552, -0.6307085060763604, -0.015654194987939865, 0.246246215953204],  # Camera y-axis -> World y-axis
        [-0.004430537670670494, -0.019364920995560968, 0.9998026656149626, -0.880815170689076], # Camera x-axis -> World -z-axis
        [0, 0, 0, 1]   # Homogeneous coordinates
    ])

    print("\nNew Transformation Matrix (World to Current Frame):")
    T_world_to_current2 = update_camera_pose(transformation)
    print(T_world_to_current2)

    """
    -0.630914015914675 0.7757556783140641 0.012273226385246232 7.60666536072954
    -0.7758406136151782 -0.6307366795317234 -0.015575087753203145 0.246270702230251
    -0.004341288707415834 -0.0193486086523643 0.9998033729466893 -0.880807258137499
    0.0 0.0 0.0 1.0
    """
    
    show_points(left1, pl1, right1, pr1, points1)

    #transform points1 to the world frame
    points1_world = np.dot(T_cam_imu0, np.dot(T_world_to_current, np.vstack((points1.T, np.ones(points1.shape[0])))))
    points1_world = points1_world[:3].T

    #Get old camera pos in world coords
    old_cam_pos = np.dot(T_cam_imu0, np.dot(T_world_to_current, np.array([0, 0, 0, 1])))
    old_cam_pos = old_cam_pos[:3]

    #transform points2 to the new world frame
    points2_world = np.dot(T_cam_imu0, np.dot(T_world_to_current2, np.vstack((points2.T, np.ones(points2.shape[0])))))
    points2_world = points2_world[:3].T

    #Get new camera pos in world coords
    new_cam_pos = np.dot(T_cam_imu0, np.dot(T_world_to_current2, np.array([0, 0, 0, 1])))
    new_cam_pos = new_cam_pos[:3]

    #plot the old points in the world frame in red, and the new ones in blue in the same plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points1_world[:, 0], points1_world[:, 1], points1_world[:, 2], c='r', marker='o')
    ax.scatter(points2_world[:, 0], points2_world[:, 1], points2_world[:, 2], c='b', marker='o')

    #plot old camera pos in green
    ax.scatter(old_cam_pos[0], old_cam_pos[1], old_cam_pos[2], c='g', marker='v', s=100)

    #plot new camera pos in purple
    ax.scatter(new_cam_pos[0], new_cam_pos[1], new_cam_pos[2], c='purple', marker='^', facecolors='none', s=100)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()




#List of images (L & R)1, (L & R)2, (L & R)3, ... (L & R)n
# Find common features shared within the set
# Overlapping windows of frames and pick features you can find in all frames

#Once identified, go back into LR pairs and find 3D point cloud for those features

#Classes:
#Image Pair
#Feature
#Point Cloud - List of points and those points are associated with features