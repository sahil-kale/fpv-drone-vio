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
        return matches


#Filter matches to remove outliers
class FeatureMatchFilter(ABC):
    """
    Abstract class for feature match filtering
    """
    @abstractmethod
    def filter_matches(self, matches):
        """
        Take matches and return filtered matches
        """
        pass

class RatioTestFilter(FeatureMatchFilter):
    """
    Filter matches using Lowe's ratio test
    """

    def __init__(self, ratio=0.75):
        self.ratio = ratio
    
    def filter_matches(self, matches):
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
        self.ransac = cv2.RANSACReprojectionSolver(reproj_thresh, min_matches)
    
    def filter_matches(self, matches, keypoints_left, keypoints_right):
        if len(matches) < self.min_matches:
            return []
        src_pts = np.float32([keypoints_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        mask = self.ransac.estimate(src_pts, dst_pts, None)
        filtered_matches = [m for i, m in enumerate(matches) if mask[i] == 1]
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

        # Extract extrinsic parameters (Rotation & Translation)
        T = np.array(data["cam1"]["T_cn_cnm1"])  # Transformation matrix
        self.R = T[:3, :3]  # First 3x3 block is the rotation matrix
        self.t = T[:3, 3].reshape(3, 1)  # Last column is the translation vector

        # Compute projection matrices
        self.P0 = np.hstack((self.K0, np.zeros((3, 1))))  # P0 = K0 * [I | 0]
        Rt = np.hstack((self.R, self.t))  # Combine R and t
        self.P1 = self.K1 @ Rt  # P1 = K1 * [R | t]

    def triangulate_points(self, points_left, points_right):
        """
        Triangulates 3D points from corresponding feature points in left and right images.

        :param points_left: Nx2 array of 2D points in the left image.
        :param points_right: Nx2 array of 2D points in the right image.
        :return: Nx3 array of triangulated 3D points in the left camera frame.
        """
        # Convert to homogeneous coordinates (adding 1s as the third dimension)
        points_left_hom = np.vstack((points_left.T, np.ones((1, points_left.shape[0]))))
        points_right_hom = np.vstack((points_right.T, np.ones((1, points_right.shape[0]))))

        # Perform triangulation
        points_4D = cv2.triangulatePoints(self.P0, self.P1, points_left_hom[:2], points_right_hom[:2])

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

def show_keypoints(keypoints, image, draw_rich_keypoints = False):
    if draw_rich_keypoints:
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None,
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                             color=(0, 255, 0))  # Green keypoints
    else:
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None,
                                                 color=(0, 255, 0))  # Green keypoints

    # Show the image using Matplotlib (better for noteb ooks)
    plt.figure(figsize=(8, 6))
    plt.imshow(image_with_keypoints, cmap="gray")
    plt.title("Detected Keypoints")
    plt.axis("off")
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

    # Draw points on the first image
    for point in points1:
        x, y = point
        # Draw a smaller filled circle for each point
        cv2.circle(image_with_points1, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green filled circle

    # Draw points on the second image
    for point in points2:
        x, y = point
        # Draw a smaller filled circle for each point
        cv2.circle(image_with_points2, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green filled circle

    # Create a subplot to show the two images side by side, and a 3D plot for the point cloud
    fig = plt.figure(figsize=(18, 6))

    # Plot the first image with points
    ax1 = fig.add_subplot(131)
    ax1.imshow(image_with_points1, cmap="gray")
    ax1.set_title("Image 1 with Points")
    ax1.axis("off")

    # Plot the second image with points
    ax2 = fig.add_subplot(132)
    ax2.imshow(image_with_points2, cmap="gray")
    ax2.set_title("Image 2 with Points")
    ax2.axis("off")

    # Plot the 3D point cloud
    ax3 = fig.add_subplot(133, projection='3d')
    point_cloud = np.array(point_cloud)

    # Extract X, Y, Z coordinates from the point cloud
    X = point_cloud[:, 0]
    Y = point_cloud[:, 1]
    Z = point_cloud[:, 2]

    # Plot the points in 3D space
    ax3.scatter(X, Y, Z, c='r', marker='o')
    # Plot the origin (0, 0, 0) as a unique point
    ax3.scatter(0, 0, 0, c='b', marker='^', s=100, label='Origin')  # Blue triangle, size 100

    # Set labels for the 3D plot
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title("3D Point Cloud")

    # Display the plot
    plt.tight_layout()
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

if __name__ == "__main__":
    # load in a stereo pair and two sequential flames
    frame1 = interface.VisionInputFrame("image_0_0.png", "image_1_0.png")
    frame2 = interface.VisionInputFrame("image_0_1.png", "image_1_1.png")

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

    # Ransac filter is not working since CV2 does not have a class called RANSACReprojectionSolver
    filter = RatioTestFilter()
    filtered_matches1 = filter.filter_matches(matches1)
    filtered_matches2 = filter.filter_matches(matches2)

    StereoPair = StereoProjection("camchain-..indoor_forward_calib_snapdragon_cam.yaml")

    pl1, pr1 = extract_points_from_matches(filtered_matches1, l1_keypoints, r1_keypoints)
    points1 = StereoPair.triangulate_points(np.array(pl1), np.array(pr1))

    pl2, pr2 = extract_points_from_matches(filtered_matches2, l2_keypoints, r2_keypoints)
    points2 = StereoPair.triangulate_points(np.array(pl2), np.array(pr2))

    # plot_3d_point_cloud(points1)
    show_points(left1, pl1, right2, pl2, points1)

    # plot_3d_point_cloud(points2)


#List of images (L & R)1, (L & R)2, (L & R)3, ... (L & R)n
# Find common features shared within the set
# Overlapping windows of frames and pick features you can find in all frames

#Once identified, go back into LR pairs and find 3D point cloud for those features

#Classes:
#Image Pair
#Feature
#Point Cloud - List of points and those points are associated with features