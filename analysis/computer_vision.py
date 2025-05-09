import interface
import cv2
import numpy as np
from abc import ABC, abstractmethod
import yaml
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp


#Load left and right images from the interface
def load_images(input:interface.VisionInputFrame):
    left_image = cv2.imread(input.get_image_left_path())
    right_image = cv2.imread(input.get_image_right_path())
    return left_image, right_image

#Preprocess images
def preprocess_images(left_image, right_image):
    #Convert to grayscale
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    left_gray = cv2.bilateralFilter(left_gray, d=9, sigmaColor=75, sigmaSpace=75)
    right_gray = cv2.bilateralFilter(right_gray, d=9, sigmaColor=75, sigmaSpace=75)

    return left_gray, right_gray

# abstract feature extractor class
# makes it easier to try different algorithms in testing
class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, image):
        pass


class ORBFeatureExtractor(FeatureExtractor):
    def __init__(self, n_features=500):
        self.n_features = n_features
        self.orb = cv2.ORB_create(nfeatures=n_features)
    
    def extract_features(self, image):
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors.astype(np.float32)  # Ensure descriptors are float32 for matching


class SIFTFeatureExtractor(FeatureExtractor):
    def __init__(self, n_features=3000):
        self.n_features = n_features
        self.sift = cv2.SIFT_create(nfeatures=n_features)
    
    def extract_features(self, image):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return keypoints, descriptors


class AKAZEFeatureExtractor(FeatureExtractor):
    def __init__(self, n_features=500):
        self.n_features = n_features
        self.akaze = cv2.AKAZE_create()
    
    def extract_features(self, image):
        keypoints, descriptors = self.akaze.detectAndCompute(image, None)
        return keypoints, descriptors.astype(np.float32)  # Ensure descriptors are float32 for matching

# abstract feature matcher class
# makes it easier to test different feature matchers
class FeatureMatcher(ABC):
    @abstractmethod
    def match_features(self, left_descriptors, right_descriptors):
        pass


class BFMatcher(FeatureMatcher):
    # brute force matcher
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def match_features(self, left_descriptors, right_descriptors):
        matches = self.bf.match(left_descriptors, right_descriptors)
        return matches


class FLANNMatcher(FeatureMatcher):
    def __init__(self, trees=10, checks=100):
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict(checks=60))
    
    def match_features(self, left_descriptors, right_descriptors):
        matches = self.flann.knnMatch(left_descriptors, right_descriptors, k=2)
        # Convert matches from list of lists to list of DMatch objects
        matches = [m[0] for m in matches if len(m) == 2]
        return matches


# abstract class for matched feature filtering
class FeatureMatchFilter(ABC):
    @abstractmethod
    def filter_matches(self, matches, keypoints_left=None, keypoints_right=None):
        pass


class RatioTestFilter(FeatureMatchFilter):
    # Lowe's ratio test
    def __init__(self, ratio=0.5):
        self.ratio = ratio
    
    def filter_matches(self, matches, keypoints_left=None, keypoints_right=None):
        filtered_matches = []
        for m, n in matches:
            if m.distance < self.ratio * n.distance:
                filtered_matches.append(m)
        return filtered_matches


class RANSACFilter(FeatureMatchFilter):
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
    def __init__(self, yaml_file, distortion=None):
        # Initializes the StereoProjection class by loading camera intrinsics and
        # extrinsics from the YAML calibration file.
        self.yaml_file = yaml_file
        self.K0 = None  # Left camera intrinsic matrix
        self.K1 = None  # Right camera intrinsic matrix
        self.R = None  # Rotation matrix (left to right camera)
        self.t = None  # Translation vector (left to right camera)
        self.P0 = None  # Left camera projection matrix
        self.P1 = None  # Right camera projection matrix
        self.dist_coeffs0 = None  # Distortion coefficients for left camera
        self.dist_coeffs1 = None  # Distortion coefficients for right camera
        
        self.distortion_type = distortion

        # Load the YAML data and compute projection matrices
        self.load_from_yaml()

    def load_from_yaml(self):
        # loads the camera parameters from the yaml file
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

        self.resX0 = data["cam0"]["resolution"][0]
        self.resY0 = data["cam0"]["resolution"][1]

        self.resX1 = data["cam1"]["resolution"][0]
        self.resY1 = data["cam1"]["resolution"][1]

        # Extract extrinsic parameters (Rotation & Translation)
        T = np.array(data["cam1"]["T_cn_cnm1"])  # Transformation matrix
        self.R = T[:3, :3]  # First 3x3 block is the rotation matrix
        self.t = T[:3, 3].reshape(3, 1)  # Last column is the translation vector

        #Calculate projection matrices
        self.calculate_projection_matrices()

    def calculate_projection_matrices(self):
        if self.distortion_type == "fisheye":
            # Use fisheye model for distortion correction
            self.dist_coeffs0 = np.array([self.dist_coeffs0[0], self.dist_coeffs0[1], self.dist_coeffs0[2], self.dist_coeffs0[3]])
            self.dist_coeffs1 = np.array([self.dist_coeffs1[0], self.dist_coeffs1[1], self.dist_coeffs1[2], self.dist_coeffs1[3]])

            self.K0_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K0, self.dist_coeffs0, (self.resX0, self.resY0), np.eye(3), balance=1)
            self.K1_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K1, self.dist_coeffs1, (self.resX1, self.resY1), np.eye(3), balance=1)
            
            self.P0 = self.K0_new @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P0 = K0_new * [I | 0]
            self.P1 = self.K1_new @ np.hstack((self.R, self.t))  # P1 = K1_new * [R | t]

        elif self.distortion_type == "normal":
            # Use standard distortion model
            self.dist_coeffs0 = np.array([self.dist_coeffs0[0], self.dist_coeffs0[1], self.dist_coeffs0[2], self.dist_coeffs0[3], 0])
            self.dist_coeffs1 = np.array([self.dist_coeffs1[0], self.dist_coeffs1[1], self.dist_coeffs1[2], self.dist_coeffs1[3], 0])

            self.K0_new, _ = cv2.getOptimalNewCameraMatrix(self.K0, self.dist_coeffs0, (self.resX0, self.resY0), 1, (self.resX0, self.resY0))
            self.K1_new, _ = cv2.getOptimalNewCameraMatrix(self.K1, self.dist_coeffs1, (self.resX1, self.resY1), 1, (self.resX1, self.resY1))

        else:
            # Don't use distortion
            self.P0 = self.K0 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P0 = K0 * [I | 0]
            self.P1 = self.K1 @ np.hstack((self.R, self.t))  # P1 = K1 * [R | t]


    #Function to calculate and visualize distortion correction
    def undistort_image(self, image, camera_matrix, dist_coeffs, camera="left"):
        if self.distortion_type == None:
            return image
        else:
            h, w = image.shape[:2]
            if camera=="left":
                undistortion_matrix = self.K0_new
            else:
                undistortion_matrix = self.K1_new
            
            if self.distortion_type == "fisheye":
                undistorted_image = cv2.fisheye.undistortImage(image, camera_matrix, dist_coeffs, Knew=undistortion_matrix)
            else:
                undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, Knew=undistortion_matrix)
            return undistorted_image


    def triangulate_points(self, points_left, points_right, use_normalized_projection=False):
        points_left = np.array(points_left, dtype=np.float32)
        points_right = np.array(points_right, dtype=np.float32)

        # Convert points to the shape expected by OpenCV functions
        points_left = points_left.reshape(-1, 1, 2)
        points_right = points_right.reshape(-1, 1, 2)

        # Normalize the undistorted points
        points_left_normalized = cv2.convertPointsToHomogeneous(points_left).reshape(-1, 3).T
        points_right_normalized = cv2.convertPointsToHomogeneous(points_right).reshape(-1, 3).T

        # Perform triangulation
        points_4D = cv2.triangulatePoints(self.P0, self.P1, points_left_normalized[:2], points_right_normalized[:2])

        points_3D = points_4D[:3] / points_4D[3]  # Convert from homogeneous to Euclidean
        return points_3D.T


    def plot_undistorted_points(self, points_left, points_right, image_left, image_right, axes=None, animate=False):

        #Find the undistorted images
        image_left_undistorted = self.undistort_image(image_left, self.K0, self.dist_coeffs0, camera="left")
        image_right_undistorted = self.undistort_image(image_right, self.K1, self.dist_coeffs1, camera="right")

        #Undistort the points
        points_left_undistorted = self.undistort_points(points_left, camera="left")
        points_right_undistorted = self.undistort_points(points_right, camera="right")

        # Convert to keypoints
        points_left_undistorted = [cv2.KeyPoint(p[0], p[1], 1) for p in points_left_undistorted]
        points_right_undistorted = [cv2.KeyPoint(p[0], p[1], 1) for p in points_right_undistorted]

        if animate:
            animate_keypoints(image_left_undistorted, points_left_undistorted, image_right_undistorted, points_right_undistorted, axes)
        else:
            #Plot the points on the images
            show_keypoints(image_left_undistorted, points_left_undistorted, image_right_undistorted, points_right_undistorted)

    def undistort_points(self, points, camera="left"):
        if camera == "left":
            undistortion_matrix = self.K0_new
            camera_matrix = self.K0
            distortion_coefficients = self.dist_coeffs0
        else:
            undistortion_matrix = self.K1_new
            camera_matrix = self.K1
            distortion_coefficients = self.dist_coeffs1

        if self.distortion_type == "fisheye":
            points_undistorted = cv2.fisheye.undistortPoints(
                np.array(points).reshape(-1, 1, 2), camera_matrix, distortion_coefficients
            ).reshape(-1, 2)  # Ensure shape (N, 2)
        elif self.distortion_type == "normal":
            points_undistorted = cv2.undistortPoints(
                np.array(points).reshape(-1, 1, 2), camera_matrix, distortion_coefficients, P=undistortion_matrix
            ).reshape(-1, 2)

        # Convert back to pixel coordinates
        points_undistorted = np.squeeze(points_undistorted)  # Remove unnecessary dimensions

        # Ensure the points are in pixel coordinates
        points_undistorted = np.dot(undistortion_matrix, np.vstack((points_undistorted.T, np.ones(points_undistorted.shape[0]))))

        points_undistorted = points_undistorted[:2].T  # Extract x, y
        
        return points_undistorted

    def print_matrices(self):
        print("\nIntrinsic Matrix (K0 - Left Camera):\n", self.K0)
        print("\nIntrinsic Matrix (K1 - Right Camera):\n", self.K1)
        print("\nRotation Matrix (R - Left to Right Camera):\n", self.R)
        print("\nTranslation Vector (t - Left to Right Camera):\n", self.t)
        print("\nProjection Matrix for Left Camera (P0):\n", self.P0)
        print("\nProjection Matrix for Right Camera (P1):\n", self.P1)


def show_keypoints(image1, keypoints1, image2, keypoints2, draw_rich_keypoints=False):
    if draw_rich_keypoints:
        image_with_keypoints1 = cv2.drawKeypoints(image1, keypoints1, None,
                                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                                  color=(0, 255, 0))
    else:
        image_with_keypoints1 = cv2.drawKeypoints(image1, keypoints1, None,
                                                  color=(0, 255, 0))
    if draw_rich_keypoints:
        image_with_keypoints2 = cv2.drawKeypoints(image2, keypoints2, None,
                                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                                  color=(0, 255, 0))
    else:
        image_with_keypoints2 = cv2.drawKeypoints(image2, keypoints2, None,
                                                  color=(0, 255, 0))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(image_with_keypoints1, cmap="gray")
    axes[0].set_title("Image 1 with Keypoints")
    axes[0].axis("off")

    axes[1].imshow(image_with_keypoints2, cmap="gray")
    axes[1].set_title("Image 2 with Keypoints")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def animate_keypoints(image1, keypoints1, image2, keypoints2, axes, draw_rich_keypoints=False):
    image_with_keypoints1 = cv2.drawKeypoints(
        image1, keypoints1, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if draw_rich_keypoints else 0,
        color=(0, 255, 0)
    )

    image_with_keypoints2 = cv2.drawKeypoints(
        image2, keypoints2, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if draw_rich_keypoints else 0,
        color=(0, 255, 0)
    )

    if not hasattr(axes[0], "img"):
        axes[0].img = axes[0].imshow(image_with_keypoints1, cmap="gray")
        axes[1].img = axes[1].imshow(image_with_keypoints2, cmap="gray")
    else:
        axes[0].img.set_data(image_with_keypoints1)
        axes[1].img.set_data(image_with_keypoints2)

    plt.draw()
    plt.pause(0.05)

def show_points(image1, points1, image2, points2, point_cloud):
    image_with_points1 = image1.copy()
    image_with_points2 = image2.copy()
    
    combined_data = [(point[0], point, points2[i], point_cloud[i])
                    for i, point in enumerate(points1)]
    combined_data.sort(key=lambda x: float(x[0]))  # Sort by x-coordinate as float
    
    sorted_points1 = [data[1] for data in combined_data]
    sorted_points2 = [data[2] for data in combined_data]
    sorted_point_cloud = np.array([data[3] for data in combined_data])
    
    num_points = len(sorted_points1)
    cmap = plt.cm.get_cmap('hsv', num_points)
    colors = [cmap(i)[:3] for i in range(num_points)]
    
    opencv_colors = [(int(b*255), int(g*255), int(r*255)) for r, g, b in colors]
    
    for i, point in enumerate(sorted_points1):
        x, y = point
        color = opencv_colors[i]
        cv2.circle(image_with_points1, (int(x), int(y)), 3, color, -1)
    
    for i, point in enumerate(sorted_points2):
        x, y = point
        color = opencv_colors[i]
        cv2.circle(image_with_points2, (int(x), int(y)), 3, color, -1)
    
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(131)
    ax1.imshow(cv2.cvtColor(image_with_points1, cv2.COLOR_BGR2RGB))
    ax1.set_title("Image 1 with Points")
    ax1.axis("off")
    
    ax2 = fig.add_subplot(132)
    ax2.imshow(cv2.cvtColor(image_with_points2, cv2.COLOR_BGR2RGB))
    ax2.set_title("Image 2 with Points")
    ax2.axis("off")
    
    ax3 = fig.add_subplot(133, projection='3d')
    
    X = sorted_point_cloud[:, 0]
    Y = sorted_point_cloud[:, 1]
    Z = sorted_point_cloud[:, 2]
    
    for i in range(len(X)):
        ax3.scatter(X[i], Y[i], Z[i], c=[colors[i]], marker='o', s=10)
    
    ax3.scatter(0, 0, 0, c='k', marker='^', s=100, label='Origin')  # Black triangle

    #visualize camera reference frame
    ax3.plot([0, 100], [0, 0], [0, 0], c='r')
    ax3.plot([0, 0], [0, 100], [0, 0], c='g')
    ax3.plot([0, 0], [0, 0], [0, 100], c='k')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title("3D Point Cloud")
    
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    ax3.set_xlim(0 - 20, 0 + 20)
    ax3.set_ylim(0 - 20, 0 + 20)
    ax3.set_zlim(0 - 20, 0 + 20)
    
    plt.tight_layout()
    plt.show()


def show_matches(image1, image2, matches, keypoints1, keypoints2):
    image_with_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', marker='o')
    # plot the origin
    ax.scatter(0, 0, 0, c='b', marker='^', s=100, label='Origin')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


#Find the transformation matrix between the two frames using the two point clouds
def find_transformation(dst_points, src_points):
    # Filter out points that are behind the camera
    valid_indices = (src_points[:, 2] > 0) & (dst_points[:, 2] > 0)
    src_points = src_points[valid_indices]
    dst_points = dst_points[valid_indices]
    
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
    
    # handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    
    # find translation
    t = dst_centroid - R @ src_centroid
    
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    
    return T

#Iterative function for finding transformation robust to outliers
def find_transformation_iterative(dst_points, src_points, threshold=0.1, max_iterations=5, tol=1e-6):
    T = np.eye(4)
    src_points_homogeneous = np.hstack((src_points, np.ones((src_points.shape[0], 1))))
    
    for _ in range(max_iterations):
        # Apply current transformation to the original source points (avoids accumulating drift)
        src_transformed = (T @ src_points_homogeneous.T).T[:, :3]

        # Compute distances
        distances = np.linalg.norm(src_transformed - dst_points, axis=1)

        # Robust weighting using Huber loss
        delta = threshold  # Huber delta parameter
        weights = np.where(distances <= delta, 1, delta / np.maximum(distances, 1e-8))
        weights /= np.sum(weights)  # Normalize

        # Compute weighted centroids
        src_centroid = np.average(src_transformed, axis=0, weights=weights)
        dst_centroid = np.average(dst_points, axis=0, weights=weights)

        # Center points
        src_centered = src_transformed - src_centroid
        dst_centered = dst_points - dst_centroid

        # Apply weights
        src_centered_weighted = src_centered * weights[:, np.newaxis]
        dst_centered_weighted = dst_centered * weights[:, np.newaxis]

        # Compute weighted covariance
        H = src_centered_weighted.T @ dst_centered_weighted
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure a proper rotation (prevent reflection)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = dst_centroid - R @ src_centroid

        # Construct new transformation matrix
        T_new = np.eye(4)
        T_new[:3, :3] = R
        T_new[:3, 3] = t

        # Damped update to prevent overshooting (smooth blending)
        alpha = 0.5  # Lower values slow down updates
        T = alpha * (T_new @ T) + (1 - alpha) * T

        # Check convergence
        if np.linalg.norm(T_new - T) < tol:
            break

    return T


def update_camera_pose(T_relative):
    global T_world_to_current
    T_world_to_current = T_world_to_current @ T_relative  # Matrix multiplication
    return T_world_to_current


T_cam02imu0 = np.array([[-0.02822879, 0.01440125, 0.99949774, 0.00110212],
            [-0.99960149, -0.00041887, -0.02822568, 0.02170142],
            [ 0.00001218, -0.99989621, 0.01440734, -0.00005928],
            [ 0., 0., 0., 1. ]])

T_cam2drone = T_cam02imu0

class FeatureData:
    def __init__(self, filtered_matches, left_keypoints, left_descriptors, right_keypoints, right_descriptors):
        self.filtered_matches = filtered_matches
        self.left_keypoints = left_keypoints
        self.left_descriptors = left_descriptors
        self.right_keypoints = right_keypoints
        self.right_descriptors = right_descriptors


class VisionRelativeOdometryCalculator:
    def __init__(self, initial_camera_input:interface.VisionInputFrame, feature_extractor:FeatureExtractor, feature_matcher:FeatureMatcher, feature_match_filter:FeatureMatchFilter, alpha=1, transformation_threshold=0.1):
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.feature_match_filter = feature_match_filter
        self.current_feature_data = None
        self.previous_feature_data = None
        self.StereoPair = StereoProjection("analysis/camchain-..indoor_forward_calib_snapdragon_cam.yaml", distortion="fisheye")
        self.update_current_feature_data(initial_camera_input)
        self.previous_feature_data = self.current_feature_data
        self.alpha=alpha
        self.filtered_R = None
        self.filtered_t = None
        self.transformation_threshold = transformation_threshold
    
    def update_current_feature_data(self, input:interface.VisionInputFrame):
        left_image, right_image = load_images(input)
        left_preprocessed, right_preprocessed = preprocess_images(left_image, right_image)
        
        left_keypoints, left_descriptors = self.feature_extractor.extract_features(left_preprocessed)
        right_keypoints, right_descriptors = self.feature_extractor.extract_features(right_preprocessed)

        matches = self.feature_matcher.match_features(left_descriptors, right_descriptors)

        #Undistort the keypoints before filtering
        # Extract points from keypoints for undistortion
        left_points = np.array([kp.pt for kp in left_keypoints], dtype=np.float32)
        right_points = np.array([kp.pt for kp in right_keypoints], dtype=np.float32)
        
        # Undistort the points
        left_points_undistorted = self.StereoPair.undistort_points(left_points, camera="left")
        right_points_undistorted = self.StereoPair.undistort_points(right_points, camera="right")
        
        # Create new keypoints with undistorted coordinates
        for i, kp in enumerate(left_keypoints):
            kp.pt = (left_points_undistorted[i][0], left_points_undistorted[i][1])
            
        for i, kp in enumerate(right_keypoints):
            kp.pt = (right_points_undistorted[i][0], right_points_undistorted[i][1])

        filtered_matches = self.feature_match_filter.filter_matches(matches, left_keypoints, right_keypoints)

        self.current_feature_data = FeatureData(filtered_matches, left_keypoints, left_descriptors, right_keypoints, right_descriptors)
    
    def isolate_common_matches(self):
        filtered_matches1 = self.previous_feature_data.filtered_matches
        filtered_matches2 = self.current_feature_data.filtered_matches

        matches_between_frames = self.feature_matcher.match_features(self.previous_feature_data.left_descriptors,
                                                                     self.current_feature_data.left_descriptors)
        filtered_matches_between_frames = self.feature_match_filter.filter_matches(matches_between_frames,
                                                                                   self.previous_feature_data.left_keypoints,
                                                                                   self.current_feature_data.left_keypoints)
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
        
        return consistent_matches1, consistent_matches2
    
    def calculate_relative_odometry_homogenous(self, input_frame:interface.VisionInputFrame, camera_frame=False) -> np.ndarray:
        #Update the current feature data from the input images
        self.update_current_feature_data(input_frame)

        #Isolate the common matches between the current and previous frames
        consistent_matches1, consistent_matches2 = self.isolate_common_matches()

        pl1, pr1 = extract_points_from_matches(consistent_matches1,
                                               self.previous_feature_data.left_keypoints,
                                               self.previous_feature_data.right_keypoints)
        points1 = self.StereoPair.triangulate_points(np.array(pl1), np.array(pr1), use_normalized_projection=True)

        pl2, pr2 = extract_points_from_matches(consistent_matches2,
                                        self.current_feature_data.left_keypoints,
                                        self.current_feature_data.right_keypoints)
        points2 = self.StereoPair.triangulate_points(np.array(pl2), np.array(pr2), use_normalized_projection=True)

        transformation = find_transformation_iterative(points1, points2, threshold =self.transformation_threshold)
        if not(camera_frame):
            transformation = (
                T_cam2drone @
                transformation @ 
                np.linalg.inv(T_cam2drone)
            )

        if self.alpha != 1:
            # Apply exponential smoothing to the transformation matrix
            R_new = transformation[:3, :3]
            t_new = transformation[:3, 3]
            if self.filtered_R is None:
                self.filtered_R = R_new.copy()
                self.filtered_t = t_new.copy()
            else:
                # Filter the translation
                self.filtered_t = self.alpha * t_new + (1 - self.alpha) * self.filtered_t

                # Filter the rotation using quaternions
                key_times = [0, 1]
                key_rotations = Rotation.from_matrix([self.filtered_R, R_new])
                slerp = Slerp(key_times, key_rotations)
                self.filtered_R = slerp(self.alpha).as_matrix()
            
            # Combine filtered rotation and translation into a transformation matrix
            transformation = np.eye(4)
            transformation[:3, :3] = self.filtered_R
            transformation[:3, 3] = self.filtered_t

        self.previous_feature_data = self.current_feature_data

        return transformation
    
    def calculate_relative_odometry(self, input_frame:interface.VisionInputFrame, camera_frame=False) -> interface.VisionRelativeOdometry:
        if camera_frame:
            homo_transformation = self.calculate_relative_odometry(input_frame, camera_frame=True)
        else:
            homo_transformation = self.calculate_relative_odometry_homogenous(input_frame, camera_frame=False)
        return interface.create_VisionRelativeOdometry_from_homogeneous_matrix(homo_transformation)

    def plot_point_clouds(self, points1_world, points2_world):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points1_world[:, 0], points1_world[:, 1], points1_world[:, 2], c='r', marker='o')
        ax.scatter(points2_world[:, 0], points2_world[:, 1], points2_world[:, 2], c='b', marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

if __name__ == "__main__":
    # load in a stereo pair and two sequential frames
    frame1 = interface.VisionInputFrame("dataset/vio_dataset_1/img/image_0_1500.png", "dataset/vio_dataset_1/img/image_1_1500.png", 0)
    frame2 = interface.VisionInputFrame("dataset/vio_dataset_1/img/image_0_1501.png", "dataset/vio_dataset_1/img/image_1_1501.png", 1)

    left1, right1 = load_images(frame1)
    left2, right2 = load_images(frame2)

    left1_gs, right1_gs = preprocess_images(left1, right1)
    left2_gs, right2_gs = preprocess_images(left2, right2)

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

    StereoPair = StereoProjection("analysis/camchain-..indoor_forward_calib_snapdragon_cam.yaml", distortion="fisheye")

    pl1, pr1 = extract_points_from_matches(consistent_matches1, l1_keypoints, r1_keypoints)
    points1 = StereoPair.triangulate_points(np.array(pl1), np.array(pr1), use_normalized_projection=True)

    pl2, pr2 = extract_points_from_matches(consistent_matches2, l2_keypoints, r2_keypoints)
    points2 = StereoPair.triangulate_points(np.array(pl2), np.array(pr2), use_normalized_projection=True)

    StereoPair.plot_undistorted_points(pl1, pr1, left1, right1)
    
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

    true_second_position = np.array([
    [-0.630914015914675, 0.7757556783140641, 0.012273226385246232, 7.60666536072954],
    [-0.7758406136151782, -0.6307366795317234, -0.015575087753203145, 0.246270702230251],
    [-0.004341288707415834, -0.0193486086523643, 0.9998033729466893, -0.880807258137499],
    [0.0, 0.0, 0.0, 1.0]])

    # plots the point clouds
    show_points(left1, pl1, right1, pr1, points1)
    show_points(left2, pl2, right2, pr2, points2)

    #transform points1 to the world frame
    points1_world = np.dot(T_cam_imu0, np.dot(T_world_to_current, np.vstack((points1.T, np.ones(points1.shape[0])))))
    points1_world = points1_world[:3].T

    # Get old camera pos in world coords
    old_cam_pos = np.dot(T_world_to_current, np.array([0, 0, 0, 1]))
    old_cam_pos = old_cam_pos[:3]

    # Transform points2 to the new world frame
    points2_world = np.dot(T_cam_imu0, np.dot(T_world_to_current2, np.vstack((points2.T, np.ones(points2.shape[0])))))
    points2_world = points2_world[:3].T

    # Get new camera pos in world coords
    new_cam_pos = np.dot(T_world_to_current2, np.array([0, 0, 0, 1]))
    new_cam_pos = new_cam_pos[:3]

    true_cam_pos = np.dot(true_second_position, np.array([0, 0, 0, 1]))
    true_cam_pos = true_cam_pos[:3]

    #Find error between true second position and calculated second position
    error = np.linalg.norm(true_second_position - T_world_to_current2)
    print("Error between true second position and calculated second position: ", error)
    #print deltas in x, y, and z between second position and known second position
    print(f"Delta x: {true_cam_pos[0] - new_cam_pos[0]}")
    print(f"Delta y: {true_cam_pos[1] - new_cam_pos[1]}")
    print(f"Delta z: {true_cam_pos[2] - new_cam_pos[2]}")

    print(old_cam_pos[0], old_cam_pos[1], old_cam_pos[2])
    print(true_cam_pos[0], true_cam_pos[1], true_cam_pos[2])

    #plot the old points in the world frame in red, and the new ones in blue in the same plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points1_world[:, 0], points1_world[:, 1], points1_world[:, 2], c='r', marker='o')
    ax.scatter(points2_world[:, 0], points2_world[:, 1], points2_world[:, 2], c='b', marker='o')

    #plot old camera pos in green
    ax.scatter(true_cam_pos[0], true_cam_pos[1], true_cam_pos[2], c='g', marker='v', s=100)

    #plot new camera pos in purple
    ax.scatter(new_cam_pos[0], new_cam_pos[1], new_cam_pos[2], c='purple', marker='^', facecolors='none', s=100)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
