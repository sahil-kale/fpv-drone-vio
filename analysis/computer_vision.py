import interface
import cv2
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

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


#Images L & R --> groups of points on each image corresponding
#to common features

#List of images (L & R)1, (L & R)2, (L & R)3, ... (L & R)n
# Find common features shared within the set
# Overlapping windows of frames and pick features you can find in all frames

#Once identified, go back into LR pairs and find 3D point cloud for those features

#Classes:
#Image Pair
#Feature
#Point Cloud - List of points and those points are associated with features