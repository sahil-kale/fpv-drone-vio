import interface
import cv2
import numpy as np
import pandas as pd

class FeatureExtractor:
    @staticmethod
    def extract_features(input: interface.VisionInputFrame):
        """
        Extract features from an image
        """
        pass


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