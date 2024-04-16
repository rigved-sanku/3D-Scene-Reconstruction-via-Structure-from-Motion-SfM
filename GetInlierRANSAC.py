import numpy as np
from EstimateFundamentalMatrix import *


def error_F(pts1, pts2, F):
    '''
    Compute the epipolar constraint error between two points and the fundamental matrix.
    '''
    # Homogeneous coordinates
    x1 = np.array([pts1[0], pts1[1], 1])
    x2 = np.array([pts2[0], pts2[1], 1]).T

    # Compute error using epipolar constraint equation: x2.T * F * x1
    error = np.dot(x2, np.dot(F, x1))

    return np.abs(error)

def getInliers(pts1, pts2, idx):
    '''
    Use RANSAC to find inliers given corresponding points in two images and their indices.
    '''
    # RANSAC parameters
    num_iterations = 2000
    error_threshold = 0.002
    max_inliers = 0
    inliers_idx = []
    F_inliers = None

    for _ in range(num_iterations):
        # Randomly select 8 points for estimating fundamental matrix
        random_indices = np.random.choice(pts1.shape[0], 8, replace=False)
        x1 = pts1[random_indices,:]
        x2 = pts2[random_indices,:]
        
        # Estimate fundamental matrix using selected points
        F = EstimateFundamentalMatrix(x1, x2)
        if F is not None:
            # Calculate error for each point and check if it's an inlier
            inliers = [idx[i] for i in range(pts1.shape[0]) if error_F(pts1[i], pts2[i], F) < error_threshold]
            num_inliers = len(inliers)
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                inliers_idx = inliers
                F_inliers = F

    return F_inliers, inliers_idx