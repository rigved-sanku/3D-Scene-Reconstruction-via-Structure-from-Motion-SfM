import numpy as np

def DisambiguatePose(r_set, c_set, x3D_set):
    '''
    Resolve pose ambiguity by selecting the camera pose with the maximum number of reconstructed points in front of the cameras.
    source : https://www.cis.upenn.edu/~cis580/Spring2015/Projects/proj2/proj2.pdf
    '''
    best_i = 0
    max_positive_depths = 0

    # Iterate through all camera poses
    for i in range(len(r_set)):
        R, C = r_set[i], c_set[i]
        r3 = R[2, :].reshape(1, -1)  # Extract the third row of the rotation matrix
        x3D = x3D_set[i]
        x3D = x3D / x3D[:, 3].reshape(-1, 1)  # Normalize 3D points
        x3D = x3D[:, 0:3]  # Extract only the spatial coordinates of 3D points

        # Check the positivity of depth for each 3D point
        n_positive_depths = DepthPositivityConstraint(x3D, r3, C)

        # Update the maximum positive depths and the corresponding index
        if n_positive_depths > max_positive_depths:
            best_i = i
            max_positive_depths = n_positive_depths

    # Select the camera pose with the maximum number of positive depths
    R, C, x3D = r_set[best_i], c_set[best_i], x3D_set[best_i]

    return R, C, x3D 

def DepthPositivityConstraint(x3D, r3, C):
    '''
    Check the positivity of depth for each reconstructed 3D point.
    '''
    n_positive_depths = 0
    
    # Iterate through each reconstructed 3D point
    for X in x3D:
        X = X.reshape(-1, 1)  # Reshape to column vector
        C = C.reshape(-1, 1)  # Reshape to column vector
        
        # Check the positivity of depth
        if r3.dot(X - C).T > 0 and X[2] > 0:  # Check positivity of dot product and z-coordinate
            n_positive_depths += 1

    return n_positive_depths