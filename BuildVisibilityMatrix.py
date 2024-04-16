import numpy as np

def getObservationsIndexAndVizMat(X_found, filtered_feature_flag, nCam):
    '''
    Get the indices of 3D points observed in at least one of the first nCam cameras and their visibility matrix.
    '''
    # Create a binary array to indicate if a 3D point is visible in any of the first nCam cameras
    bin_temp = np.zeros((filtered_feature_flag.shape[0]), dtype=int)
    for n in range(nCam + 1):
        bin_temp = bin_temp | filtered_feature_flag[:, n]

    # Get the indices of 3D points observed in at least one of the first nCam cameras
    X_index = np.where((X_found.reshape(-1)) & (bin_temp))
    
    # Extract the visibility matrix for the observed 3D points
    visibility_matrix = X_found[X_index].reshape(-1, 1)
    for n in range(nCam + 1):
        visibility_matrix = np.hstack((visibility_matrix, filtered_feature_flag[X_index, n].reshape(-1, 1)))

    # Get the shape of the visibility matrix
    o, c = visibility_matrix.shape
    
    # Return the indices and visibility matrix
    return X_index, visibility_matrix[:, 1:c]