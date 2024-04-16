import numpy as np

def getEssentialMatrix(K, F):
    '''
    Compute the essential matrix from the fundamental matrix and camera calibration matrix.
    '''
    # Compute essential matrix
    E = np.dot(K.T, np.dot(F, K))

    # Ensure essential matrix has the correct form
    U, S, V = np.linalg.svd(E)
    S = [1, 1, 0]  # Ensure the last singular value is zero for rank-2 matrix
    E = np.dot(U, np.dot(np.diag(S), V))

    return E