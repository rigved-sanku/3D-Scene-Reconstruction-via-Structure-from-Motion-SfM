import numpy as np

def ExtractCameraPose(E):
    '''
    Extract possible camera poses from the essential matrix.
    '''
    # Perform SVD decomposition of essential matrix
    U, _, V_T = np.linalg.svd(E)

    # Define the skew-symmetric matrix
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Initialize lists to store rotation matrices and translation vectors
    R = []
    C = []

    # Calculate possible camera poses
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))

    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    # Ensure correct orientation of rotation matrices and translation vectors
    for i in range(4):
        if np.linalg.det(R[i]) < 0:
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C