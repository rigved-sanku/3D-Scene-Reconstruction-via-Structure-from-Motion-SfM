import numpy as np

def linearTriangulation(K, C1, R1, C2, R2, x1, x2):
    """
    Perform linear triangulation to estimate 3D points from corresponding 2D points in two camera views.
    """
    # Construct identity matrix
    I = np.identity(3)

    # Reshape translation vectors for matrix multiplication
    C1 = np.reshape(C1, (3, 1))
    C2 = np.reshape(C2, (3, 1))

    # Construct projection matrices
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    # Extract rows of projection matrices
    p1T = P1[0, :].reshape(1, 4)
    p2T = P1[1, :].reshape(1, 4)
    p3T = P1[2, :].reshape(1, 4)
    p_1T = P2[0, :].reshape(1, 4)
    p_2T = P2[1, :].reshape(1, 4)
    p_3T = P2[2, :].reshape(1, 4)

    X = []
    for i in range(x1.shape[0]):
        # Extract coordinates of corresponding 2D points
        x = x1[i, 0]
        y = x1[i, 1]
        x_ = x2[i, 0]
        y_ = x2[i, 1]

        # Construct matrix A for linear equation Ax = 0
        A = []
        A.append((y * p3T) - p2T)
        A.append(p1T - (x * p3T))
        A.append((y_ * p_3T) - p_2T)
        A.append(p_1T - (x_ * p_3T))

        # Convert A to numpy array and reshape
        A = np.array(A).reshape(4, 4)

        # Perform SVD and extract solution
        _, _, vt = np.linalg.svd(A)
        v = vt.T
        x = v[:, -1]
        X.append(x)

    # Convert list of points to numpy array
    return np.array(X)