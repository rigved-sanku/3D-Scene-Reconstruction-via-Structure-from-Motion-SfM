import numpy as np

def normalize(uv):
    '''
    Normalize 2D points by scaling and translation.
    '''
    # Compute mean of points
    uv_mean = np.mean(uv, axis=0)
    u_mean, v_mean = uv_mean[0], uv_mean[1]
    
    # Compute differences from mean
    u_cap, v_cap = uv[:, 0] - u_mean, uv[:, 1] - v_mean

    # Compute scale factor
    s = (2 / np.mean(u_cap ** 2 + v_cap ** 2)) ** 0.5
    T_scale = np.diag([s, s, 1])
    T_trans = np.array([[1, 0, -u_mean], [0, 1, -v_mean], [0, 0, 1]])
    T = T_scale.dot(T_trans)

    # Apply transformation
    x_ = np.column_stack((uv, np.ones(len(uv))))  #[x,y,1]
    x_norm = (T.dot(x_.T)).T     #x_ = T.x

    return x_norm, T

def EstimateFundamentalMatrix(pts1, pts2):
    '''
    Estimate fundamental matrix between two sets of corresponding points.
    '''
    normalised = True

    x1, x2 = pts1, pts2

    if x1.shape[0] > 7:
        if normalised:
            # Normalize points
            x1_norm, T1 = normalize(x1)
            x2_norm, T2 = normalize(x2)
        else:
            x1_norm, x2_norm = x1, x2

        # Compute matrix A for SVD
        A = np.zeros((len(x1_norm), 9))
        for i in range(len(x1_norm)):
            x_1, y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2, y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1 * x_2, x_2 * y_1, x_2, y_2 * x_1, y_2 * y_1, y_2, x_1, y_1, 1])

        # Perform SVD
        U, S, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1].reshape(3, 3)

        # Ensure rank-2 by setting last singular value to zero
        U, S, VT = np.linalg.svd(F)
        S[2] = 0
        F = np.dot(U, np.dot(np.diag(S), VT))

        if normalised:
            # Denormalize fundamental matrix
            F = np.dot(T2.T, np.dot(F, T1))
            F /= F[2, 2]  # Normalize the scale

        return F
    else:
        return None
    