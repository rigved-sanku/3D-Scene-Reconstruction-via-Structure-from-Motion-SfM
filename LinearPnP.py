import numpy as np

def ProjectionMatrix(R, C, K):
    '''
    Compute the projection matrix from camera pose and intrinsic matrix.
    '''
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P


def homo(pts):
    ''''
    Add a homogeneous coordinate to the given points.
    '''
    return np.hstack((pts, np.ones((pts.shape[0], 1))))


def reprojectionErrorPnP(x3D, pts, K, R, C):
    '''
    Compute reprojection error for Perspective-n-Point (PnP) algorithm.
    '''
    P = ProjectionMatrix(R, C, K)
    
    Error = []
    for X, pt in zip(x3D, pts):

        p_1T, p_2T, p_3T = P  # Rows of P
        p_1T, p_2T, p_3T = p_1T.reshape(1, -1), p_2T.reshape(1, -1), p_3T.reshape(1, -1)
        X = homo(X.reshape(1, -1)).reshape(-1, 1)  # Make X a column of homogeneous vector
        
        ## Reprojection error for reference camera points 
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X), p_3T.dot(X))
        v_proj = np.divide(p_2T.dot(X), p_3T.dot(X))

        E = np.square(v - v_proj) + np.square(u - u_proj)

        Error.append(E)

    mean_error = np.mean(np.array(Error).squeeze())
    return mean_error


def PnP(X_set, x_set, K):
    '''
    Perspective-n-Point (PnP) algorithm to estimate camera pose.
    '''
    N = X_set.shape[0]
    
    X_4 = homo(X_set)
    x_3 = homo(x_set)
    
    # Normalize x
    K_inv = np.linalg.inv(K)
    x_n = K_inv.dot(x_3.T).T
    
    for i in range(N):
        X = X_4[i].reshape((1, 4))
        zeros = np.zeros((1, 4))
        
        u, v, _ = x_n[i]
        
        u_cross = np.array([[0, -1, v],
                            [1,  0 , -u],
                            [-v, u, 0]])
        X_tilde = np.vstack((np.hstack((   X, zeros, zeros)), 
                            np.hstack((zeros,     X, zeros)), 
                            np.hstack((zeros, zeros,     X))))
        a = u_cross.dot(X_tilde)
        
        if i > 0:
            A = np.vstack((A, a))
        else:
            A = a
            
    _, _, VT = np.linalg.svd(A)
    P = VT[-1].reshape((3, 4))
    R = P[:, :3]
    U_r, D, V_rT = np.linalg.svd(R)  # Ensure orthonormality
    R = U_r.dot(V_rT)
    
    C = P[:, 3]
    C = - np.linalg.inv(R).dot(C)
    
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
        
    return R, C