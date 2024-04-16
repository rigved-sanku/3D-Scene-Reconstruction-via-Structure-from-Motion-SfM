import numpy as np
import scipy.optimize as optimize

def ProjectionMatrix(R, C, K):
    '''
    Compute the projection matrix from camera pose and intrinsic matrix.
    '''
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P


def NonLinearTriangulation(K, pts1, pts2, x3D, R1, C1, R2, C2):
    '''
    Non-linear triangulation to optimize 3D points using reprojection error minimization.
    '''
    
    P1 = ProjectionMatrix(R1, C1, K) 
    P2 = ProjectionMatrix(R2, C2, K)
    
    if pts1.shape[0] != pts2.shape[0] != x3D.shape[0]:
        raise ValueError('Check point dimensions - level nlt')

    x3D_ = []
    for i in range(len(x3D)):
        optimized_params = optimize.least_squares(fun=ReprojectionLoss, x0=x3D[i], method="trf", args=[pts1[i], pts2[i], P1, P2])
        X1 = optimized_params.x
        x3D_.append(X1)
    return np.array(x3D_)


def ReprojectionLoss(X, pts1, pts2, P1, P2):
    '''
    Compute reprojection error for non-linear triangulation optimization.
    '''
    
    p1_1T, p1_2T, p1_3T = P1  # Rows of P1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1, -1), p1_2T.reshape(1, -1), p1_3T.reshape(1, -1)

    p2_1T, p2_2T, p2_3T = P2  # Rows of P2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1, -1), p2_2T.reshape(1, -1), p2_3T.reshape(1, -1)

    # Reprojection error for reference camera points - j = 1
    u1, v1 = pts1[0], pts1[1]
    u1_proj = np.divide(p1_1T.dot(X), p1_3T.dot(X))
    v1_proj = np.divide(p1_2T.dot(X), p1_3T.dot(X))
    E1 = np.square(v1 - v1_proj) + np.square(u1 - u1_proj)

    # Reprojection error for second camera points - j = 2    
    u2, v2 = pts2[0], pts2[1]
    u2_proj = np.divide(p2_1T.dot(X), p2_3T.dot(X))
    v2_proj = np.divide(p2_2T.dot(X), p2_3T.dot(X))    
    E2 = np.square(v2 - v2_proj) + np.square(u2 - u2_proj)
    
    error = E1 + E2
    return error.squeeze()