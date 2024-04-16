import numpy as np
from scipy.spatial.transform import Rotation 
import scipy.optimize as optimize

def getRotation(Q, type_='q'):
    '''
    Convert quaternion or rotation vector to rotation matrix.
    '''
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()

def homo(pts):
    '''
    Convert points to homogeneous coordinates.
    '''
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def getQuaternion(R):
    '''
    Convert rotation matrix to quaternion.
    '''
    Q = Rotation.from_matrix(R)
    return Q.as_quat()

def ProjectionMatrix(R, C, K):
    '''
    Compute the projection matrix.
    '''
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P

def NonLinearPnP(K, pts, x3D, R0, C0):
    '''    
    Non-linear optimization for Perspective-n-Point (PnP) problem.
    '''
    Q = getQuaternion(R0)
    X0 = np.concatenate((Q, C0)) 

    optimized_params = optimize.least_squares(
        fun=PnPLoss,
        x0=X0,
        method="trf",
        args=[x3D, pts, K])
    X1 = optimized_params.x
    Q = X1[:4]
    C = X1[4:]
    R = getRotation(Q)
    return R, C

def PnPLoss(X0, x3D, pts, K):
    '''
    Compute reprojection error for PnP problem.
    '''
    Q, C = X0[:4], X0[4:].reshape(-1, 1)
    R = getRotation(Q)
    P = ProjectionMatrix(R, C, K)
    
    Error = []
    for X, pt in zip(x3D, pts):
        p_1T, p_2T, p_3T = P

        X = homo(X.reshape(1, -1)).reshape(-1, 1) 
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X), p_3T.dot(X))
        v_proj = np.divide(p_2T.dot(X), p_3T.dot(X))

        E = np.square(v - v_proj) + np.square(u - u_proj)
        Error.append(E)

    sumError = np.mean(np.array(Error).squeeze())
    return sumError