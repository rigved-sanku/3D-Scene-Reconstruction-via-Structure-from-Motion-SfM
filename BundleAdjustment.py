import numpy as np
import time
import cv2
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from BuildVisibilityMatrix import *

"https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html"

"This is for Large-Scale bundle Adjustment in Scipy"

def getRotation(Q, type_ = 'q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()

def getEuler(R2):
    euler = Rotation.from_matrix(R2)
    return euler.as_rotvec()


def get2DPoints(X_index, visiblity_matrix, feature_x, feature_y):
    "Get 2D Points from the feature x and feature y having same index from the visibility matrix"
    pts2D = []
    visible_feature_x = feature_x[X_index]
    visible_feature_y = feature_y[X_index]
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                pt = np.hstack((visible_feature_x[i,j], visible_feature_y[i,j]))
                pts2D.append(pt)
    return np.array(pts2D).reshape(-1, 2) 

def getCameraPointIndices(visiblity_matrix):

    "From Visibility Matrix take away indices of point visible from Camera pose by taking indices of cam as well"

    camera_indices = []
    point_indices = []
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                camera_indices.append(j)
                point_indices.append(i)

    return np.array(camera_indices).reshape(-1), np.array(point_indices).reshape(-1)

def bundle_adjustment_sparsity(X_found, filtered_feature_flag, nCam):
    
    "Create Sparsity Matrix"

    number_of_cam = nCam + 1
    X_index, visiblity_matrix = getObservationsIndexAndVizMat(X_found.reshape(-1), filtered_feature_flag, nCam)
    n_observations = np.sum(visiblity_matrix)
    n_points = len(X_index[0])

    m = n_observations * 2
    n = number_of_cam * 6 + n_points * 3   #Here we don't take focal length and 2 radial distortion parameters, i.e no - 6. We just refine orientation and translation of 3d point and not cam parameters.
    A = lil_matrix((m, n), dtype=int)
    # print(m, n)


    i = np.arange(n_observations)
    camera_indices, point_indices = getCameraPointIndices(visiblity_matrix)

    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, (nCam)* 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, (nCam) * 6 + point_indices * 3 + s] = 1

    return A    

def project(points_3d, camera_params, K):
    def projectPoint_(R, C, pt3D, K):
        P2 = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
        x3D_4 = np.hstack((pt3D, 1))
        x_proj = np.dot(P2, x3D_4.T)
        x_proj /= x_proj[-1]
        return x_proj

    x_proj = []
    for i in range(len(camera_params)):
        R = getRotation(camera_params[i, :3], 'e')
        C = camera_params[i, 3:].reshape(3,1)
        pt3D = points_3d[i]
        pt_proj = projectPoint_(R, C, pt3D, K)[:2]
        x_proj.append(pt_proj)    
    return np.array(x_proj)

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def fun(x0, nCam, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    number_of_cam = nCam + 1
    camera_params = x0[:number_of_cam * 6].reshape((number_of_cam, 6))
    points_3d = x0[number_of_cam * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    error_vec = (points_proj - points_2d).ravel()
    
    return error_vec  


def BundleAdjustment(X_index,visibility_matrix,X_all,X_found,feature_x,feature_y, filtered_feature_flag, R_set, C_set, K, nCam):
    points_3d = X_all[X_index]
    points_2d = get2DPoints(X_index,visibility_matrix,feature_x,feature_y)

    RC = []
    for i in range(nCam+1):
        C, R = C_set[i], R_set[i]
        Q = getEuler(R)
        RC_ = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        RC.append(RC_)
    RC = np.array(RC, dtype=object).reshape(-1,6)

    x0 = np.hstack((RC.ravel(), points_3d.ravel()))
    n_pts = points_3d.shape[0]

    camera_indices, points_indices = getCameraPointIndices(visibility_matrix)

    A = bundle_adjustment_sparsity(X_found,filtered_feature_flag,nCam)
    t0 = time.time()
    res = least_squares(fun,x0,jac_sparsity=A, verbose=2,x_scale='jac', ftol=1e-10, method='trf',
                        args=(nCam, n_pts, camera_indices, points_indices, points_2d,K))

    t1 = time.time()
    print("Time required to run Bundle Adj: ", t1-t0, "s \nA matrix shape: ",A.shape,"\n######")

    x1 = res.x
    no_of_cams = nCam + 1
    optim_cam_param = x1[:no_of_cams*6].reshape((no_of_cams,6))
    optim_pts_3d = x1[no_of_cams*6:].reshape((n_pts,3))

    optim_X_all = np.zeros_like(X_all)
    optim_X_all[X_index] = optim_pts_3d

    optim_C_set , optim_R_set = [], []
    for i in range(len(optim_cam_param)):
        R = getRotation(optim_cam_param[i,:3], 'e')
        C = optim_cam_param[i,3:].reshape(3,1)
        optim_C_set.append(C)
        optim_R_set.append(R)

    return optim_R_set, optim_C_set, optim_X_all