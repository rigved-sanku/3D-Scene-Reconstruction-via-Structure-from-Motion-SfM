""""
RBE549: Classical and Deep Learning Approaches for Computer Vision (Spring 2024)
Project 3: Building built in minutes: SfM and NeRF

Author:
Rigved Sanku(sanku@wpi.edu) and Smit Shah(mshah1@wpi.edu)
Worcester Polytechnic Institute, Ms Robotics
Following functions are somewhat Inspired by Sakshi Kakde
"""



import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits import mplot3d

from EstimateFundamentalMatrix import *
from GetInlierRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from NonlinearTriangulation import *
from DisambiguateCameraPose import *
from LinearPnP import *
from PnPRANSAC import *
from NonlinearPnP import *
from BuildVisibilityMatrix import *
from BundleAdjustment import *


def projectionMatrix(R, C, K):
    '''
    Compute the projection matrix from camera pose and intrinsic matrix.
    
    Args:
    - R: numpy array, rotation matrix
    - C: numpy array, translation vector
    - K: numpy array, camera intrinsic matrix
    
    Returns:
    - P: numpy array, projection matrix
    '''
    C = np.reshape(C, (3, 1))
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P


def ReProjectionError(X, pt1, pt2, R1, C1, R2, C2, K):
    '''
    Compute the reprojection error given 3D point, corresponding 2D points,
    and camera poses.
    '''
    p1 = projectionMatrix(R1, C1, K)
    p2 = projectionMatrix(R2, C2, K)

    p1_1T, p1_2T, p1_3T = p1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1, 4), p1_2T.reshape(1, 4), p1_3T.reshape(1, 4)

    p2_1T, p2_2T, p2_3T = p2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1, 4), p2_2T.reshape(1, 4), p2_3T.reshape(1, 4)

    X = X.reshape(4, 1)

    # Reprojection error w.r.t 1st reference camera points
    u1, v1 = pt1[0], pt1[1]
    u1_projection = np.divide(p1_1T.dot(X), p1_3T.dot(X))
    v1_projection = np.divide(p1_2T.dot(X), p1_3T.dot(X))
    err1 = np.square(v1 - v1_projection) + np.square(u1 - u1_projection)

    # Reprojection error w.r.t 2nd reference camera points
    u2, v2 = pt2[0], pt2[1]
    u2_projection = np.divide(p2_1T.dot(X), p2_3T.dot(X))
    v2_projection = np.divide(p2_2T.dot(X), p2_3T.dot(X))
    err2 = np.square(v2 - v2_projection) + np.square(u2 - u2_projection)

    return err1, err2


def features_extraction(data):
    # Number of images
    no_of_images = 5
    
    # Initialize lists to store feature data
    feature_rgb_values = []
    feature_x = []
    feature_y = []
    feature_flag = []

    # Loop through each image
    for n in range(1, no_of_images):
        # File path for each image
        file = f"{data}/matching{n}.txt"
        
        # Open the matching file for the current image
        with open(file, "r") as matching_file:
            nfeatures = 0  # Initialize the number of features for each image
            for i, row in enumerate(matching_file):
                if i == 0:
                    # Extract the number of features from the first row
                    row_elements = row.split(':')
                    nfeatures = int(row_elements[1])
                else:
                    # Initialize arrays to store x, y, and flag data for each image
                    x_row = np.zeros((1, no_of_images))
                    y_row = np.zeros((1, no_of_images))
                    flag_row = np.zeros((1, no_of_images), dtype=int)
                    
                    # Parse the row elements
                    row_elements = row.split()
                    columns = [float(x) for x in row_elements]
                    columns = np.asarray(columns)

                    # Extract RGB values
                    nMatches = columns[0]
                    r_value, b_value, g_value = columns[1:4]
                    feature_rgb_values.append([r_value, g_value, b_value])

                    # Extract current x and y coordinates
                    current_x, current_y = columns[4], columns[5]
                    x_row[0, n - 1] = current_x
                    y_row[0, n - 1] = current_y
                    flag_row[0, n - 1] = 1

                    # Extract additional matches if available
                    m = 1
                    while nMatches > 1:
                        image_id, image_id_x, image_id_y = columns[5+m:8+m]
                        m = m + 3
                        nMatches = nMatches - 1

                        x_row[0, int(image_id) - 1] = image_id_x
                        y_row[0, int(image_id) - 1] = image_id_y
                        flag_row[0, int(image_id) - 1] = 1

                    # Append the extracted data to the corresponding lists
                    feature_x.append(x_row)
                    feature_y.append(y_row)
                    feature_flag.append(flag_row)

    # Convert lists to numpy arrays and reshape them
    feature_x = np.asarray(feature_x).reshape(-1, no_of_images)
    feature_y = np.asarray(feature_y).reshape(-1, no_of_images)
    feature_flag = np.asarray(feature_flag).reshape(-1, no_of_images)
    feature_rgb_values = np.asarray(feature_rgb_values).reshape(-1, 3)

    return feature_x, feature_y, feature_flag, feature_rgb_values




def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Outputs', default='../Outputs/', help='Outputs are saved here')
    Parser.add_argument('--Data', default='../Data/P3Data', help='Data')

    Args = Parser.parse_args()
    Data = Args.Data
    Output = Args.Outputs

    #Images
    images = []
    for i in range(1,6): #6 images given
        path = Data + "/" + str(i) + ".png"
        image = cv2.imread(path)
        if image is not None:
            images.append(image)
        else:
            print("No image is found")

    #Feature Correspondence
    "We have 5 images and 4 matching .txt files"
    "nFeatures: (the number of feature points of the ith image - each following row specifies matches across images given a feature location in the ith image.)"
    "Each Row: (the number of matches for the jth feature) (Red Value) (Green Value) (Blue Value) (ucurrent image) (vcurrent image) (image id) (uimage id image) (vimage id image) (image id) (uimage id image) (v_{image id image}) …"

    feature_x, feature_y, feature_flag, feature_rgb_values =features_extraction(Data)
    # print(feature_x.shape, feature_y.shape, feature_flag.shape, feature_rgb_values.shape) We get (3833 feature points total)

    filtered_feature_flag = np.zeros_like(feature_flag) #np.zeros has limit which is solve by zeros_like
    f_matrix = np.empty(shape=(5,5), dtype=object)

    for i in range(0,4): #No of Images = 5
        for j in range(i+1,5):

            idx = np.where(feature_flag[:,i] & feature_flag[:,j])
            pts1 = np.hstack((feature_x[idx,i].reshape((-1,1)), feature_y[idx,i].reshape((-1,1))))
            pts2 = np.hstack((feature_x[idx,j].reshape((-1,1)), feature_y[idx,j].reshape((-1,1))))
            idx = np.array(idx).reshape(-1)
            
            if len(idx) > 8:
                F_inliers, inliers_idx = getInliers(pts1,pts2,idx)
                print("Between Images: ",i,"and",j,"NO of Inliers: ", len(inliers_idx), "/", len(idx) )
                f_matrix[i,j] = F_inliers
                filtered_feature_flag[inliers_idx,j] = 1
                filtered_feature_flag[inliers_idx,i] = 1
    
    
    print("######Obtained Feature Points after RANSAC#######")
    print("Starting with 1st 2 images")

    
    #Compute Essential Matrix, Estimate Pose, Triangulate
    F12 = f_matrix[0,1]

    
    #K is given
    K = np.array([[531.122155322710, 0 ,407.192550839899],[0, 531.541737503901, 313.308715048366],[0,0,1]])
    E12 = getEssentialMatrix(K,F12)

    #Estimating the Camera Pose
    R_set, C_set = ExtractCameraPose(E12)

    idx = np.where(filtered_feature_flag[:,0] & filtered_feature_flag[:,1])
    pts1 = np.hstack((feature_x[idx,0].reshape((-1,1)), feature_y[idx,0].reshape((-1,1))))
    pts2 = np.hstack((feature_x[idx,1].reshape((-1,1)), feature_y[idx,1].reshape((-1,1))))

    R1_ = np.identity(3)
    C1_ = np.zeros((3,1))

    pts3D_4 = []
    for i in range(len(C_set)):
        x1 = pts1
        x2 = pts2
        X = linearTriangulation(K, C1_, R1_, C_set[i], R_set[i], x1, x2)

        #Now we get 4 poses, we need to select unique one with maximum positive depth points
        X = X/X[:,3].reshape(-1,1)
        pts3D_4.append(X)

    R_best, C_best, X = DisambiguatePose(R_set,C_set,pts3D_4)
    X = X/X[:,3].reshape(-1,1)

    #Non-Linear Triangulation
    print("######Performing Non-Linear Triangulation######")
    X_refined = NonLinearTriangulation(K,pts1,pts2,X,R1_,C1_,R_best,C_best)
    # print(X_refined.shape)
    X_refined = X_refined / X_refined[:,3].reshape(-1,1)
    # print(X_refined.shape)

    total_err1 = []
    for pt1, pt2, X_3d in zip(pts1,pts2,X):
        err1, err2 = ReProjectionError(X_3d,pt1,pt2,R1_,C1_,R_best,C_best,K)
        total_err1.append(err1+err2)
    
    mean_err1 = np.mean(total_err1)

    total_err2 = []
    for pt1, pt2, X_3d in zip(pts1,pts2,X_refined):
        err1, err2 = ReProjectionError(X_3d,pt1,pt2,R1_,C1_,R_best,C_best,K)
        total_err2.append(err1+err2)
    
    mean_err2 = np.mean(total_err2)

    print("Between images",0+1,1+1,"Before optimization Linear Triang: ", mean_err1, "After optimization Non-Linear Triang: ", mean_err2)

    "Resistering Cam 1 and 2"
    X_all = np.zeros((feature_x.shape[0],3))
    cam_indices = np.zeros((feature_x.shape[0],1), dtype = int)
    X_found = np.zeros((feature_x.shape[0],1), dtype = int)

    X_all[idx] = X[:,:3]
    X_found[idx] = 1
    cam_indices[idx] = 1
    X_found[np.where(X_all[:2]<0)] = 0

    C_set = []
    R_set = []

    C0 = np.zeros(3)
    R0 = np.identity(3)
    C_set.append(C0)
    R_set.append(R0)
    C_set.append(C_best)
    R_set.append(R_best)

    print("#########Registered Cam 1 and Cam 2 ############")

    for i in range(2,5):
        print("Registering Image: ", str(i+1))
        feature_idx_i = np.where(X_found[:,0] & filtered_feature_flag[:,i])
        if len(feature_idx_i[0]) < 8:
            print("Got ", len(feature_idx_i), "common points between X and ", i, "image")
            continue

        pts_i = np.hstack((feature_x[feature_idx_i, i].reshape(-1,1), feature_y[feature_idx_i, i].reshape(-1,1)))
        X = X_all[feature_idx_i,:].reshape(-1,3)

        ##### Here We starts PnP
        R_init, C_init = PnPRANSAC(K,pts_i,X, iter=1000, thresh=5)
        linear_error_pnp = reprojectionErrorPnP(X, pts_i, K, R_init, C_init)
        
        Ri, Ci = NonLinearPnP(K, pts_i, X, R_init, C_init)
        non_linear_error_pnp = reprojectionErrorPnP(X, pts_i, K, Ri, Ci)
        print("Initial linear PnP error: ", linear_error_pnp, " Final Non-linear PnP error: ", non_linear_error_pnp)

        C_set.append(Ci)
        R_set.append(Ri)
        print("Print ho bhadvya")
        ###### WE start with the triangulation
        
        for k in range(0,i):
            idx_X_pts = np.where(filtered_feature_flag[:,k] & filtered_feature_flag[:,i])
            idx_X_pts = np.asarray(idx_X_pts)
            idx_X_pts = np.squeeze(idx_X_pts)

            if (len(idx_X_pts)<8):
                continue

            x1 = np.hstack((feature_x[idx_X_pts,k].reshape(-1,1), feature_y[idx_X_pts,k].reshape(-1,1)))
            x2 = np.hstack((feature_x[idx_X_pts,i].reshape(-1,1), feature_y[idx_X_pts,i].reshape(-1,1)))

            # print(x1.shape,x2.shape)
            # print(np.array(R_set[k]).shape,C_set[k])
            X_d = linearTriangulation(K,C_set[k],R_set[k],Ci,Ri,x1,x2)
            # print("burr",X_d,X_d.shape)
            X_d = X_d/X_d[:,3].reshape(-1,1)
            # print("burr",X_d,X_d.shape)
            linear_err = []
            pts1 , pts2 = x1, x2
            for pt1, pt2, X_3d in zip(pts1,pts2,X_d):
                err1, err2 = ReProjectionError(X_3d,pt1,pt2,R_set[k],C_set[k],Ri,Ci,K)
                linear_err.append(err1+err2)
    
            mean_linear_err = np.mean(linear_err)
            # print(mean_linear_err)
            
            X = NonLinearTriangulation(K,x1,x2,X_d,R_set[k],C_set[k],Ri,Ci)
            # print(X.shape)
            X = X/X[:,3].reshape(-1,1)
            
            non_linear_err = []
            for pt1, pt2, X_3d in zip(pts1,pts2,X):
                err1, err2 = ReProjectionError(X_3d,pt1,pt2,R_set[k],C_set[k],Ri,Ci,K)
                non_linear_err.append(err1+err2)
    
            mean_nonlinear_err = np.mean(non_linear_err)
            print("Linear Triang error: ", mean_linear_err, "Non-linear Triang error: ", mean_nonlinear_err)

            X_all[idx_X_pts] = X[:,:3]
            X_found[idx_X_pts] = 1

            print("Appended", idx_X_pts[0], "Points Between ", k, "and ",i )

            
            ##Visibility Matrix
            X_index, visibility_matrix = getObservationsIndexAndVizMat(X_found,filtered_feature_flag,nCam=i)
            # print(X_index,visibility_matrix)
            
            ##Bundle Adjustment
            print("########Bundle Adjustment Started")
            R_set_, C_set_, X_all = BundleAdjustment(X_index, visibility_matrix,X_all,X_found,feature_x,feature_y,filtered_feature_flag,R_set,C_set,K,nCam=i)
            # print(np.array(R_set).shape,np.array(C_set).shape,X_all.shape)
            
            for k in range(0,i+1):
                idx_X_pts = np.where(X_found[:,0] & filtered_feature_flag[:,k])
                x = np.hstack((feature_x[idx_X_pts,k].reshape(-1,1), feature_y[idx_X_pts,k].reshape(-1,1)))
                X = X_all[idx_X_pts]
                BundAdj_error = reprojectionErrorPnP(X,x,K,R_set_[k],C_set_[k])
                print("########Error after Bundle Adjustment: ", BundAdj_error)

            print("############Regestired camera: ", i+1,"############################")

    X_found[X_all[:,2]<0] = 0
    print("#############DONE###################")

    feature_idx = np.where(X_found[:,0])
    X = X_all[feature_idx]
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]

    #####2D Plotting
    fig = plt.figure(figsize = (10,10))
    plt.xlim(-4,6)
    plt.ylim(-2,12)
    plt.scatter(x,z,marker='.',linewidths=0.5, color = 'blue')
    for i in range(0, len(C_set_)):
        R1 = getEuler(R_set_[i])
        R1 = np.rad2deg(R1)
        plt.plot(C_set_[i][0],C_set_[i][2], marker=(3,0, int(R1[1])), markersize=15, linestyle='None')

    plt.savefig(Output+'2D.png')
    plt.show()


    ######3D Plotting
    fig1= plt.figure(figsize= (5,5))
    ax = plt.axes(projection="3d")
    ax.scatter3D(x,y,z,color="green")
    plt.show()
    plt.savefig(Output+'3D.png')



if __name__ == '__main__':
    main()
