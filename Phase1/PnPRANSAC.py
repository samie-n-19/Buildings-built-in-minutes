

import numpy as np
from LinearPnP import linear_pnp

def get_projection_matrix(K, R, C):
    """
    Constructs the projection matrix P = K[R|-RC]
    
    Args:
        K: Camera intrinsic matrix
        R: Rotation matrix
        C: Camera center
    
    Returns:
        P: Projection matrix
    """
    C = np.reshape(C, (3, 1))
    # I = np.identity(3)
    P = K  @ np.hstack((R, -R @ C))
    return P

def compute_reprojection_error(P, point_2d, point_3d):
    """
    Computes reprojection error between projected and actual 2D points
    
    Args:
        P: Projection matrix
        point_2d: 2D image point
        point_3d: 3D world point
    
    Returns:
        error: Reprojection error
    """
    # Extract rows of P
    p1, p2, p3 = P[0], P[1], P[2]
    
    # Add homogeneous coordinate to 3D point
    X_homog = np.append(point_3d, 1)
    
    # Project 3D point to image
    u_proj = np.dot(p1, X_homog) / np.dot(p3, X_homog)
    v_proj = np.dot(p2, X_homog) / np.dot(p3, X_homog)
    
    # Compute error
    u, v = point_2d
    error = (u - u_proj)**2 + (v - v_proj)**2
    error = np.sqrt(error)
    return error

def pnp_ransac(X_world, x_image, K, num_iterations=1000, error_thresh=100.0):
    """
    Estimates camera pose using PnP with RANSAC to remove outliers.
    
    Args:
        X_world: (Nx3) 3D world points
        x_image: (Nx2) Corresponding 2D image points
        K: (3x3) Camera intrinsic matrix
        num_iterations: Number of RANSAC iterations
        error_thresh: Maximum reprojection error to be considered an inlier
    
    Returns:
        best_R: (3x3) Best estimated rotation matrix
        best_C: (3x1) Best estimated camera center
        inliers: Indices of inlier correspondences
    """
    best_R, best_C = None, None
    best_inliers = []
    best_error = float('inf')
    
    for i in range(num_iterations):
        # Check if we have enough points
        if len(X_world) < 6:
            print(f" Not enough points for PnP RANSAC! Only {len(X_world)} available.")
            return None, None, []
        
        # Randomly select 6 points
        sample_idx = np.random.choice(len(X_world), 6, replace=False)
        
        # Estimate pose using Linear PnP
        R_temp, C_temp = linear_pnp(X_world[sample_idx], x_image[sample_idx], K)
        # print("THIS IS R_TEMP", R_temp)
        # print("THIS IS C_TEMP", C_temp)
        # Construct projection matrix
        # P = get_projection_matrix(K, R_temp, C_temp)
        P = K @ np.hstack((R_temp, -R_temp @ C_temp.reshape(3, 1)))
        # print(P)
        
        # Compute errors and find inliers
        inliers = []
        errors = []
        
        for j in range(len(X_world)):
            error = compute_reprojection_error(P, x_image[j], X_world[j])
            errors.append(error)
            
            if error < error_thresh:
                inliers.append(j)
        
        # Update best model if more inliers found
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R, best_C = R_temp, C_temp
            best_error = np.mean(errors)
    
    # Recompute final pose using all inliers if we found any
    if len(best_inliers) > 0:
        inlier_X = X_world[best_inliers]
        inlier_x = x_image[best_inliers]
        best_R, best_C = linear_pnp(inlier_X, inlier_x, K)
    
    print(f"PnP RANSAC found {len(best_inliers)} inliers out of {len(X_world)} points")
    print(f"Mean reprojection error: {best_error:.4f}")
    
    return best_R, best_C, best_inliers