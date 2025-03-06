import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from PnPRANSAC import *

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

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
    C = C.reshape(3, 1)
    P = K @ np.hstack((R, -R @ C))
    return P

def reprojection_error_pnp(P, point_2d, point_3d):
    """
    Computes reprojection error between projected and actual 2D points
    
    Args:
        P: Projection matrix
        point_2d: 2D image point
        point_3d: 3D world point
    
    Returns:
        error: Reprojection error
    """
    # Add homogeneous coordinate to 3D point
    X_homog = np.append(point_3d, 1)
    
    # Project 3D point to image
    x_proj = P @ X_homog
    x_proj = x_proj[:2] / x_proj[2]
    
    # Compute error
    error = np.linalg.norm(point_2d - x_proj, ord = 2)
    return error

def reprojection_loss_pnp(params, points, world_points, K):
    """
    Computes the mean reprojection error for all points
    
    Args:
        params: Combined camera center and quaternion [C, q]
        points: 2D image points
        world_points: 3D world points
        K: Camera intrinsic matrix
    
    Returns:
        Mean reprojection error
    """
    C = params[:3]
    q = params[3:]
    
    # Convert quaternion to rotation matrix
    R = Rotation.from_quat(q).as_matrix()
    
    # Get projection matrix
    # P = get_projection_matrix(K, R, C)
    P = K @ np.hstack((R, -R @ C.reshape(3, 1)))    
    
    # Compute errors for all points
    errors = []
    for i in range(len(points)):
        error = reprojection_error_pnp(P, points[i], world_points[i])
        errors.append(error)
    
    return errors

def nonlinear_pnp(X_world, x_image, K, R_init, t_init):
    """
    Optimizes camera pose using non-linear least squares to minimize reprojection error.
    
    Args:
        X_world: (Nx3) 3D points in world coordinates.
        x_image: (Nx2) Corresponding 2D points in image coordinates.
        K: (3x3) Camera intrinsic matrix.
        R_init: (3x3) Initial rotation matrix.
        t_init: (3x1) Initial translation vector.
        
    Returns:
        R_opt: (3x3) Optimized rotation matrix.
        C_opt: (3x1) Optimized camera center.
    """
    # Convert R to quaternion for optimization
    q_init = Rotation.from_matrix(R_init).as_quat()
    t_init = t_init.flatten()
    # Convert translation to camera center: C = -R^T * t
    # C_init = -R_init.T @ t_init
    # C_init = C_init.flatten()
    
    # Combine into a single parameter array
    initial_params = np.hstack((t_init, q_init))
    
    # Run optimization
    result = least_squares(
        reprojection_loss_pnp, 
        initial_params, 
        args=[x_image, X_world, K], 
        # method='trf',
        # max_nfev=5000
    )
    
    # Retrieve optimized parameters
    params = result.x
    C_opt = params[:3].reshape(3, 1)
    R_opt = Rotation.from_quat(params[3:]).as_matrix()
    
    # Convert back to translation if needed
    # t_opt = -R_opt @ C_opt
    
    print(f"Non-Linear PnP optimization completed")
    # print(f" Initial parameters: C={q_init_init}, R={R_init}")
    # print(f" Optimized parameters: C={t_opt.T}, R={R_opt}")
    
    return R_opt, C_opt
