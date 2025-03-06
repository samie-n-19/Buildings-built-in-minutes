import numpy as np
from scipy.optimize import least_squares

def project_point(P, X):
    """
    Projects a 3D point X into 2D using projection matrix P.
    
    Args:
        P: 3x4 projection matrix
        X: 3D point (3,)
        
    Returns:
        2D projected point (2,)
    """
    X_homog = np.append(X, 1)  # Convert to homogeneous coordinates
    x_proj = P @ X_homog
    return x_proj[:2] / x_proj[2]  # Convert back to 2D

def reprojection_error(X, P1, P2, points1, points2):
    """
    Computes the reprojection error between observed and projected points.
    
    Args:
        X: Flattened array of 3D points
        P1, P2: Projection matrices for the two cameras
        points1, points2: Observed 2D points in the two images
        
    Returns:
        Array of reprojection errors
    """
    N = len(points1)
    X = X.reshape((N, 3))  # Reshape to (N, 3)
    
    # Vectorized computation of projected points
    X_homog = np.hstack((X, np.ones((N, 1))))
    
    # Project all points at once
    x1_proj = (P1 @ X_homog.T).T
    x2_proj = (P2 @ X_homog.T).T
    
    # Normalize homogeneous coordinates
    x1_proj = x1_proj[:, :2] / x1_proj[:, 2:3]
    x2_proj = x2_proj[:, :2] / x2_proj[:, 2:3]
    
    # Compute errors (flattened for least_squares)
    errors = np.vstack([
        points1 - x1_proj,
        points2 - x2_proj
    ]).flatten()
    
    return errors

def nonlinear_triangulation(P1, P2, points1, points2, X_init):
    """
    Optimizes 3D points using non-linear least squares with fast convergence.
    
    Args:
        P1, P2: 3x4 projection matrices for the two cameras
        points1, points2: Nx2 arrays of 2D points in the two images
        X_init: Nx3 array of initial 3D points from linear triangulation
        
    Returns:
        Nx3 array of refined 3D points
    """
    print("Starting Non-Linear Triangulation...")
    
    # Make sure inputs are numpy arrays
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    X_init = np.asarray(X_init)
    
    # Handle size mismatch if it exists
    min_size = min(len(X_init), len(points1), len(points2))
    X_init = X_init[:min_size]
    points1 = points1[:min_size]
    points2 = points2[:min_size]

    
    
    # Compute initial reprojection error
    initial_error = np.mean(reprojection_error(X_init.flatten(), P1, P2, points1, points2)**2)
    print(f"Initial mean reprojection error: {initial_error:.6f}")
    
    # Run non-linear optimization with early stopping
    result = least_squares(
        reprojection_error,
        X_init.flatten(),
        args=(P1, P2, points1, points2),
        method='trf',
        # loss='linear',  # Change to 'huber' if you need robustness to outliers
        # ftol=1e-3,      # Relaxed tolerance for faster convergence
        # xtol=1e-3,
        # gtol=1e-3,
        # max_nfev=100, 
        #    # Limit number of function evaluations for speed
        # verbose=2
    )
    
    # Reshape result back to Nx3 array
    X_refined = result.x.reshape(-1, 3)
    
    # Compute final reprojection error
    final_error = np.mean(reprojection_error(X_refined.flatten(), P1, P2, points1, points2)**2)
    print(f"Final mean reprojection error: {final_error:.6f}")
    print(f"Error reduction: {100 * (initial_error - final_error) / initial_error:.2f}%")
    
    return X_refined
