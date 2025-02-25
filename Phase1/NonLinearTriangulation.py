import numpy as np
from scipy.optimize import least_squares

def project_point(P, X):
    """
    Projects a 3D point X into 2D using projection matrix P.
    
    Args:
        P: (3x4) Camera projection matrix.
        X: (3,) or (4,) 3D point in homogeneous coordinates.
    
    Returns:
        (2,) Reprojected 2D point.
    """
    X_homog = np.append(X, 1)  # Convert to homogeneous if not already
    x_proj = P @ X_homog
    return x_proj[:2] / x_proj[2]  # Convert back to 2D

def reprojection_error(X, P1, P2, points1, points2):
    """
    Computes the reprojection error between observed and projected points.
    
    Args:
        X: (3N,) Flattened array of 3D points.
        P1, P2: (3x4) Projection matrices.
        points1, points2: (Nx2) Observed 2D points.
    
    Returns:
        Residuals (error terms).
    """
    N = len(points1)
    X = X.reshape((N, 3))  # Reshape to (N, 3)
    errors = []

    for i in range(N):
        x1_proj = project_point(P1, X[i])
        x2_proj = project_point(P2, X[i])
        
        # Compute squared error to ensure non-negativity
        err1 = np.linalg.norm(points1[i] - x1_proj) ** 2
        err2 = np.linalg.norm(points2[i] - x2_proj) ** 2
        
        errors.extend([err1, err2]) 

    return np.array(errors)

def nonlinear_triangulation(P1, P2, points1, points2, X_init):
    print("✅ Entering Non-Linear Triangulation Optimization...")

    # Debug: Check matrix shapes
    print(f"P1 Shape: {P1.shape}, P2 Shape: {P2.shape}")
    print(f"Points1 Shape: {points1.shape}, Points2 Shape: {points2.shape}")
    print(f"Initial X Shape: {X_init.shape}")

    # Remove points with negative depth and large errors
    valid_idx = (X_init[:, 2] > 0) & (np.linalg.norm(X_init, axis=1) < 100)  # Remove very far points
    X_init = X_init[valid_idx]
    points1 = points1[valid_idx]
    points2 = points2[valid_idx]

    if len(X_init) == 0:
        print("🚨 No valid 3D points remaining after filtering! Exiting.")
        return None

    # 🔹 Select only the first 100 points for faster optimization
    num_points = min(len(X_init), 100)
    X_init = X_init[:num_points]
    points1 = points1[:num_points]
    points2 = points2[:num_points]


    # Flatten initial 3D points for optimizer
    X_init_flat = X_init.flatten()

    # Compute initial reprojection error before optimization
    initial_error = np.sum(reprojection_error(X_init.flatten(), P1, P2, points1, points2))
    print(f"📌 Initial Reprojection Error: {initial_error}")

    initial_residuals = reprojection_error(X_init_flat, P1, P2, points1, points2)
    print(f"Initial residuals size: {initial_residuals.shape}")


    # Flatten the initial 3D points for optimizer
    # X_init_flat = X_init.flatten()

    print("First 5 initial triangulated points before optimization:\n", X_init[:5])

# Check if any points have large or negative values
    if np.any(np.abs(X_init) > 1000):
        print("🚨 Warning: Large 3D points detected! Rescaling might be needed.")
    if np.any(X_init[:, 2] < 0):
        print("🚨 Warning: Some 3D points have negative depth! Check triangulation.")


    result = least_squares(reprojection_error, X_init_flat, args=(P1, P2, points1, points2), method='trf', max_nfev=15, xtol=1e-4, ftol=1e-4, verbose=1)

    # Compute final reprojection error after optimization
    final_X = result.x.reshape((-1, 3))
    final_error = np.sum(reprojection_error(final_X.flatten(), P1, P2, points1, points2))
    print(f"✅ Final Reprojection Error: {final_error}")

    return final_X  # Reshape back to (Nx3)


