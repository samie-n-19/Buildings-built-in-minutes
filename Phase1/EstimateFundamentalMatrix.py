import numpy as np
np.random.seed(100)


def normalize_points(points):
    """
    Normalize points to improve numerical stability
    
    Args:
        points: Nx2 array of points
        
    Returns:
        normalized_points: Nx2 array of normalized points
        T: 3x3 transformation matrix
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Shift points to have centroid at origin
    shifted_points = points - centroid
    
    # Calculate average distance from origin
    avg_dist = np.mean(np.sqrt(np.sum(shifted_points**2, axis=1)))
    
    # Scale factor to make average distance sqrt(2)
    scale = np.sqrt(2) / avg_dist if avg_dist > 0 else 1.0
    
    # Transformation matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    # Apply transformation
    normalized_points = np.zeros_like(points)
    for i, point in enumerate(points):
        p_homogeneous = np.append(point, 1)
        p_transformed = T @ p_homogeneous
        normalized_points[i] = p_transformed[:2] / p_transformed[2]
    normalized_centroid = np.mean(normalized_points, axis=0)
    # print(f"Norma;ised centroid: {normalized_centroid}")
    return normalized_points, T

def estimate_fundamental_matrix(points1, points2):
    """
    Estimate the fundamental matrix from corresponding points
    using the normalized 8-point algorithm.
    
    Args:
        points1: Nx2 array of points in the first image
        points2: Nx2 array of points in the second image
        
    Returns:
        F: 3x3 fundamental matrix
    """
    # Normalize points
    points1_norm, T1 = normalize_points(points1)
    points2_norm, T2 = normalize_points(points2)
    
    # Build the constraint matrix
    A = np.zeros((len(points1), 9))
    for i in range(len(points1)):
        x1, y1 = points1_norm[i]
        x2, y2 = points2_norm[i]
        A[i] = [x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Solve for F using SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    # Enforce rank 2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    
    # Denormalize
    F = T2.T @ F @ T1
    
    # Normalize F
    # F = F / F[2, 2]
    F[0,0] = -F[0,0]
    
    return F

def compute_epipolar_distance(points1, points2, F):
    """
    Compute the symmetric epipolar distance between points and epipolar lines
    
    Args:
        points1: Nx2 array of points in the first image
        points2: Nx2 array of points in the second image
        F: 3x3 fundamental matrix
        
    Returns:
        distances: N array of distances
    """
    n = points1.shape[0]
    points1_homogeneous = np.hstack((points1, np.ones((n, 1))))
    points2_homogeneous = np.hstack((points2, np.ones((n, 1))))
    
    # Epipolar lines in the second image
    lines2 = points1_homogeneous @ F.T
    
    # Epipolar lines in the first image
    lines1 = points2_homogeneous @ F
    
    # Normalize lines
    norms1 = np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)
    norms2 = np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2)
    
    # Calculate distances
    distances1 = np.abs(np.sum(points1_homogeneous * lines1, axis=1)) / norms1
    distances2 = np.abs(np.sum(points2_homogeneous * lines2, axis=1)) / norms2
    
    # Symmetric distance
    distances = (distances1 + distances2) / 2
    
    return distances
