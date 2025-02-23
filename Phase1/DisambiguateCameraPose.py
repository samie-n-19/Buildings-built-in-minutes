import numpy as np
from LinearTriangulation import linear_triangulation

def disambiguate_camera_pose(candidate_poses, P1, points1, points2, K):
    """
    Selects the correct camera pose by checking the cheirality condition.
    
    Args:
        candidate_poses: List of four tuples (R, C), each containing:
                        - R: (3x3) Rotation matrix
                        - C: (3x1) Camera center
        P1: (3x4) Projection matrix for the first camera.
        points1: (Nx2) 2D points in the first image.
        points2: (Nx2) Corresponding 2D points in the second image.
        K: (3x3) Intrinsic camera matrix.

    Returns:
        best_R: (3x3) Correct rotation matrix.
        best_C: (3x1) Correct camera center.
        best_X: (Nx3) The best set of 3D points.
    """
    best_pose = None
    max_valid_points = 0
    best_X = None

    for R, C in candidate_poses:
        # Compute the projection matrix P2 for this camera pose
        C = C.reshape((3, 1))  # Ensure C is a column vector
        P2 = K @ np.hstack((R, -R @ C))  # P2 = K[R | -RC]

        # Triangulate 3D points using the given P1 and this P2
        X = linear_triangulation(P1, P2, points1, points2)

        # Check cheirality condition: Points must be in front of both cameras
        valid_points = 0
        for i in range(X.shape[0]):
            X_homogeneous = np.hstack((X[i], 1))  # Convert to homogeneous coordinates

            # Check if the point is in front of both cameras
            z1 = P1[2] @ X_homogeneous  # Depth w.r.t. first camera
            z2 = P2[2] @ X_homogeneous  # Depth w.r.t. second camera

            if z1 > 0 and z2 > 0:
                valid_points += 1  # This is a valid camera pose

        # Keep the pose with the maximum number of valid 3D points
        if valid_points > max_valid_points:
            max_valid_points = valid_points
            best_pose = (R, C)
            best_X = X

    best_R, best_C = best_pose
    return best_R, best_C, best_X