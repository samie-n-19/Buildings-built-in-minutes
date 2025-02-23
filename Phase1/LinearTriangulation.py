import numpy as np

def linear_triangulation(P1, P2, points1, points2):
    """
    Performs linear triangulation using the Direct Linear Transformation (DLT) method.

    Args:
        P1: (3x4) numpy array representing the first camera projection matrix.
        P2: (3x4) numpy array representing the second camera projection matrix.
        points1: (Nx2) numpy array of 2D points in the first image.
        points2: (Nx2) numpy array of corresponding 2D points in the second image.

    Returns:
        X: (Nx3) numpy array of triangulated 3D points in homogeneous coordinates.
    """
    num_points = points1.shape[0]
    X = np.zeros((num_points, 3))  # 3D points storage

    for i in range(num_points):
        x1, y1 = points1[i]
        x2, y2 = points2[i]

        # Construct the linear system AX = 0
        A = np.array([
            x1 * P1[2] - P1[0],  # x1 * P1 row3 - P1 row1
            y1 * P1[2] - P1[1],  # y1 * P1 row3 - P1 row2
            x2 * P2[2] - P2[0],  # x2 * P2 row3 - P2 row1
            y2 * P2[2] - P2[1],  # y2 * P2 row3 - P2 row2
        ])

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1]  # Solution is the last row of Vt

        # Convert to inhomogeneous coordinates
        X[i] = X_homogeneous[:3] / X_homogeneous[3]  # Normalize by last coordinate

    return X