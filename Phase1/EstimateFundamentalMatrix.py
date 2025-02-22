import numpy as np

def normalize_points(points):
    if points.shape[0] == 0:
        raise ValueError("Empty points array")
    if points.shape[1] != 2:
        raise ValueError(f"Expected points to have 2 columns, but got {points.shape[1]}")
    
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    scale = np.sqrt(2) / np.mean(np.linalg.norm(points_centered, axis=1))
    T = np.array([[scale, 0, -scale*centroid[0]],
                  [0, scale, -scale*centroid[1]],
                  [0, 0, 1]])
    points_homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    normalized_points = (T @ points_homogeneous.T).T[:, :2]
    return normalized_points, T

def estimate_fundamental_matrix(points1, points2):
    normalized_points1, T1 = normalize_points(points1)
    normalized_points2, T2 = normalize_points(points2)

    A = np.zeros((len(points1), 9))
    for i in range(len(points1)):
        x1, y1 = normalized_points1[i]
        x2, y2 = normalized_points2[i]
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V

    F = T2.T @ F @ T1
    return F / F[2, 2]

