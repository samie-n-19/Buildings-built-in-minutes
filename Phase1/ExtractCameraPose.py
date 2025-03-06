import numpy as np

def extract_camera_pose(E):
#     """
#     Extracts candidate camera poses from the essential matrix E.
    
#     This function computes the SVD of E and then uses the standard method
#     to obtain two possible rotation matrices and one translation vector (up to scale).
#     The four candidate solutions are:
#         (R1, +t), (R1, -t), (R2, +t), and (R2, -t),
#     where R1 = U * W * V^T and R2 = U * W^T * V^T, with:
#         W = [[0, -1, 0],
#              [1,  0, 0],
#              [0,  0, 1]].
    
#     Args:
#         E (np.ndarray): A 3x3 essential matrix.
    
#     Returns:
#         list: A list of four tuples, each containing a rotation matrix (3x3)
#               and a translation vector (3,). Note that the translation is only
#               determined up to scale.
#     """
#     U, _, Vt = np.linalg.svd(E)
#     W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
#     # Compute possible rotations
#     R1 = U @ W @ Vt
#     R2 = U @ W @ Vt
#     R3 = U @ W.T @ Vt.T
#     R4 = U @ W.T @ Vt.T

#     # Possible translation vectors
#     C1 = U[:, 2].reshape(3, 1)
#     C2 = -U[:, 2].reshape(3, 1)
#     C3 = U[:, 2].reshape(3, 1)
#     C4 = -U[:, 2].reshape(3, 1)

#     # Ensure valid rotations
#     if np.linalg.det(R1) < 0:
#         R1 = -R1
#         C1 = -C1
#     if np.linalg.det(R2) < 0:
#         R2 = -R2
#         C2 = -C2
#     if np.linalg.det(R3) < 0:
#         R3 = -R3
#         C3 = -C3
#     if np.linalg.det(R4) < 0:
#         R4 = -R4
#         C4 = -C4        

    
#     # Store camera poses in candidates list
#     candidates = [(R1, C1), (R2, C2), (R3, C3), (R4, C4)]
    
#     return candidates


# def get_camera_poses(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Four possible camera rotations
    R1 = np.dot(np.dot(U, W), Vt)
    R2 = np.dot(np.dot(U, W), Vt)
    R3 = np.dot(np.dot(U, W.T), Vt)
    R4 = np.dot(np.dot(U, W.T), Vt)

    # Four possible camera positions
    C1 = U[:, 2].reshape(3, 1)
    C2 = -U[:, 2].reshape(3, 1)
    C3 = U[:, 2].reshape(3, 1)
    C4 = -U[:, 2].reshape(3, 1)
    
    # check if determinant of all R is negative
    if np.linalg.det(R1) < 0:
        R1 = -R1
        C1 = -C1
    if np.linalg.det(R2) < 0:
        R2 = -R2
        C2 = -C2
    if np.linalg.det(R3) < 0:
        R3 = -R3
        C3 = -C3
    if np.linalg.det(R4) < 0:
        R4 = -R4
        C4 = -C4

    return [(R1, C1), (R2, C2), (R3, C3), (R4, C4)]