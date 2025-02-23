import numpy as np

def extract_camera_pose(E):
    """
    Extracts candidate camera poses from the essential matrix E.
    
    This function computes the SVD of E and then uses the standard method
    to obtain two possible rotation matrices and one translation vector (up to scale).
    The four candidate solutions are:
        (R1, +t), (R1, -t), (R2, +t), and (R2, -t),
    where R1 = U * W * V^T and R2 = U * W^T * V^T, with:
        W = [[0, -1, 0],
             [1,  0, 0],
             [0,  0, 1]].
    
    Args:
        E (np.ndarray): A 3x3 essential matrix.
    
    Returns:
        list: A list of four tuples, each containing a rotation matrix (3x3)
              and a translation vector (3,). Note that the translation is only
              determined up to scale.
    """
    # Compute the SVD of the essential matrix.
    U, S, Vt = np.linalg.svd(E)
    
    # Ensure that U and Vt represent proper rotations (determinant +1)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Define the special matrix W.
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    
    # Compute the two possible rotation matrices.
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # Check for reflections and correct if necessary.
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    
    # The translation vector is the third column of U (up to scale).
    t = U[:, 2]
    
    # Form the four candidate camera poses.
    candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    
    return candidates