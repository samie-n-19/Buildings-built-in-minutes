import numpy as np
def essential_matrix_from_fundamental(F, K):
    """
    Compute the essential matrix from the fundamental matrix F and camera intrinsic matrix K.

    Args:
        F: (3, 3) numpy array representing the fundamental matrix.
        K: (3, 3) numpy array representing the camera intrinsic matrix.

    Returns:
        E: (3, 3) numpy array representing the essential matrix.
    """
    # Compute the essential matrix using the relation: E = K^T * F * K
    E = K.T @ F @ K

    # Enforce the singular value constraint: E should have singular values (s, s, 0)
    U, S, Vt = np.linalg.svd(E)
    
    # Compute the average of the first two singular values
    s_avg = (S[0] + S[1]) / 2.0
    S_new = [s_avg, s_avg, 0]
    
    # Reconstruct the essential matrix with the adjusted singular values
    E_corrected = U @ np.diag(S_new) @ Vt

    return E_corrected