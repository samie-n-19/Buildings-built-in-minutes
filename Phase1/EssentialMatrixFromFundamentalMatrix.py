import numpy as np
np.random.seed(100)

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
    
    # Enforce the singular value constraint for essential matrix
    U, S, Vt = np.linalg.svd(E)
    
    # Essential matrix should have two equal singular values and one zero
    # Set singular values to [1, 1, 0]
    S_new = np.array([1, 1, 0])
    
    # Reconstruct the essential matrix with the corrected singular values
    E_corrected = U @ np.diag(S_new) @ Vt
    
    return E_corrected
