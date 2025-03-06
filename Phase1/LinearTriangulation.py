import numpy as np

def linear_triangulation(K, R1, C1, R2, C2, points1, points2):
    """
    Triangulates 3D points from two sets of corresponding 2D points and camera parameters.
    
    Args:
        K: Camera intrinsic matrix (3x3)
        R1, C1: Rotation matrix and camera center for first camera
        R2, C2: Rotation matrix and camera center for second camera
        points1: 2D points in first image (Nx2)
        points2: 2D points in second image (Nx2)
        
    Returns:
        X: Triangulated 3D points (Nx3)
    """
    # Ensure inputs are properly shaped
    C1 = C1.reshape(3, 1)
    C2 = C2.reshape(3, 1)
    
    # Compute projection matrices
    P1 = K @ np.hstack((R1, -R1 @ C1))
    P2 = K @ np.hstack((R2, -R2 @ C2))
    print("This si my P2", P2)
    # Get number of points to triangulate
    num_points = points1.shape[0]
    X = np.zeros((num_points, 3))
    
    # Extract rows from projection matrices for efficiency
    p1_1, p1_2, p1_3 = P1[0], P1[1], P1[2]
    p2_1, p2_2, p2_3 = P2[0], P2[1], P2[2]
    
    # Triangulate each point
    for i in range(num_points):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        
        # Build the A matrix for DLT
        A = np.vstack([
            y1 * p1_3 - p1_2,
            p1_1 - x1 * p1_3,
            y2 * p2_3 - p2_2,
            p2_1 - x2 * p2_3
        ])
        
        # Solve for X using SVD
        _, _, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1]
        # print(X_homogeneous)
        # Convert to inhomogeneous coordinates
        X[i] = X_homogeneous[:3] / X_homogeneous[3]
        # print(X[i])
    return X


def triangulate_points(R1, C1, R2, C2, points1,points2, K):

    # Compute the projection matrices for the two cameras
    Translation1 = - R1 @ C1
    P1 = K @ np.hstack((R1, Translation1)) # 3x4

    Translation2 = - R2 @ C2
    P2 = K @ np.hstack((R2, Translation2))
    # print(P2)
    
    I = np.identity(3)
    P1 = np.dot(K, np.dot(R1, np.hstack((I,-C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I,-C2))))
    # print(P2)
    
    p1T = P1[0,:].reshape(1,4)
    p2T = P1[1,:].reshape(1,4)
    p3T = P1[2,:].reshape(1,4)

    p_1T = P2[0,:].reshape(1,4)
    p_2T = P2[1,:].reshape(1,4)
    p_3T = P2[2,:].reshape(1,4)
    
    image1_uv = points1
    image2_uv = points2

    # Convert homogeneous coordinates to 3D points
    world_points = []
    
    for i in range(len(points1)):
        
        x = image1_uv[i,0] 
        y = image1_uv[i,1]
        x_ = image2_uv[i,0]
        y_ = image2_uv[i,1]

        A = []
        A.append((y * p3T) - p2T)
        A.append(p1T - (x * p3T))
        A.append((y_ * p_3T) - p_2T)
        A.append(p_1T - (x_ * p_3T))

        A = np.array(A).reshape(4,4)

        _,_,vt = np.linalg.svd(A)
        v = vt.T
        x = v[:,-1]
        world_points.append(x)

    X = np.array(world_points)
    X = X/X[:,3].reshape(-1,1)
    return X

def linear_triangulation_batch(P1, P2, points1, points2):
    """
    Triangulates 3D points from two sets of corresponding 2D points and projection matrices.
    This version accepts projection matrices directly for more flexibility.
    
    Args:
        P1: Projection matrix for first camera (3x4)
        P2: Projection matrix for second camera (3x4)
        points1: 2D points in first image (Nx2)
        points2: 2D points in second image (Nx2)
        
    Returns:
        X: Triangulated 3D points (Nx3)
    """
    # Get number of points to triangulate
    num_points = points1.shape[0]
    X = np.zeros((num_points, 3))
    
    # Extract rows from projection matrices for efficiency
    p1_1, p1_2, p1_3 = P1[0], P1[1], P1[2]
    p2_1, p2_2, p2_3 = P2[0], P2[1], P2[2]
    
    # Triangulate each point
    for i in range(num_points):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        
        # Build the A matrix for DLT
        A = np.vstack([
            y1 * p1_3 - p1_2,
            p1_1 - x1 * p1_3,
            y2 * p2_3 - p2_2,
            p2_1 - x2 * p2_3
        ])
        
        # Solve for X using SVD
        _, _, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1]
        
        # Convert to inhomogeneous coordinates
        X[i] = X_homogeneous[:3] / X_homogeneous[3]
    
    return X

