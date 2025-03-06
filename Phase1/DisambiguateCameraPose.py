
import numpy as np

def disambiguate_camera_pose(camera_poses, triangulated_points):
    """ Selects the correct camera pose by checking the cheirality condition """
    max_points_in_front = -1
    best_camera_pose = None
    best_world_points = None
    
    for i, (R, C) in enumerate(camera_poses):
        # Filter valid points before checking cheirality
        valid_points = [X for X in triangulated_points[i] if X[2] > 0 and np.linalg.norm(X) < 100]
        
        # If valid points exist, check cheirality condition
        num_points_in_front = check_cheirality(R, C, np.array(valid_points))
        
        if num_points_in_front > max_points_in_front:
            max_points_in_front = num_points_in_front
            best_camera_pose = (R, C)
            best_world_points = np.array(valid_points)  # Store filtered 3D points
    
    return best_camera_pose, best_world_points

def check_cheirality(R, C, X):
    """ Checks if the triangulated 3D points are in front of the camera """
    num_points_in_front = 0
    r3 = R[2, :].reshape(1, 3)  # Extract third row of R
    
    for i in range(X.shape[0]):
        X_homogeneous = X[i].reshape(-1, 1)
        condition = r3 @ (X_homogeneous[:3] - C)
        
        if condition > 0 and X_homogeneous[2] > 0:
            num_points_in_front += 1
    # print(f"Number of points in front: {num_points_in_front}")
    return num_points_in_front
