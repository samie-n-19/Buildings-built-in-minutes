import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix

def projection_matrix(K, R, C):
    """
    Compute the projection matrix P = K[R | -RC]
    """
    C = np.reshape(C, (3, 1))
    P = np.dot(K, np.dot(R, np.hstack((np.eye(3), -C))))
    return P


def camera_index(v):
    idx_camer = []
    idx_point = []
    for i, row in enumerate(v):
        for j, visible in enumerate(row):
            if visible:
                idx_camer.append(j)
                idx_point.append(i)
    return np.array(idx_camer), np.array(idx_point)

def bundle_adjustment_sparsity(n_cameras, n_points, v):
    camera_indices, point_indices = camera_index(v)
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    # n_observations = np.sum(visibility_matrix)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
    A = A.tocsc()

    return A

def error_for_reprojection(params, uv, v, num_cameras, num_points, camera_matrix):
    # Extract camera parameters and world points from params
    camera_angles = params[:num_cameras * 3].reshape((num_cameras, 3))
    camera_rm = [R.from_euler('zyx', r).as_matrix() for r in camera_angles]
    camera_rm = np.array(camera_rm)
    # print(f"Camera rotation matrices: {camera_rotation_matrices.shape}")
    camera_t = params[num_cameras * 3:num_cameras * 6].reshape((num_cameras, 3))
    world_points = params[num_cameras * 6:].reshape((num_points, 3))
    
    error = []
    E1 = 0
    for point_idx, visibility_row in enumerate(v):
        world = np.append(world_points[point_idx], 1)
        # print(visibility_row)
        for image_idx, visible in enumerate(visibility_row):
            if visible: 
                img_2d_points = uv[:,point_idx, image_idx]
                # Extract and convert camera parameters (rotation and translation)
                rotation_matrix = camera_rm[image_idx]
                translation = camera_t[image_idx]

                P = camera_matrix @ np.hstack((rotation_matrix, -rotation_matrix @ translation.reshape(3, 1)))
            
                p1_1T, p1_2T, p1_3T = P # rows of P
                p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)
                
                u1,v1 = img_2d_points[0], img_2d_points[1]
                u1_proj = np.divide(p1_1T.dot(world) , p1_3T.dot(world))
                v1_proj =  np.divide(p1_2T.dot(world) , p1_3T.dot(world))

                E1x = np.square(v1 - v1_proj)
                E1y = np.square(u1 - u1_proj)
                E1 += E1x + E1y
                error.append(E1x)
                error.append(E1y)
    return np.array(error).ravel()

def bundle_adjustment(uv, world_points, v, R_All, C_All, num_cameras, camera_matrix):
    """
    Perform bundle adjustment to refine camera poses and 3D points.
    """
    num_points = world_points.shape[0]
    
    R_euler = np.array([R.from_matrix(R_mat).as_euler('xyz') for R_mat in R_All])
    
    initial_params = np.hstack((R_euler.ravel(), C_All.ravel(), world_points.ravel()))
    
    A = bundle_adjustment_sparsity(num_cameras, num_points, v)
    
    result = least_squares(
        error_for_reprojection, 
        initial_params, 
        jac_sparsity=A, 
        method="dogbox", #dogbox 
        ftol=1e-20, 
        xtol=1e-20,
        gtol=1e-15,
        verbose=2, 
        args=(uv, v, num_cameras, num_points, camera_matrix)
    )
    
    optimized_params = result.x
    R_optimized = np.array([R.from_euler('xyz', r).as_matrix() for r in optimized_params[:num_cameras * 3].reshape((num_cameras, 3))])
    C_optimized = optimized_params[num_cameras * 3:num_cameras * 6].reshape((num_cameras, 3))
    world_points_optimized = optimized_params[num_cameras * 6:].reshape((num_points, 3))
    
    return R_optimized, C_optimized, world_points_optimized
