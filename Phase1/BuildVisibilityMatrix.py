
def BuildVisibilityMatrix(corresponding_points_dict, world_points_dict, matches_dict, num_of_images):
    """
    Build the visibility matrix for bundle adjustment.
    
    Args:
        corresponding_points_dict: Dictionary with keys 0 to len(world_points)-1, containing 2D pixel coordinates (u,v) for the first image
        world_points_dict: Dictionary with keys 0 to len(world_points)-1, containing 3D world point coordinates
        matches_dict: Dictionary with keys like (1,2), (1,2,3), etc. containing feature matches across images
        num_of_images: Number of images/cameras in the system
        
    Returns:
        V: Binary visibility matrix of shape (num_of_images, len(world_points_dict))
        UV_matrix: Matrix of UV values with same shape as V, containing pixel coordinates where points are visible
    """
    import numpy as np
    
    # Initialize visibility matrix with zeros
    num_points = len(world_points_dict)
    V = np.zeros((num_of_images, num_points), dtype=int)
    
    # Initialize UV matrix with zeros (or NaN)
    UV_matrix = np.zeros((num_of_images, num_points, 2), dtype=float)
    
    # For each 3D point
    for point_idx in range(num_points):
        # All points are visible in the first image (from corresponding_points_dict)
        V[0, point_idx] = 1
        
        # Get the 2D coordinates in the first image and store in UV matrix
        if point_idx in corresponding_points_dict:
            point_2d = corresponding_points_dict[point_idx]
            UV_matrix[0, point_idx] = point_2d
            
            # Search through matches_dict to find in which other images this point is visible
            for key, matches in matches_dict.items():
                for match in matches:
                    # Check if the first element of the match corresponds to image 1 and has the same coordinates
                    if match[0][0] == 1 and abs(match[0][1] - point_2d[0]) < 1e-5 and abs(match[0][2] - point_2d[1]) < 1e-5:
                        # If found, mark this point as visible in all other images in this match
                        for img_match in match[1:]:  # Skip the first image
                            img_idx = img_match[0] - 1  # Convert to 0-based indexing
                            if 0 <= img_idx < num_of_images:
                                V[img_idx, point_idx] = 1
                                # Store the UV values for this point in this image
                                UV_matrix[img_idx, point_idx] = [img_match[1], img_match[2]]
    
    return V.T, UV_matrix.T

