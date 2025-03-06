import numpy as np
import cv2
from EstimateFundamentalMatrix import *

import random
random.seed(429)

# from fundamental_matrix import estimate_fundamental_matrix, compute_epipolar_distance

def ransac_fundamental_matrix(matches_dict, pair, num_iterations=1000, threshold=10):
    """
    Estimate the fundamental matrix using RANSAC
    
    Args:
        matches_dict: Dictionary of matches from read_matching_files_modified
        pair: Tuple (i, j) specifying the image pair
        num_iterations: Number of RANSAC iterations
        threshold: Distance threshold for inliers
        
    Returns:
        best_F: Best fundamental matrix
        best_inliers: Indices of inlier matches
    """
    matches = matches_dict[pair]
    
    # Extract points from matches
    points1 = np.array([[match[0][1], match[0][2]] for match in matches])
    points2 = np.array([[match[1][1], match[1][2]] for match in matches])
    
    n_matches = len(matches)
    best_F = None
    best_inliers = []
    max_inliers = 0
    
    # Minimum number of points needed to estimate F
    min_points = 8
    
    if n_matches < min_points:
        print(f"Not enough matches for pair {pair}: {n_matches} < {min_points}")
        return None, []
    
    for _ in range(num_iterations):
        # Randomly select 8 matches
        sample_indices = random.sample(range(n_matches), min_points)
        # state = random.getstate()
        # print(f"Current RNG state: {state}")
        sample_points1 = points1[sample_indices]
        sample_points2 = points2[sample_indices]
        
        # Estimate fundamental matrix from sample
        F = estimate_fundamental_matrix(sample_points1, sample_points2)
        
        # Calculate distances
        distances = compute_epipolar_distance(points1, points2, F)
        
        # Find inliers
        inliers = np.where(distances < threshold)[0]
        n_inliers = len(inliers)
        
        # Update best model if we found more inliers
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            best_F = F
            best_inliers = inliers
    
    # Recompute F using all inliers for better accuracy
    if max_inliers >= min_points:
        inlier_points1 = points1[best_inliers]
        inlier_points2 = points2[best_inliers]
        best_F = estimate_fundamental_matrix(inlier_points1, inlier_points2)
        # best_F = best_F / best_F[2, 2]
    
    print(f"Pair {pair}: Found {max_inliers} inliers out of {n_matches} matches")
    return best_F, best_inliers, inlier_points1, inlier_points2
