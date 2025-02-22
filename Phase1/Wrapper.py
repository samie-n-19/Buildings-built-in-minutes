import numpy as np
import cv2
import os
from EstimateFundamentalMatrix import estimate_fundamental_matrix

def load_matches(filepath, image_index):
    matches_data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    n_features = int(lines[0].split(':')[1].strip())
    print(f"Number of features: {n_features}")

    for i in range(1, len(lines)):
        parts = lines[i].split()
        try:
            num_matches = int(parts[0])
            ucurrent = float(parts[4])
            vcurrent = float(parts[5])

            for j in range(num_matches):
                image_id = int(parts[6 + j*3])
                u_image_id = float(parts[7 + j*3])
                v_image_id = float(parts[8 + j*3])
                if image_id == (image_index + 1):
                    matches_data.append([ucurrent, vcurrent, u_image_id, v_image_id])
        except (IndexError, ValueError) as e:
            continue

    return np.array(matches_data)

def get_inlier_ransac(points1, points2, num_iterations=1000, distance_threshold=1.0):
    """
    Estimates the fundamental matrix and inlier correspondences using RANSAC.

    Args:
        points1: NumPy array of shape (N, 2) representing points in image 1.
        points2: NumPy array of shape (N, 2) representing corresponding points in image 2.
        num_iterations: Number of RANSAC iterations (M in the pseudo-code).
        distance_threshold: Threshold for the epipolar constraint (epsilon in the pseudo-code).

    Returns:
        A tuple containing:
            - best_F: The best estimated fundamental matrix.
            - best_inliers: A list of indices of the inlier matches.
    """

    n = 0  # Best inlier count
    best_F = None
    best_inliers = []
    N = len(points1)  # Total number of matches
    if N < 8:
        print("Not enough matches to run RANSAC.")
        return None, []

    for i in range(num_iterations):  # Outer loop (M iterations)
        try:
            # 1. Random Sampling: Choose 8 correspondences randomly
            sample_indices = np.random.choice(N, 8, replace=False)
            sample_points1 = points1[sample_indices]
            sample_points2 = points2[sample_indices]

            # 2. Estimate Fundamental Matrix (using your existing function)
            F = estimate_fundamental_matrix(sample_points1, sample_points2)

            # 3. Inner Loop (Consensus Set): Check all N matches
            S = []  # Support set (indices of inliers)
            for j in range(N):
                # Epipolar constraint check
                point1_homo = np.append(points1[j], 1)  # Convert to homogeneous coordinates
                point2_homo = np.append(points2[j], 1)

                algebraic_distance = np.abs(point2_homo @ F @ point1_homo)  # Algebraic distance

                # Inlier test
                if algebraic_distance < distance_threshold:
                    S.append(j)

            # 4. Update Best Inlier Set
            if len(S) > n:
                n = len(S)
                best_inliers = S
                best_F = F

        except Exception as e:
            print(f"Error in RANSAC iteration {i}: {e}")
            continue  # Handle potential errors gracefully

    return best_F, best_inliers

def process_images(image_dir, num_images, camera_matrix, matching_files_prefix="matching"):
    all_Fs = []
    all_inliers = []
    matching_files_dir = "/home/samruddhi/Documents/CV Projects/CV_p2/YourDirectoryID_p2/P2Data"

    for i in range(1, num_images):
        image1_path = f"{image_dir}/{i}.png"
        image2_path = f"{image_dir}/{i+1}.png"
        matching_file_path = f"{matching_files_dir}/{matching_files_prefix}{i}.txt"

        print(f"Processing {image1_path} and {image2_path}")
        print(f"Using matches from {matching_file_path}")

        matches = load_matches(matching_file_path, i)

        if len(matches) < 8:
            print(f"Not enough matches for image pair {i} and {i+1}. Skipping.")
            continue

        points1 = matches[:, :2]
        points2 = matches[:, 2:]

        F, inliers = get_inlier_ransac(points1, points2)

        if F is None:
            print(f"Failed to estimate Fundamental Matrix for image pair {i} and {i+1}. Skipping.")
            inliers = []
        else:
            print(f"Fundamental Matrix between {i}.png and {i+1}.png:")
            print(F)
            print(f"Number of inliers: {len(inliers)}")

        all_Fs.append(F)
        all_inliers.append(inliers)

        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        # Extract inlier points
        inlier_points1 = points1[inliers]
        inlier_points2 = points2[inliers]

        # Convert points to KeyPoint objects
        keypoints1_in = [cv2.KeyPoint(x, y, 1) for x, y in inlier_points1]
        keypoints2_in = [cv2.KeyPoint(x, y, 1) for x, y in inlier_points2]

        # Create matches list
        matches_in = [cv2.DMatch(i, i, 0) for i in range(len(keypoints1_in))]

        # Create a blank output image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        height = max(h1, h2)
        width = w1 + w2
        output_img = np.zeros((height, width, 3), dtype=img1.dtype)
        output_img[:h1, :w1, :] = img1
        output_img[:h2, w1:w1 + w2, :] = img2

        # Offset keypoints in the second image
        keypoints2_in_offset = [cv2.KeyPoint(kp.pt[0] + w1, kp.pt[1], kp.size) for kp in keypoints2_in]

        # Draw inlier matches in green
        output_img = cv2.drawMatches(
            img1, keypoints1_in, img2, keypoints2_in_offset,
            matches_in, output_img,
            matchColor=(0, 255, 0),  # Green for inliers
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        cv2.imwrite(f"{matching_files_dir}/matches_{i}.png", output_img)
        print(f"Saved visualization to: {matching_files_dir}/matches_{i}.png")

    return all_Fs, all_inliers

if __name__ == "__main__":
    image_directory = "/home/samruddhi/Documents/CV Projects/CV_p2/YourDirectoryID_p2/P2Data"
    number_of_images = 5
    
    calibration_file = "/home/samruddhi/Documents/CV Projects/CV_p2/YourDirectoryID_p2/P2Data/calibration.txt"

    print(f"Looking for calibration file at: {calibration_file}")
    
    try:
        camera_matrix = np.loadtxt(calibration_file)
        camera_matrix = camera_matrix.reshape(3, 3)
        print("Camera Matrix:\n", camera_matrix)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure that calibration.txt exists at the specified path.")
        exit(1)

    # Process images and matches
    fundamental_matrices, inlier_indices = process_images(image_directory, number_of_images, camera_matrix)

    for i, (F, inliers) in enumerate(zip(fundamental_matrices, inlier_indices)):
        print(f"\nResults for image pair {i+1} and {i+2}:")
        print(f"Fundamental Matrix:\n{F}")
        print(f"Number of inliers: {len(inliers)}")
