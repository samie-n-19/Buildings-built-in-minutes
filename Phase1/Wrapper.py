import numpy as np
import cv2
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

def get_inliers_ransac(points1, points2, num_iterations=1000, distance_threshold=0.01):
    if len(points1) < 8 or len(points2) < 8:
        print(f"Not enough points for RANSAC. points1: {len(points1)}, points2: {len(points2)}")
        return None, []

    best_F = None
    best_inliers = []
    num_points = len(points1)

    for _ in range(num_iterations):
        try:
            sample_indices = np.random.choice(num_points, 8, replace=False)
            sample_points1 = points1[sample_indices]
            sample_points2 = points2[sample_indices]

            F = estimate_fundamental_matrix(sample_points1, sample_points2)
            errors = compute_epipolar_error(F, points1, points2)
            inliers = np.where(errors < distance_threshold)[0]

            if len(inliers) > len(best_inliers):
                best_F = F
                best_inliers = inliers
        except Exception as e:
            print(f"Error in RANSAC iteration: {e}")
            continue

    return best_F, best_inliers

def compute_epipolar_error(F, points1, points2):
    points1_homogeneous = np.column_stack((points1, np.ones(len(points1))))
    points2_homogeneous = np.column_stack((points2, np.ones(len(points2))))
    lines = (F @ points1_homogeneous.T).T
    errors = np.sum(points2_homogeneous * lines, axis=1)**2 / np.sum(lines[:, :2]**2, axis=1)
    return errors

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

        F, inliers = get_inliers_ransac(points1, points2)

        if F is None:
            print(f"Failed to estimate Fundamental Matrix for image pair {i} and {i+1}. Skipping.")
            continue

        print(f"Fundamental Matrix between {i}.png and {i+1}.png:")
        print(F)
        print(f"Number of inliers: {len(inliers)}")

        all_Fs.append(F)
        all_inliers.append(inliers)

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

