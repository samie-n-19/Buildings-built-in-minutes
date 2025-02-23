import numpy as np
import cv2
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen" 
# matplotlib.use('Agg') 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from EstimateFundamentalMatrix import estimate_fundamental_matrix
from EssentialMatrixFromFundamentalMatrix import essential_matrix_from_fundamental
from ExtractCameraPose import extract_camera_pose
from LinearTriangulation import linear_triangulation
from DisambiguateCameraPose import disambiguate_camera_pose
from NonLinearTriangulation import nonlinear_triangulation



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
    matching_files_dir = "../P2Data"

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

        E = essential_matrix_from_fundamental(F, camera_matrix)
        print("Essential Matrix:")
        print(E)


        candidates = extract_camera_pose(E)
        print("Candidate Camera Poses:")
        for idx, (R, t) in enumerate(candidates):
            print(f"\nCandidate {idx+1}:")
            print("Rotation Matrix:")
            print(R)
            print("Translation Vector (up to scale):")
            print(t)
            # Debugging print: Checking if we reach triangulation
            print("\n✅ Extracted all camera poses, proceeding to Triangulation...\n")


        # Step 4: Define Camera Projection Matrices
        P1 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P1 = K[I | 0]

        # Compute all triangulations for all four candidate poses
        X_all = []
        X_refined_all = []  # Store refined points for comparison
        # for R, C in candidates:
        #     print(f"Processing Pose with R:\n{R}\nC (Before Reshaping):\n{C}")
        #     C = C.reshape((3, 1))  # Ensure C is a column vector
        #     print(f"C (After Reshaping):\n{C}")  # Debugging step
        #     P2 = camera_matrix @ np.hstack((R, -R @ C.reshape((3, 1))))
            
        #     # Step 1: Compute initial 3D points using Linear Triangulation
        #     X_init = linear_triangulation(P1, P2, points1, points2)
        #     print(f"Linear Triangulation Output (First 5 Points):\n{X_init[:5]}")

        #     if np.isnan(X_init).any() or np.isinf(X_init).any():
        #         print("🚨 Error: X_init contains NaN or Inf values! Triangulation failed.")
        #         exit(1)
        #     if np.isnan(X_init).any() or np.isinf(X_init).any():
        #         print("🚨 Error: X_init contains NaN or Inf values! Skipping Non-Linear Triangulation.")
        #         continue
            
        #     # Step 2: Refine 3D points using Non-Linear Triangulation
        #     X_refined = nonlinear_triangulation(P1, P2, points1, points2, X_init)
        #     print(f"Non-Linear Triangulation Output (First 5 Points):\n{X_refined[:5]}")
            
        #     # Store results
        #     X_all.append(X_init)
        #     X_refined_all.append(X_refined)

        # Process only the first two camera poses
    best_R = None
    best_C = None
    best_X = None
    max_valid_points = 0

    for i, (R, C) in enumerate(candidates[:2]):  # Only use first 2 poses
        print(f"Processing Pose {i+1} with R:\n{R}\nC:\n{C}")

        C = C.reshape((3, 1))  # Ensure C is a column vector
        P2 = camera_matrix @ np.hstack((R, -R @ C))  # Compute projection matrix

        # Step 1: Compute initial 3D points using Linear Triangulation
        X_init = linear_triangulation(P1, P2, points1, points2)
        print(f"Linear Triangulation Output (First 5 Points):\n{X_init[:5]}")

        if np.isnan(X_init).any() or np.isinf(X_init).any():
            print("🚨 Error: X_init contains NaN or Inf values! Skipping.")
            continue  # Skip this pose if triangulation is invalid

        # Step 2: Count valid points with positive depth (Cheirality check)
        valid_points = np.sum(X_init[:, 2] > 0)

        print(f"Pose {i+1}: {valid_points} valid points with positive depth.")

        if valid_points > max_valid_points:
            max_valid_points = valid_points
            best_R, best_C, best_X = R, C, X_init  # Keep the best pose

    # Ensure a valid pose is selected
    if best_R is None or best_C is None:
        print("🚨 No valid camera pose found! Exiting.")
        exit(1)

    print(f"✅ Best Camera Pose Selected with {max_valid_points} valid points.")

    # Step 3: Perform Non-Linear Triangulation on the best pose
    if best_C[2] < 0:  
        print("🔄 Flipping translation vector direction to keep points in front.")
        best_C = -best_C

    P2_best = camera_matrix @ np.hstack((best_R, -best_R @ best_C))
    X_refined = nonlinear_triangulation(P1, P2_best, points1, points2, best_X)

    print(f"✅ Non-Linear Triangulation Output (First 5 Points):\n{X_refined[:5]}")
    # Store results
    X_all.append(X_init)
    X_refined_all.append(X_refined)

# Save and visualize refined 3D points
# plot_3D_reconstruction(X_refined_all, candidates, i)
    save_3D_plot(X_refined_all, candidates, i)

        
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
    print("✅ Triangulation completed successfully, proceeding to visualization...\n")

        

        

    return all_Fs, all_inliers





# def plot_3D_reconstruction(X_refined_all, candidates, img_pair_index):
#     """
#     Saves the 3D reconstructed points for each image pair using refined triangulation.

#     Args:
#         X_refined_all: List of triangulated 3D points for each candidate pose (Non-Linear).
#         candidates: List of four camera pose candidates (R, C).
#         img_pair_index: Index of the image pair being processed.
#     """
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')

#     colors = ['b', 'r', 'g', 'm']  # Different colors for each candidate pose

#     for i, X in enumerate(X_refined_all):
#         ax.scatter(X[:, 0], X[:, 2], X[:, 1], c=colors[i], marker='.', s=2, label=f'Pose {i+1}')

#         # Plot camera positions
#         R, C = candidates[i]
#         ax.scatter(C[0], C[2], C[1], c=colors[i], marker='x', s=100, label=f'Cam {i+1}')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Z')  # Swapping Y and Z for a top-down view
#     ax.set_zlabel('Y')
#     ax.set_title("Refined 3D Triangulation with Non-Linear Optimization")

#     ax.view_init(elev=90, azim=-90)  # Adjust perspective to match reference image
#     ax.legend()

#     save_path = f"../P2Data/reconstruction_refined_{img_pair_index}.png"
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save high-quality image
#     print(f"✅ Non-Linear 3D reconstruction saved to: {save_path}")

#     plt.close(fig)  # Close figure to avoid memory issues

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def save_3D_plot(X, save_path="triangulation_output.png", title="3D Reconstructed Points"):
    """
    Saves the triangulated 3D points plot as an image file.

    Args:
        X: (Nx3) numpy array of triangulated 3D points.
        save_path: (str) Path to save the plot.
        title: (str) Title of the plot.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='.', s=2, label='Triangulated Points')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.legend()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to avoid memory issues

    print(f"✅ 3D Triangulation plot saved to: {save_path}")

# ✅ Call the function after Non-Linear Triangulation
save_path = "output/3D_reconstruction_first_two_images.png"



    


if __name__ == "__main__":
    image_directory = "../P2Data"
    number_of_images = 5
    
    calibration_file = "../P2Data/calibration.txt"

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
