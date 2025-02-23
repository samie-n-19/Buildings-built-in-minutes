import numpy as np
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Import functions from other scripts
from EstimateFundamentalMatrix import estimate_fundamental_matrix
from EssentialMatrixFromFundamentalMatrix import essential_matrix_from_fundamental
from ExtractCameraPose import extract_camera_pose
from LinearTriangulation import linear_triangulation
from DisambiguateCameraPose import disambiguate_camera_pose
from NonLinearTriangulation import nonlinear_triangulation

def load_matches(filepath, image_index):
    """Loads matching feature points from a file."""
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
                image_id = int(parts[6 + j * 3])
                u_image_id = float(parts[7 + j * 3])
                v_image_id = float(parts[8 + j * 3])
                if image_id == (image_index + 1):
                    matches_data.append([ucurrent, vcurrent, u_image_id, v_image_id])
        except (IndexError, ValueError):
            continue

    return np.array(matches_data)


def get_inlier_ransac(points1, points2, num_iterations=1000, distance_threshold=1.0):
    """Estimates the fundamental matrix and inlier correspondences using RANSAC."""
    n = 0  # Best inlier count
    best_F = None
    best_inliers = []
    N = len(points1)

    if N < 8:
        print("Not enough matches to run RANSAC.")
        return None, []

    for i in range(num_iterations):
        try:
            sample_indices = np.random.choice(N, 8, replace=False)
            sample_points1 = points1[sample_indices]
            sample_points2 = points2[sample_indices]

            F = estimate_fundamental_matrix(sample_points1, sample_points2)

            S = []
            for j in range(N):
                point1_homo = np.append(points1[j], 1)
                point2_homo = np.append(points2[j], 1)

                algebraic_distance = np.abs(point2_homo @ F @ point1_homo)

                if algebraic_distance < distance_threshold:
                    S.append(j)

            if len(S) > n:
                n = len(S)
                best_inliers = S
                best_F = F

        except Exception as e:
            print(f"Error in RANSAC iteration {i}: {e}")
            continue

    return best_F, best_inliers

def plot_initial_triangulation(X, save_path="../P2Data/initial_triangulation.png"):
    """Visualizes and saves the initial triangulated 3D points."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='.', s=2, label="Initial Triangulated Points")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Initial Triangulation (Before Non-Linear Optimization)")
    plt.legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Initial triangulation plot saved: {save_path}")



def save_3D_plot(X, save_path="triangulation_output.png", title="3D Reconstructed Points"):
    """Saves the triangulated 3D points plot as an image file."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='.', s=2, label='Triangulated Points')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ 3D Triangulation plot saved to: {save_path}")


def main():
# """Main function to execute the SfM pipeline for the first image pair."""

    # Define dataset paths
    image_directory = "../P2Data"
    number_of_images = 5
    calibration_file = "../P2Data/calibration.txt"

    # Load Camera Calibration
    print(f"Looking for calibration file at: {calibration_file}")
    try:
        camera_matrix = np.loadtxt(calibration_file)
        camera_matrix = camera_matrix.reshape(3, 3)
        print("✅ Camera Matrix Loaded:\n", camera_matrix)
    except FileNotFoundError as e:
        print(f"🚨 Error: {e}\nPlease ensure that calibration.txt exists.")
        return

    # Process first image pair (ONLY image 1 and 2)
    image1_path = f"{image_directory}/1.png"
    image2_path = f"{image_directory}/2.png"
    matching_file_path = f"{image_directory}/matching1.txt"

    print(f"Processing {image1_path} and {image2_path}...")

    # Load matches
    matches = load_matches(matching_file_path, image_index=1)
    if len(matches) < 8:
        print("❌ Not enough matches for triangulation. Exiting.")
        return

    points1 = matches[:, :2]
    points2 = matches[:, 2:]

    # Estimate Fundamental Matrix using RANSAC
    F, inliers = get_inlier_ransac(points1, points2)
    points1, points2 = points1[inliers], points2[inliers]

    print(f"✅ Estimated Fundamental Matrix:\n{F}")

    # Compute Essential Matrix
    E = essential_matrix_from_fundamental(F, camera_matrix)
    print(f"✅ Estimated Essential Matrix:\n{E}")

    # Extract 4 Candidate Camera Poses
    candidates = extract_camera_pose(E)
    print("✅ Candidate Camera Poses Extracted.")

    # Define Projection Matrix for First Camera
    P1 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P1 = K[I | 0]

    # Select the Best Camera Pose Using Triangulation
    best_R, best_C, best_X = None, None, None
    max_valid_points = 0

    
    for i, (R, C) in enumerate(candidates):  # Check all 4 poses
        print(f"🔄 Testing Candidate Pose {i+1}:\nRotation:\n{R}\nTranslation:\n{C}")

        C = C.reshape((3, 1))  # Ensure column vector
        P2 = camera_matrix @ np.hstack((R, -R @ C))  # Compute projection matrix

        # Perform Linear Triangulation
        X_init = linear_triangulation(P1, P2, points1, points2)

        if np.isnan(X_init).any() or np.isinf(X_init).any():
            print(f"🚨 Pose {i+1} produced NaN/Inf values! Skipping this pose.")
            continue

        # Apply Cheirality Check (Z must be positive in both cameras)
        valid_points = np.sum(X_init[:, 2] > 0)
        print(f"🔹 Pose {i+1}: {valid_points} valid 3D points.")

        # Select the best pose with the highest valid points
        if valid_points > max_valid_points:
            max_valid_points = valid_points
            best_R, best_C, best_X = R, C, X_init

    # Ensure a valid pose was selected
    if best_R is None or best_C is None:
        print("🚨 No valid camera pose found! Exiting.")
        return
    
    
    # 🔹 Additional Check: If too many points have negative depth, reconsider pose
    num_negative_depth = np.sum(best_X[:, 2] < 0)
    if num_negative_depth > len(best_X) * 0.3:  # If more than 30% of points are behind the camera
        print(f"⚠️ Warning: Too many points have negative depth ({num_negative_depth}/{len(best_X)}). "
            f"Reevaluating pose selection...")

        # Try selecting the second-best pose instead
        sorted_poses = sorted(zip(valid_points, candidates), reverse=True)  # Sort by valid points count
        if len(sorted_poses) > 1:
            second_best_pose = sorted_poses[1]  # Pick the next best pose
            best_R, best_C = second_best_pose[1]

            print(f"🔄 Switching to second-best pose with {second_best_pose[0]} valid points.")

            # Recompute triangulation for the new pose
            best_C = best_C.reshape((3, 1))
            P2_best = camera_matrix @ np.hstack((best_R, -best_R @ best_C))
            best_X = linear_triangulation(P1, P2_best, points1, points2)



    print(f"✅ Best Camera Pose Selected with {max_valid_points} valid points.")
    plot_initial_triangulation(best_X)
    # Perform Non-Linear Triangulation
    if best_C[2] < 0:  
        print("🔄 Flipping translation vector direction to keep points in front.")
        best_C = -best_C

    P2_best = camera_matrix @ np.hstack((best_R, -best_R @ best_C))
    X_refined = nonlinear_triangulation(P1, P2_best, points1, points2, best_X)

    print(f"✅ Non-Linear Triangulation Output (First 5 Points):\n{X_refined[:5]}")

    # Save and visualize 3D reconstruction
    save_3D_plot(X_refined, save_path="output/3D_reconstruction_first_pair.png")

    print("✅ SfM pipeline for first image pair completed successfully!")

if __name__ == "__main__":
    main()
