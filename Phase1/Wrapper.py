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


def render_epipolar_lines(image1, image2, epilines, keypoints1, keypoints2):
    """
    Render epipolar lines and corresponding points on two images.

    Args:
        image1: First image to overlay epipolar lines.
        image2: Second image to overlay points.
        epilines: Epipolar lines in the second image.
        keypoints1: Key points in the first image.
        keypoints2: Corresponding key points in the second image.

    Returns:
        modified_img1: Image with epipolar lines.
        modified_img2: Image with key points.
    """
    img1_mod = image1.copy()
    img2_mod = image2.copy()
    rows, cols = img1_mod.shape[:2]

    # Define a fixed set of colors
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 165, 0)]

    for i, (line, pt1, pt2) in enumerate(zip(epilines, keypoints1, keypoints2)):
        color = colors[i % len(colors)]  # Cycle through color set
        x0, y0 = int(0), int(-line[2] / line[1])
        x1, y1 = int(cols), int(-(line[2] + line[0] * cols) / line[1])

        img1_mod = cv2.line(img1_mod, (x0, y0), (x1, y1), color, 1)

        # ✅ Fix: Convert (1, 2) shape arrays to (x, y) tuples
        pt1 = tuple(pt1.ravel().astype(int))  # Flatten and convert to tuple
        pt2 = tuple(pt2.ravel().astype(int))

        # Now safely call cv2.circle
        img1_mod = cv2.circle(img1_mod, pt1, 4, color, -1)
        img2_mod = cv2.circle(img2_mod, pt2, 4, color, -1)

    return img1_mod, img2_mod




def display_epipolar_geometry(image1, image2, keypoints1, keypoints2, F_matrix, title="Epipolar Geometry", save_path="output/epipolar_geometry.png"):
    """
    Display and save epipolar geometry by plotting lines on both images.

    Args:
        image1, image2: Input images.
        keypoints1, keypoints2: Corresponding points in both images.
        F_matrix: Computed fundamental matrix.
        title: Title of the plot.
        save_path: File path to save the plotted image.
    """
    keypoints1 = keypoints1.reshape(-1, 1, 2).astype(np.float32)
    keypoints2 = keypoints2.reshape(-1, 1, 2).astype(np.float32)

    # Compute epilines for both images
    epilines1 = cv2.computeCorrespondEpilines(keypoints1, 1, F_matrix).reshape(-1, 3)
    epilines2 = cv2.computeCorrespondEpilines(keypoints2, 2, F_matrix).reshape(-1, 3)

    img2_with_lines, img1_with_points = render_epipolar_lines(image2, image1, epilines1, keypoints2, keypoints1)
    img1_with_lines, img2_with_points = render_epipolar_lines(image1, image2, epilines2, keypoints1, keypoints2)

    # Create figure and plot the images
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img1_with_lines)
    plt.title("Epipolar Lines on Image 1")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(img2_with_lines)
    plt.title("Epipolar Lines on Image 2")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Epipolar geometry image saved at: {save_path}")

    # Show the plot
    plt.show()

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


def get_inlier_ransac(points1, points2, num_iterations=1000, distance_threshold=0.005):#1.0):
    """Estimates the fundamental matrix and inlier correspondences using RANSAC."""
    n = 0  # Best inlier count
    best_F = None
    best_inliers = []
    N = len(points1)
    np.random.seed(100)

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

                algebraic_distance = np.abs(point2_homo @ F @ point1_homo.T)

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


def plot_initial_triangulation_2D(X_candidates, save_path="../P2Data/initial_triangulation_2D.png"):
    """Plots and saves the initial triangulated 2D XZ projection with disambiguity (all four candidate poses)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['blue', 'red', 'magenta', 'brown']  # Different colors for each candidate pose

    for i, X in enumerate(X_candidates):
        ax.scatter(X[:, 0], X[:, 2], c=colors[i], marker='.', s=2, label=f"Pose {i+1}")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Initial Triangulation (XZ Plane, Showing Disambiguity)")
    
    # ✅ Set better zoomed-in axis limits
    ax.set_xlim(-20, 20)
    ax.set_ylim(-30, 30)

    ax.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Initial triangulation (XZ plane) saved: {save_path}")




# def save_3D_plot(X, save_path="triangulation_output.png", title="3D Reconstructed Points"):
#     """Saves the triangulated 3D points plot as an image file."""
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')

#     ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='.', s=2, label='Triangulated Points')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title(title)

#     plt.legend()

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)

#     print(f"✅ 3D Triangulation plot saved to: {save_path}")

def save_3D_plot(X, camera_poses=[], save_path="triangulation_output.png", title="3D Reconstructed Points"):
    """Saves the triangulated 3D points plot as an image file with camera frustums."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the triangulated points
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='.', s=2, label='Triangulated Points')

    # Plot camera frustums
    for i, (R, C) in enumerate(camera_poses):
        C = C.flatten()
        ax.scatter(C[0], C[1], C[2], c='r', marker='o', s=50, label=f"Camera {i+1}" if i == 0 else "")
        ax.quiver(C[0], C[1], C[2], R[0, 2], R[1, 2], R[2, 2], length=0.5, color='r', label="Camera Direction" if i == 0 else "")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ 3D Triangulation plot saved to: {save_path}")


# def plot_2D_structure(X, plane="XY", save_path="../P2Data/2D_projection.png"):
#     """Plots the 2D projection of the 3D reconstructed points."""
#     fig, ax = plt.subplots(figsize=(8, 6))

#     if plane == "XY":
#         ax.scatter(X[:, 0], X[:, 1], c='b', marker='.', s=2)
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#     elif plane == "XZ":
#         ax.scatter(X[:, 0], X[:, 2], c='g', marker='.', s=2)
#         ax.set_xlabel("X")
#         ax.set_ylabel("Z")
#     elif plane == "YZ":
#         ax.scatter(X[:, 1], X[:, 2], c='r', marker='.', s=2)
#         ax.set_xlabel("Y")
#         ax.set_ylabel("Z")
#     else:
#         print("Invalid plane selection! Choose from XY, XZ, or YZ.")
#         return

#     ax.set_title(f"2D Projection on {plane} Plane")
#     plt.grid(True)

#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     # plt.show()

#     print(f"✅ 2D Projection plot saved to: {save_path}")

def plot_2D_projections(X_initial, X_refined, save_path="2D_projection_comparison.png"):
    """Plots the 2D projections of initial and refined triangulated points with good contrast."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'initial': 'blue', 'refined': 'red'}  # Blue for Linear, Red for Non-Linear

    # Scatter plot for initial (linear) and refined (non-linear) triangulation
    ax.scatter(X_initial[:, 0], X_initial[:, 2], c=colors['initial'], marker='.', s=2, alpha=0.5, label="Linear (Initial)")
    ax.scatter(X_refined[:, 0], X_refined[:, 2], c=colors['refined'], marker='.', s=2, label="Non-Linear (Refined)")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("2D Projection in XZ Plane: Linear vs Non-Linear Triangulation")

    # ✅ Zoom to the center region for better visualization
    ax.set_xlim(-15, 15)  # Adjusted zoom
    ax.set_ylim(-20, 20)  # Adjusted zoom

    ax.legend()
    ax.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ XZ Plane Projection comparison saved to: {save_path}")

    
def main():
    """Main function to execute the SfM pipeline for the first image pair."""

    # Define dataset paths
    image_directory = "../P2Data"
    calibration_file = "../P2Data/calibration.txt"

    # Load Camera Calibration
    print(f"Looking for calibration file at: {calibration_file}")
    try:
        camera_matrix = np.loadtxt(calibration_file).reshape(3, 3)
        print("✅ Camera Matrix Loaded:\n", camera_matrix)
    except FileNotFoundError:
        print("🚨 Error: Calibration file not found! Ensure calibration.txt exists.")
        return

    # Load first image pair (ONLY image 1 and 2)
    image1_path = f"{image_directory}/1.png"
    image2_path = f"{image_directory}/2.png"
    matching_file_path = f"{image_directory}/matching1.txt"

    print(f"Processing images: {image1_path} and {image2_path}...")

    # Load matches
    matches = load_matches(matching_file_path, image_index=1)
    if len(matches) < 8:
        print("❌ Not enough matches for triangulation. Exiting.")
        return

    points1, points2 = matches[:, :2], matches[:, 2:]

    # Estimate Fundamental Matrix using RANSAC
    F, inliers = get_inlier_ransac(points1, points2)
    points1, points2 = points1[inliers], points2[inliers]

    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # ✅ Display and save epipolar geometry
    epipolar_image_path = "../P2Data/epipolar_geometry.png"
    display_epipolar_geometry(img1, img2, points1, points2, F, save_path=epipolar_image_path)
    print(f"✅ Epipolar geometry image saved at: {epipolar_image_path}")

    # Compute Essential Matrix
    E = essential_matrix_from_fundamental(F, camera_matrix)
    print(f"✅ Estimated Essential Matrix:\n{E}")

    # Extract 4 Candidate Camera Poses
    candidates = extract_camera_pose(E)
    print("✅ Candidate Camera Poses Extracted.")

    X_candidates = []
    camera_poses = []
    P1 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P1 = K[I | 0]

    # Compute triangulation for all 4 candidate poses
    for i, (R, C) in enumerate(candidates):
        print(f"🔄 Testing Candidate Pose {i+1}")
        C = C.reshape((3, 1))
        P2 = camera_matrix @ np.hstack((R, -R @ C))  # Compute projection matrix

        # Perform Linear Triangulation for this pose
        X_init = linear_triangulation(P1, P2, points1, points2)
        X_candidates.append(X_init)
        camera_poses.append((R, C.flatten()))

    # ✅ Plot disambiguity (2D XZ plane visualization)
    plot_initial_triangulation_2D(X_candidates, save_path="../P2Data/initial_triangulation_2D.png")

    # Define Projection Matrix for First Camera
    P1 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P1 = K[I | 0]

    # Select the Best Camera Pose Using Triangulation
    best_R, best_C, best_X = None, None, None
    max_valid_points = 0
    camera_poses = []

    for i, (R, C) in enumerate(candidates):
        print(f"🔄 Testing Candidate Pose {i+1}:\nRotation:\n{R}\nTranslation:\n{C}")

        C = C.reshape((3, 1))
        P2 = camera_matrix @ np.hstack((R, -R @ C))  # Compute projection matrix

        # Perform Linear Triangulation
        X_init = linear_triangulation(P1, P2, points1, points2)

        if np.isnan(X_init).any() or np.isinf(X_init).any():
            print(f"🚨 Pose {i+1} produced NaN/Inf values! Skipping this pose.")
            continue

        # Apply Cheirality Check (Z must be positive in both cameras)
        valid_points = np.sum(X_init[:, 2] > 0)
        print(f"🔹 Pose {i+1}: {valid_points} valid 3D points.")

        # Store camera pose for visualization
        camera_poses.append((R, C.flatten()))

        # Select the best pose
        if valid_points > max_valid_points:
            max_valid_points = valid_points
            best_R, best_C, best_X = R, C, X_init

    # Ensure a valid pose was selected
    if best_R is None or best_C is None:
        print("🚨 No valid camera pose found! Exiting.")
        return

    print(f"✅ Best Camera Pose Selected with {max_valid_points} valid points.")

    # ✅ Perform Non-Linear Triangulation
    if best_C[2] < 0:
        print("🔄 Flipping translation vector to keep points in front.")
        best_C = -best_C

    P2_best = camera_matrix @ np.hstack((best_R, -best_R @ best_C))
    X_refined = nonlinear_triangulation(P1, P2_best, points1, points2, best_X)

    print(f"✅ Non-Linear Triangulation Output (First 5 Points):\n{X_refined[:5]}")

    # ✅ Compare 2D projections before and after refinement
    projection_comparison_path = "../P2Data/2D_projection_comparison.png"
    plot_2D_projections(best_X, X_refined, save_path=projection_comparison_path)
    print(f"✅ 2D Projection comparison saved: {projection_comparison_path}")

    print("✅ SfM pipeline for first image pair completed successfully!")

if __name__ == "__main__":
    main()
