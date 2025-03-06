import numpy as np
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from GetInliersRANSAC import *
np.random.seed(100)


def visualize_reprojection(image_path, projected_linear, projected_nonlinear, original_points, save_path="Data/IntermediateOutputImages/"):
    """
    Creates three separate visualizations:
    1. Original points + Linear triangulation
    2. Original points + Non-linear triangulation
    3. Linear + Non-linear triangulation
    
    Args:
        image_path (str): Path to the image.
        projected_linear (Nx2 np.array): Reprojected points from Linear Triangulation.
        projected_nonlinear (Nx2 np.array): Reprojected points from Non-Linear Triangulation.
        original_points (Nx2 np.array): Original detected feature points.
        save_path (str): Base path to save the output images.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"🚨 Error: Image {image_path} not found.")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Original (Green) + Linear (Blue)
    image_linear = image.copy()
    for pt in original_points.astype(int):
        cv2.circle(image_linear, tuple(pt), 3, (0, 255, 0), -1)  # Green for Original
    for pt in projected_linear.astype(int):
        cv2.circle(image_linear, tuple(pt), 3, (0, 0, 255), -1)  # Blue for Linear
    cv2.imwrite(f"{save_path}_linear.png", image_linear)
    print(f"Original + Linear reprojection saved at {save_path}_linear.png")
    
    # 2. Original (Green) + Non-Linear (Red)
    image_nonlinear = image.copy()
    for pt in original_points.astype(int):
        cv2.circle(image_nonlinear, tuple(pt), 3, (0, 255, 0), -1)  # Green for Original
    for pt in projected_nonlinear.astype(int):
        cv2.circle(image_nonlinear, tuple(pt), 3, (0, 0, 255), -1)  # Red for Non-Linear
    cv2.imwrite(f"{save_path}_nonlinear.png", image_nonlinear)
    print(f"Original + Non-Linear reprojection saved at {save_path}_nonlinear.png")
    
    # 3. Linear (Blue) + Non-Linear (Red)
    image_comparison = image.copy()
    for pt in projected_linear.astype(int):
        cv2.circle(image_comparison, tuple(pt), 3, (255, 0, 0), -1)  # Blue for Linear
    for pt in projected_nonlinear.astype(int):
        cv2.circle(image_comparison, tuple(pt), 3, (0, 0, 255), -1)  # Red for Non-Linear
    cv2.imwrite(f"{save_path}_comparison.png", image_comparison)
    print(f"Linear + Non-Linear comparison saved at {save_path}_comparison.png")

def project_points(P, X):
    """
    Projects 3D points onto a 2D image plane using the given projection matrix.

    Args:
        P (3x4 numpy array): Camera projection matrix.
        X (Nx3 numpy array): 3D world points.

    Returns:
        Projected 2D points (Nx2 numpy array).
    """
    X_homogeneous = np.hstack((X, np.ones((X.shape[0], 1))))  # Convert to homogeneous coordinates
    # print("X_homogeneous:\n", X_homogeneous)
    x_proj = (P @ X_homogeneous.T).T  # Project
    return (x_proj[:, :2].T / x_proj[:, 2]).T  # Normalize to get 2D coordinates



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

        # Convert points to integer tuples for cv2.circle
        pt1 = tuple(pt1.astype(int))
        pt2 = tuple(pt2.astype(int))

        # Draw circles at keypoints
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
    # Reshape keypoints for cv2.computeCorrespondEpilines
    keypoints1_reshaped = keypoints1.reshape(-1, 1, 2).astype(np.float32)
    keypoints2_reshaped = keypoints2.reshape(-1, 1, 2).astype(np.float32)

    # Compute epilines for both images
    epilines1 = cv2.computeCorrespondEpilines(keypoints1_reshaped, 1, F_matrix).reshape(-1, 3)
    epilines2 = cv2.computeCorrespondEpilines(keypoints2_reshaped, 2, F_matrix).reshape(-1, 3)

    # Select a subset of points for visualization (too many make the plot cluttered)
    num_points = min(20, len(keypoints1))
    indices = np.random.choice(len(keypoints1), num_points, replace=False)
    
    # Render epipolar lines
    img2_with_lines, img1_with_points = render_epipolar_lines(
        image2, image1, 
        epilines1[indices], 
        keypoints2_reshaped[indices, 0], 
        keypoints1_reshaped[indices, 0]
    )
    
    img1_with_lines, img2_with_points = render_epipolar_lines(
        image1, image2, 
        epilines2[indices], 
        keypoints1_reshaped[indices, 0], 
        keypoints2_reshaped[indices, 0]
    )

    # Create figure and plot the images
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img1_with_lines, cv2.COLOR_BGR2RGB))
    plt.title("Epipolar Lines on Image 1")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img2_with_lines, cv2.COLOR_BGR2RGB))
    plt.title("Epipolar Lines on Image 2")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Epipolar geometry image saved at: {save_path}")
    
    # Show the plot
    # plt.show()

def plot_epipolar_lines_for_pair(matches_dict, pair, image_folder="../P2Data/"):
    """
    Plot epipolar lines for a specific image pair using the matches dictionary.
    
    Args:
        matches_dict: Dictionary of matches from read_matching_files_modified
        pair: Tuple (i, j) specifying the image pair
        image_folder: Folder containing the images
    """
    # Extract image indices
    i, j = pair
    
   # To this (adjust extension as needed):
    img1 = cv2.imread(f"{image_folder}/{i}.png")  # or whatever the correct filename is
    img2 = cv2.imread(f"{image_folder}/{j}.png")


    if img1 is None or img2 is None:
        print(f"Error: Could not load images for pair {pair}")
        return None, None
    
    # Extract matching points
    matches = matches_dict[pair]
    
    # Format points for fundamental matrix estimation
    points1 = np.array([[match[0][1], match[0][2]] for match in matches])
    points2 = np.array([[match[1][1], match[1][2]] for match in matches])
    
    # Estimate fundamental matrix using RANSAC
    from GetInliersRANSAC import ransac_fundamental_matrix
    F, inliers = ransac_fundamental_matrix(matches_dict, pair)
    
    if F is None:
        print(f"Failed to estimate fundamental matrix for pair {pair}")
        return None, None
    
    # Display epipolar geometry
    display_epipolar_geometry(
        img1, img2, 
        points1[inliers], points2[inliers], 
        F, 
        title=f"Epipolar Geometry between Images {i} and {j}",
        save_path=f"Data/IntermediateOutputImages/epipolar_geometry_{i}_{j}.png"
    )
    
    return F, inliers

def plot_matches_ransac(img1, img2, points1, points2, inliers, save_path="Data/IntermediateOutputImages/ransac_matches.png"):
    """
    Visualize the results of RANSAC by showing inliers and outliers on the image pair.
    
    Args:
        img1: First image
        img2: Second image
        points1: Points in the first image (Nx2)
        points2: Points in the second image (Nx2)
        inliers: Indices of inlier matches
        save_path: Path to save the visualization
    """
    # Create a mask for inliers
    mask = np.zeros(len(points1), dtype=bool)
    mask[inliers] = True
    
    # Create a composite image by placing images side by side
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = max(h1, h2)
    width = w1 + w2
    composite = np.zeros((height, width, 3), dtype=np.uint8)
    composite[:h1, :w1] = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    composite[:h2, w1:w1+w2] = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # Draw matches
    for i in range(len(points1)):
        pt1 = (int(points1[i, 0]), int(points1[i, 1]))
        pt2 = (int(points2[i, 0] + w1), int(points2[i, 1]))
        
        # Green for inliers, red for outliers
        color = (0, 255, 0) if mask[i] else (0, 0, 255)
        thickness = 2 if mask[i] else 1
        
        # Draw lines between matching points
        cv2.line(composite, pt1, pt2, color, thickness)
        
        # Draw points
        cv2.circle(composite, pt1, 3, color, -1)
        cv2.circle(composite, pt2, 3, color, -1)
    
    # Add text showing number of inliers/total
    text = f"Inliers: {np.sum(mask)}/{len(mask)} ({np.sum(mask)/len(mask)*100:.1f}%)"
    cv2.putText(composite, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the visualization
    cv2.imwrite(save_path, composite)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    plt.title("RANSAC Feature Matching (Green: Inliers, Red: Outliers)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"RANSAC matches visualization saved at: {save_path}")

def visualize_all_ransac_matches(matches_dict, image_folder="../P2Data/"):
    """
    Visualize RANSAC matches for all image pairs in the matches dictionary.
    
    Args:
        matches_dict: Dictionary of matches from read_matching_files_modified
        image_folder: Folder containing the images
    """
    # Create output directory
    output_dir = "Data/IntermediateOutputImages/ransac_matches"
    # os.makedirs(output_dir, exist_ok=True)
    
    # Process each image pair
    for pair in matches_dict.keys():
        i, j = pair
        
        # Load images
        img1_path = f"{image_folder}/{i}.png"
        img2_path = f"{image_folder}/{j}.png"
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"Error: Could not load images for pair {pair}")
            continue
        
        # Extract matching points
        matches = matches_dict[pair]
        points1 = np.array([[match[0][1], match[0][2]] for match in matches])
        points2 = np.array([[match[1][1], match[1][2]] for match in matches])
        
        # Estimate fundamental matrix using RANSAC
        F, inliers, _, _ = ransac_fundamental_matrix(matches_dict, pair)
        
        if F is None or len(inliers) == 0:
            print(f"Failed to estimate fundamental matrix for pair {pair}")
            continue
        
        # Visualize RANSAC matches
        save_path = f"{output_dir}/ransac_matches_{i}_{j}.png"
        plot_matches_ransac(img1, img2, points1, points2, inliers, save_path)
        
        print(f"RANSAC matches for pair {pair}: {len(inliers)} inliers out of {len(matches)} matches")

        return points1, points2

def read_matching_files_new(folder_path):
    matches_dict = {}
    
    # Process matching1.txt file
    matching_file = os.path.join(folder_path, f"../P2Data/matching1.txt")
    
    with open(matching_file, 'r') as f:
        # First line contains number of features
        num_features = int(f.readline().split(':')[1])
        
        # Read each feature line
        for line in f:
            data = line.strip().split()
            if not data:
                continue
            
            # Number of matches for this feature
            num_matches = int(data[0])
            
            # RGB values (not used in this function)
            r, g, b = int(data[1]), int(data[2]), int(data[3])
            
            # Current image coordinates
            x_curr = float(data[4])
            y_curr = float(data[5])
            
            # Process each match to determine which images share this feature
            idx = 6  # Start index for matches data
            images_with_feature = [1]  # Image 1 has this feature
            match_coordinates = {1: (x_curr, y_curr)}  # Store coordinates for image 1
            
            for _ in range(num_matches):
                # Check if we have enough data for this match
                if idx + 2 >= len(data):
                    break
                    
                # Get matching image information
                j = int(data[idx])        # matching image ID
                x_match = float(data[idx + 1])  # x coordinate in image j
                y_match = float(data[idx + 2])  # y coordinate in image j
                
                images_with_feature.append(j)
                match_coordinates[j] = (x_match, y_match)
                
                idx += 3  # Move to next match
            
            # Sort the images to ensure consistent key ordering
            images_with_feature.sort()
            
            # Create all possible combinations of images that share this feature
            # Start with pairs and go up to all images that share the feature
            for size in range(2, len(images_with_feature) + 1):
                # Create a tuple key of the images that share this feature
                key = tuple(images_with_feature[:size])
                
                # Only process the specific keys we want to keep
                if key in [(1,2), (1,2,3), (1,2,3,4), (1,2,3,4,5)]:
                    # Initialize list for this key if needed
                    if key not in matches_dict:
                        matches_dict[key] = []
                    
                    # Create the match entry with coordinates for each image in the key
                    match_entry = [(img, match_coordinates[img][0], match_coordinates[img][1]) 
                                  for img in key]
                    
                    # Add the match only if this exact match is not already present
                    if match_entry not in matches_dict[key]:
                        matches_dict[key].append(match_entry)
    
    # Filter to keep only the specific keys
    allowed_keys = [(1,2), (1,2,3), (1,2,3,4), (1,2,3,4,5)]
    matches_dict = {k: v for k, v in matches_dict.items() if k in allowed_keys}
    
    return matches_dict




def read_files(folder_path):
    dicrionary_matches = {}
 
    # Process matching1.txt through matching4.txt
    for i in range(1, 5):
        matching_file = os.path.join(folder_path, f"../P2Data/matching{i}.txt")
        
        with open(matching_file, 'r') as file:
            #
            num_features = int(file.readline().split(':')[1])
            
            # Read each feature line
            for j in file:
                data = j.strip().split()
                if not data:
                    continue
                
                # Number of matches for this feature
                matches = int(data[0])
                
                # Current image (i) coordinates
                current_x = float(data[4])
                current_y = float(data[5])
                
                # Process each match
                index = 6  # Start index for matches data
                for _ in range(matches):
                    # Check if we have enough data for this match
                    if index + 2 >= len(data):
                        break
                        
                    # Get matching image information
                    j = int(data[index])        # matching image ID
                    matching_in_x = float(data[index + 1])  # x coordinate in image j
                    matching_in_y = float(data[index + 2])  # y coordinate in image j
                    
                    # Create key for image pair (smaller index first)
                    pair = (min(i, j), max(i, j))
                    
                    # Initialize list for this pair if needed
                    if pair not in dicrionary_matches:
                        dicrionary_matches[pair] = []
                    
                    # Create the match as a list of tuples (image_id, u, v)
                    if i < j:
                        match_entry = [(i, current_x, current_y), (j, matching_in_x, matching_in_y)]
                    else:
                        match_entry = [(j, matching_in_x, matching_in_y), (i, current_x, current_y)]
                    
                    # Add the match only if this exact match is not already present
                    if match_entry not in dicrionary_matches[pair]:
                        dicrionary_matches[pair].append(match_entry)
                    
                    index += 3  # Move to next match
 
    return dicrionary_matches




def plot_initial_triangulation_2D(X_candidates, save_path="Data/IntermediateOutputImages/2initial_triangulation_2D.png"):
    """Plots and saves the initial triangulated 2D XZ projection with disambiguity."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['blue', 'red', 'magenta', 'brown']  # Different colors for each candidate pose

    for i, X in enumerate(X_candidates):
        ax.scatter(X[:, 0], X[:, 2], c=colors[i], marker='.', s=2, label=f"Pose {i+1}")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Initial Triangulation (XZ Plane, Showing Disambiguity)")
    
    # Adjust these limits to match the expected plot
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)

    ax.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Initial triangulation (XZ plane) saved: {save_path}")

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
    # plt.show()

    print(f"3D Triangulation plot saved to: {save_path}")

def plot_2D_projections(X_initial, X_refined, save_path="2D_projection_comparison.png"):
    """Plots the 2D projections of initial and refined triangulated points with good contrast."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'initial': 'blue', 'refined': 'red'}  # Blue for Linear, Red for Non-Linear

    # Scatter plot for initial (linear) and refined (non-linear) triangulation
    ax.scatter(X_initial[:, 0], X_initial[:, 2], c=colors['initial'], marker='.', s=3, alpha=0.5, label="Linear (Initial)")
    ax.scatter(X_refined[:, 0], X_refined[:, 2], c=colors['refined'], marker='.', s=2, label="Non-Linear (Refined)")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("2D Projection in XZ Plane: Linear vs Non-Linear Triangulation")

    # Zoom to the center region for better visualization
    ax.set_xlim(-15, 15)  # Adjusted zoom
    ax.set_ylim(-5, 25)  # Adjusted zoom

    ax.legend()
    ax.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    print(f"XZ Plane Projection comparison saved to: {save_path}")

def plot_2D_projections_PNP(X_initial, X_refined, save_path="Data/IntermediateOutputImages/2D_projection_PNP.png"):
    """Plots the 2D projections of initial and refined triangulated points with good contrast."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'initial': 'blue', 'refined': 'red'}  # Blue for Linear, Red for Non-Linear

    # Scatter plot for initial (linear) and refined (non-linear) triangulation
    ax.scatter(X_initial[:, 0], X_initial[:, 2], c=colors['initial'], marker='.', s=3, alpha=0.5, label="Linear PNP(Initial)")
    ax.scatter(X_refined[:, 0], X_refined[:, 2], c=colors['refined'], marker='.', s=2, label="Non-Linear PNP (Refined)")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("2D Projection in XZ Plane: Linear PNP vs Non-Linear PNP Triangulation")

    # Zoom to the center region for better visualization
    ax.set_xlim(-15, 15)  # Adjusted zoom
    ax.set_ylim(-5, 25)  # Adjusted zoom

    ax.legend()
    ax.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

    print(f"XZ Plane Projection comparison saved to: {save_path}")

    
def plot_pnp_results(X_linear, X_nonlinear, image_ids, save_path="pnp_comparison.png"):
    """Plots the Linear PnP (red) vs Non-Linear PnP (blue) outputs for given images."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, img_idx in enumerate(image_ids):
        axes[i].scatter(X_linear[i][:, 0], X_linear[i][:, 2], c='r', s=2, label='linear')
        axes[i].scatter(X_nonlinear[i][:, 0], X_nonlinear[i][:, 2], c='b', s=2, label='nonlinear')
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Z")
        axes[i].set_title(f"Image (1, {img_idx})")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()
    print(f"PnP visualization saved at: {save_path}")

def save_feature_matches_on_image(image_path, x_image, img_idx, save_path="../P2Data/"):
    """
    Saves a visualization of 2D feature correspondences by overlaying them on the real image.

    Args:
        image_path: Path to the input image.
        x_image: (Nx2) 2D feature points.
        img_idx: Image index for naming.
        save_path: Directory to save the overlaid image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"🚨 Error: Image {image_path} not found.")
        return

    # Convert to RGB for plotting
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Overlay feature points
    for point in x_image:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image_rgb, (x, y), 4, (0, 255, 0), -1)  # Green dots for features

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the visualization
    file_path = os.path.join(save_path, f"feature_matches_img{img_idx}.png")
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title(f"Feature Matches Overlaid on Image {img_idx}")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Feature matches overlaid on image saved at: {file_path}")



def plot_3d_reconstruction_with_cameras(points, R_list, C_list, save_path="3d_reconstruction.png"):
    """Plot 3D points and camera positions with proper XZ projection."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', marker='.', s=2, alpha=0.5, label="3D Points")
    
    # Plot camera positions
    camera_positions = np.array([C.flatten() for C in C_list])
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
               c='red', marker='^', s=100, label="Cameras")
    
    # Add camera indices
    for i, (x, y, z) in enumerate(zip(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2])):
        ax.text(x, y, z, f"{i+1}", fontsize=12, color='black')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Reconstruction with Camera Positions")
    
    # Set equal aspect ratio
    max_range = np.max([
        np.max(points[:, 0]) - np.min(points[:, 0]),
        np.max(points[:, 1]) - np.min(points[:, 1]),
        np.max(points[:, 2]) - np.min(points[:, 2])
    ])
    mid_x = (np.max(points[:, 0]) + np.min(points[:, 0])) * 0.5
    mid_y = (np.max(points[:, 1]) + np.min(points[:, 1])) * 0.5
    mid_z = (np.max(points[:, 2]) + np.min(points[:, 2])) * 0.5
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    ax.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a 2D projection on XZ plane
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.scatter(points[:, 0], points[:, 2], c='blue', marker='.', s=2, alpha=0.5, label="3D Points (XZ projection)")
    ax2.scatter(camera_positions[:, 0], camera_positions[:, 2], c='red', marker='^', s=100, label="Cameras")
    
    # Add camera indices
    for i, (x, z) in enumerate(zip(camera_positions[:, 0], camera_positions[:, 2])):
        ax2.text(x, z, f"{i+1}", fontsize=12, color='black')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title("XZ Plane Projection of 3D Reconstruction")
    ax2.grid(True)
    ax2.legend()
    
    plt.savefig(save_path.replace('.png', '_xz_projection.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"3D reconstruction and XZ projection saved to: {save_path}")
