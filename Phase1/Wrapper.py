import numpy as np
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Import functions from other scripts
from EstimateFundamentalMatrix import estimate_fundamental_matrix
from EssentialMatrixFromFundamentalMatrix import essential_matrix_from_fundamental
from ExtractCameraPose import extract_camera_pose
from LinearTriangulation import *
from DisambiguateCameraPose import disambiguate_camera_pose
from NonLinearTriangulation import nonlinear_triangulation
from PnPRANSAC import *
from NonLinearPnP import *
from GetInliersRANSAC import *
from LinearPnP import *
from utils import *
from BuildVisibilityMatrix import *
from BundleAdjustment import *
np.random.seed(100)




def main():
    """Main function to execute the SfM pipeline for the first image pair."""
    os.makedirs("Data/IntermediateOutputImages", exist_ok=True)

    # Example Usage
    directory_path = "../P2Data/"  # Change this to the actual path
    dictionary_matches = read_files(directory_path)
    print(f"Processed {len(dictionary_matches)} matching files successfully.")
    # Print the keys of the matches_dict
    print("Keys in matches_dict:")
    for key in dictionary_matches.keys():
        print(key)

    visualize_all_ransac_matches(dictionary_matches, directory_path)


    image1_path = "/home/sarvesh/Documents/CV_2025/p2trial/P2Data/1.png"
    image2_path = "/home/sarvesh/Documents/CV_2025/p2trial/P2Data/2.png"
    
    # Load the images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

   

    
    if img1 is None or img2 is None:
        print(f"Error: Could not load images at {image1_path} and {image2_path}")
        return
    F_matrix = []
    inliers_points1 = []
    inliers_points2 = []
    for pair in dictionary_matches.keys():
        F, inliers, i1, i2 = ransac_fundamental_matrix(dictionary_matches, pair)
        F_matrix.append(F)
        inliers_points1.append(i1)
        inliers_points2.append(i2)
    
    print("First matrix ",F_matrix[1]) 


    # os.makedirs("output", exist_ok=True)
    
   



        
    # Define dataset paths
    image_directory = "../P2Data"
    calibration_file = "../P2Data/calibration.txt"

    # Load Camera Calibration
    print(f"Looking for calibration file at: {calibration_file}")
    try:
        camera_matrix = np.loadtxt(calibration_file).reshape(3, 3)
        print("Camera Matrix Loaded:\n", camera_matrix)
    except FileNotFoundError:
        print(" Error: Calibration file not found! Ensure calibration.txt exists.")
        return

    
    
    # Compute Essential Matrix
    E = essential_matrix_from_fundamental(F_matrix[1], camera_matrix)
    # Debug: Print Essential Matrix
    print(f" Essential Matrix for pair (1, 2):\n{E}")

    # Debug: Check Singular Values of E
    U, S, Vt = np.linalg.svd(E)
    print(f" Singular Values of E: {S} (Expected: [s, s, 0])")

    # Ensure E has correct singular values
    if abs(S[2]) > 1e-6:
        print(" Error: Essential Matrix does not have the expected singular values. There might be an issue in computation.")
        return  # Stop execution

    print("Essential Matrix is valid.")

    # Extract 4 Candidate Camera Poses
    candidates = extract_camera_pose(E)

    # Debug: Print Camera Poses
    print(" Extracted Camera Poses:")
    for i, (R, C) in enumerate(candidates):
        print(f" Pose {i+1}:")
        print(f"Rotation Matrix (R):\n{R}")
        print(f"Translation Vector (C):\n{C.T}")
        print(f"Det(R): {np.linalg.det(R)} (Expected: +1)")

    # Validate Candidate Poses
    if not candidates or len(candidates) != 4:
        print(" Error: No valid camera poses found. Ensure Essential Matrix is correct.")
        return  # Stop execution


    print("Candidate Camera Poses Extracted.")


    
    points1 = inliers_points1[1]
    points2 = inliers_points2[1]

    # Load first image pair (ONLY image 1 and 2)
    image1_path = f"{image_directory}/1.png"
    image2_path = f"{image_directory}/2.png"
    matching_file_path = f"{image_directory}/matching1.txt"

    print(f"Processing images: {image1_path} and {image2_path}...")


    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    epipolar_image_path = "Data/IntermediateOutputImages/epipolar_geometry.png"
    display_epipolar_geometry(img1, img2, points1, points2, F, save_path=epipolar_image_path)
    print(f"✅ Epipolar geometry image saved at: {epipolar_image_path}")

    
    X_candidates = []
    camera_poses = []
    P1 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P1 = K[I | 0]
    R1 = np.eye(3)
    C1 = np.zeros((3, 1))
    # Compute triangulation for all 4 candidate poses
    for i, (R2, C2) in enumerate(candidates):
        print(f" Testing Candidate Pose {i+1}")
        C2 = C2.reshape((3, 1))
        # P2 = camera_matrix @ np.hstack((R, -R @ C2))  # Compute projection matrix

        # Perform Linear Triangulation for this pose

        X_init = linear_triangulation(camera_matrix,R1,C1,R2,C2,points1,points2)

        # Validate triangulation results
        if X_init is None or X_init.shape[0] == 0:
            print(f"⚠️ Warning: Candidate Pose {i+1} failed triangulation. Skipping...")
            continue  # Skip this pose instead of exiting

        print(f"Candidate Pose {i+1} produced {X_init.shape[0]} valid 3D points.")

        X_candidates.append(X_init)
        camera_poses.append((R2, C2))#.flatten()))

    # Ensure at least one valid triangulation exists
    if not X_candidates:
        print(" Error: All candidate poses failed triangulation. Exiting.")
        return  # Stop execution
 

    # Plot disambiguity (2D XZ plane visualization)
    plot_initial_triangulation_2D(X_candidates, save_path="Data/IntermediateOutputImages/initial_triangulation_2D.png")

    # Define Projection Matrix for First Camera
    P1 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))  # P1 = K[I | 0]

    best_pose, best_X = disambiguate_camera_pose(camera_poses, X_candidates)
    print("Lenght of best_X",len(best_X))
    print("Length of point1 ",len(points1))
    print("Length of point2 ",len(points2))
    # Validate Pose Selection
    if best_pose is None or best_X is None or best_X.shape[0] == 0:
        print(" Error: No valid camera pose found after disambiguation. Exiting.")
        return  # Stop execution

    best_R, best_C = best_pose
    print("Best Camera Pose Selected Successfully.")

 
    P2_best = camera_matrix @ np.hstack((best_R, -best_R @ best_C))
    print(f"Best Projection Matrix (P2) Selected:\n{P2_best}")
    best_pose_index = 0
    for i,p in enumerate(camera_poses):
        if np.array_equal(p,best_pose):
            best_pose_index = i
            break
    X_ordered = X_candidates[best_pose_index]
    # print("Length of X_ordered.................................",len(X_ordered)) 
    # print(X_ordered)
    # print("Length of best_X......................................",len(best_X))
    # print(best_X)

    # Find the indices in X_ordered that correspond to points in best_X
    indices = []
    for point in best_X:
        # Use a more robust matching approach with a higher tolerance
        distances = np.sum((X_ordered - point)**2, axis=1)
        closest_idx = np.argmin(distances)
        
        # Only add if the distance is below a threshold
        if distances[closest_idx] < 1e-4:  # Adjust threshold as needed
            indices.append(closest_idx)

    # Now extract the corresponding points from points1 and points2
    points1 = points1[indices]
    points2 = points2[indices]

    # print(f"Original points: {len(points1)}, Subset points: {len(points1_subset)}")
    # print("this is indices",indices)

    X_refined = None

   
    # Perform Non-Linear Triangulation
    X_init = best_X.reshape(-1, 3)  


    X_refined = nonlinear_triangulation(
        P1, P2_best, 
        points1, points2, 
        X_init
    )
    print("lenght of x_refined",len(X_refined))
    print("length of X_init",len(X_init))

   

    # Validate Non-Linear Triangulation
    if X_refined is None or X_refined.shape[0] == 0:
        print(" Error: Non-Linear Triangulation failed. Exiting.")
        return  # Stop execution

    print(f"Non-Linear Triangulation successful. {X_refined.shape[0]} 3D points refined.")



 

    print(f"Number of triangulated 3D points (Non-Linear): {X_refined.shape[0]}")


    # Compute the best projection matrix for the selected camera pose
    P_best = camera_matrix @ np.hstack((best_R, -best_R @ best_C))

    # Project triangulated 3D points back to the image plane
    projected_linear = project_points(P_best, best_X)  # Linear Triangulation Projection
    projected_nonlinear = project_points(P_best, X_refined)  # Non-Linear Triangulation Projection
    print(f"Length of best_X: {len(best_X)}")
    print(f"Length of projected_linear: {len(projected_linear)}")
    print(f"Length of projected_nonlinear: {len(projected_nonlinear)}")
    # Define image path used for triangulation
    image_path = "../P2Data/2.png"  # Adjust path if needed

    # Call visualization function to overlay projections on the image
    visualize_reprojection(image_path, projected_linear, projected_nonlinear, points2)

    print(f"Non-Linear Triangulation Output (First 5 Points):\n{X_refined[:5]}")

    # Compare 2D projections before and after refinement for linear and non linear triangulation 
    projection_comparison_path = "Data/IntermediateOutputImages/2D_projection_comparison.png"
    plot_2D_projections(best_X, X_refined, save_path=projection_comparison_path)
    print(f"2D Projection comparison saved: {projection_comparison_path}")


    print("SfM pipeline for first image pair completed successfully!")
    print(f"Length of points1: {len(points1)}")
    print(f"Length of X_refined: {len(X_refined)}")


    camera_poses = [(R1, C1), (best_R, best_C)]  # First two cameras

    # Ensure points1 and X_refined have the same length
    min_length = min(len(points1), len(X_refined))
    points1_trimmed = points1[:min_length]
    X_refined_trimmed = X_refined[:min_length]

    print(f" Trimming 2D points to match 3D points: {len(points1)} → {min_length}")
     ################################################################################################################################################

    #                               PNP
    # Create a dictionary to store 3D points with their corresponding 2D points as keys

    directory_path = "../P2Data/"  # Change this to the actual path
    dictionary_matches = read_matching_files_new(directory_path)
    # Print the keys and their respective lengths
    # for key, value in matches_dict.items():
        # print(f"Key: {key}, Length: {len(value)}, Value: {value}")



    # After your successful triangulation code in main()

    # Setup for PnP
    print("---------------------------------------Starting PnP----------------------------------------------------")

    # Store all camera poses and 3D points
    all_camera_poses = []
    all_camera_poses.append((R1, C1))  # First camera (reference camera)
    all_camera_poses.append((best_R, best_C))  # Second camera from triangulation

    # Store 3D points with their corresponding 2D observations
    # Create a dictionary mapping 3D point indices to their coordinates
    world_points_dict = {}
    for i, point in enumerate(X_refined):
        world_points_dict[i] = point
    
    coresponding_points_dict = {}
    for i, point in enumerate(points1):
        coresponding_points_dict[i] = point

    for img_idx in range(3, 6):  # Images 3, 4, 5
        print(f"...................................Processing Image {img_idx}...........................................")
        
        # Find the key that contains both image 1 and current image
        relevant_key = None
        for key in dictionary_matches.keys():
            if 1 in key and img_idx in key:
                relevant_key = key
                break

        print(f"Processing key: {relevant_key}")
    
        if relevant_key is None:
            print(f"No matching data found between image 1 and image {img_idx}")
            continue
        
        # Extract 2D-3D correspondences
        points_3d = []
        points_2d = []
        # cors_points = []
        
        # Get matches between image 1 and current image
        matches = dictionary_matches[relevant_key]
        # print("THIS IS THE @1........................................................................", matches)
        for match in matches:
            # Find points that exist in both image 1 and current image
            img1_idx = None
            img_curr_idx = None
            
            for i, point_info in enumerate(match):
                if point_info[0] == 1:
                    img1_idx = i
                elif point_info[0] == img_idx:
                    img_curr_idx = i
            
            if img1_idx is not None and img_curr_idx is not None:
                # Get the 2D point in current image
                x_curr = match[img_curr_idx][1]
                y_curr = match[img_curr_idx][2]
                # x1 = match[1][1]
                # y1 = match[1][2]
                
                
                # Find corresponding 3D point from triangulation
                # This requires matching with the points used in triangulation
                for j, point1 in enumerate(points1):
                    if np.linalg.norm(point1 - np.array([match[img1_idx][1], match[img1_idx][2]])) < 1e-3:
                        # Found a match, use the corresponding 3D point
                        points_3d.append(X_refined[j])
                        points_2d.append(np.array([x_curr, y_curr]))
                        # cors_points.append(np.array([x1, y1]))
                        break
        
        points_3d = np.array(points_3d)
        points_2d = np.array(points_2d)

        
        print(f"Found {len(points_3d)} 2D-3D correspondences for PnP")
        
        if len(points_3d) < 6:
            print(f"Not enough correspondences for PnP with image {img_idx}. Skipping...")
            continue
        
        # Run PnP RANSAC
        R_new, C_new, inliers = pnp_ransac(points_3d, points_2d, camera_matrix)
        
        if R_new is None or C_new is None:
            print(f"PnP RANSAC failed for image {img_idx}. Skipping...")
            continue
        
        print(f"PnP RANSAC found {len(inliers)} inliers out of {len(points_3d)} points")
        
        # Filter points to keep only inliers
        inlier_3d = points_3d[inliers]
        inlier_2d = points_2d[inliers]
        # points_cors = cors_points[inliers]
        
        # Run Non-Linear PnP for refinement
        R_refined, t_refined = nonlinear_pnp(inlier_3d, inlier_2d, camera_matrix, R_new, C_new)
        
        # Store the refined camera pose
        all_camera_poses.append((R_refined, t_refined))
        # all_camera_poses.append((R_new, C_new))
       
        
        # Compute projection matrix for this camera
        #######################################################################################################################
        P_new = camera_matrix @ np.hstack((R_refined, -R_refined @ t_refined.reshape(3, 1)))
        
        # Triangulate new points between this camera and previous cameras
        # For simplicity, we'll triangulate with the first camera
        
        # Find points visible in both image 1 and current image but not used in PnP
        new_matches = []
        for match in matches:
            img1_idx = None
            img_curr_idx = None
            
            for i, point_info in enumerate(match):
                if point_info[0] == 1:
                    img1_idx = i
                elif point_info[0] == img_idx:
                    img_curr_idx = i
            
            if img1_idx is not None and img_curr_idx is not None:
                # Check if this point was not used in PnP
                point_used = False
                for j, point1 in enumerate(points1):
                    if np.linalg.norm(point1 - np.array([match[img1_idx][1], match[img1_idx][2]])) < 1e-3:
                        point_used = True
                        break
                
                if not point_used:
                    new_matches.append(match)
        
        if len(new_matches) > 0:
            print(f"Triangulating {len(new_matches)} new points between image 1 and image {img_idx}")
            
            # Extract 2D points for triangulation
            new_points1 = []
            new_points2 = []
            
            for match in new_matches:
                for i, point_info in enumerate(match):
                    if point_info[0] == 1:
                        new_points1.append(np.array([point_info[1], point_info[2]]))
                    elif point_info[0] == img_idx:
                        new_points2.append(np.array([point_info[1], point_info[2]]))
            
            new_points1 = np.array(new_points1)
            new_points2 = np.array(new_points2)
            
            # Triangulate new points
            P1 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = P_new
            
            X_new = linear_triangulation(camera_matrix, R1, C1, R_refined, t_refined, new_points1, new_points2)
            
            # Refine with non-linear triangulation
            X_new_refined = nonlinear_triangulation(P1, P2, new_points1, new_points2, X_new)
            
            # Add new points to our collection
            for i, point in enumerate(X_new_refined):
                world_points_dict[len(world_points_dict)] = point
            
            for i, point in enumerate(new_points1):
                coresponding_points_dict[len(coresponding_points_dict)] = point

            
            print(f"Added {len(X_new_refined)} new 3D points")

        image_path = f"../P2Data/{img_idx}.png"  # Adjust path if needed
        save_path = f"Data/IntermediateOutputImages/image{img_idx}_reprojection"
        # Call visualization function to overlay projections on the image
        image = cv2.imread(image_path)
        if image is None:
            print(f" Error: Image {image_path} not found.")
            return
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 1. Original (Green) + Linear (Blue)
        image_linear = image.copy()
        original_points = points_2d
        # projected_linearpnp = project_points(P_new, points_3d)  # Linear PnP
        projected_nonlinearpnp = project_points(P_new, points_3d)  # Non-Linear PnP

        # 2. Original (Green) + Non-Linear (Red)
        image_nonlinear = image.copy()
        for pt in original_points.astype(int):
            cv2.circle(image_nonlinear, tuple(pt), 3, (0, 255, 0), -1)  # Green for Original
        for pt in projected_nonlinearpnp.astype(int):
            cv2.circle(image_nonlinear, tuple(pt), 3, (0, 0, 255), -1)  # Red for Non-Linear
        cv2.imwrite(f"{save_path}_nonlinearpnp.png", image_nonlinear)
        print(f"Original + Non-Linear reprojection saved at {save_path}_nonlinear.png")
        print("Length of world_points_dict",len(original_points))
        print("Length of all_camera_poses",len(projected_nonlinearpnp))
        

        visualize_reprojection(image_path, projected_linear, projected_nonlinear, points2, save_path=f"Data/IntermediateOutputImages/image{img_idx}_reprojection_pnp.png")


    print("Length of world_points_dict",len(world_points_dict))
    
    print("--------------------------------------PnP Done-------------------------------------------------------")

    # Convert all camera poses to arrays for visualization
    R_All = np.array([pose[0] for pose in all_camera_poses])
    C_All = np.array([pose[1].reshape(3) for pose in all_camera_poses])

    print(f"R_All: {len(R_All)} camera rotations")
    print(f"C_All: {len(C_All)} camera centers")

    # Visualize the final reconstruction
    # Extract all 3D points for visualization
    all_points = np.array(list(world_points_dict.values()))

    # Create a directory for output images if it doesn't exist
    # os.makedirs("output", exist_ok=True)



    # If you still want to use your original plot_2D_projections function
    plot_2D_projections_PNP(
        X_initial=X_refined,  # Use initial triangulated points
        X_refined=all_points,  # Use all points as refined
        save_path="Data/IntermediateOutputImages/linear_vs_final_reconstruction_pnp.png"
    )

    print(f"Final reconstruction visualization saved to output directory")
    print(f"Total number of 3D points: {len(all_points)}")
    print(f"Total number of cameras: {len(R_All)}")
            
    fig, ax = plt.subplots()
    # plt.scatter(world_points[:,0], world_points[:,2], s=1, c='r', label="Before Bundle Adjustment")
    plt.scatter(X_refined[:,0], X_refined[:,2], s=1, c='b', label="Bundle Adjustment")
    for i in range(5):
        # Convert rotation matrix to Euler angles
        A = Rotation.from_matrix(R_All[i]).as_euler('XYZ')
        A_d = np.rad2deg(A)
        # print(f"Camera {label} position: {position}, orientation: {angles_deg}")
        
        ax.plot(C_All[i][0], C_All[i][2], marker=(3, 0, int(A_d[1])), markersize=15, linestyle='None', label=f'Camera {i+1}') 
        
        # Annotate camera with label
        correction = -0.1
        ax.annotate(i+1, xy=(C_All[i][0] + correction, C_All[i][2] + correction))
    
    plt.axis([-20, 20, -10, 25])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.title("Camera Poses and World Points before and after Bundel Adjustment")
    plt.savefig("Data/IntermediateOutputImages/Camera Poses and World Points before and after Bundle Adjustment.png")
    # plt.show()          
   #------------------------------------------------VISISBILITY MATRIX---------------------------------------------------------
    v,uv = BuildVisibilityMatrix(coresponding_points_dict, world_points_dict, dictionary_matches, 5)
    np.set_printoptions(threshold=np.inf)
    # print(v)
    # print(uv)
    print(f"Shape of uv: {uv.shape}")  # Expected: (num_cameras, num_points, 2)
    # print(f"Shape of points_2d: {points_2d.shape}")  # Expected: (num_cameras, num_points, 2)
#---------------------------------------------------------------------------------------------------------------------------------------------
    world_points = np.array(list(world_points_dict.values()))
    print(f"Total 3D Points: {world_points.shape[0]}")
    # print("World Points Array:")
    # print(world_points)
    # Print the entire world_points_dict
    # print("World Points Dictionary:")
    # for key, value in world_points_dict.items():
    #     print(f"Key: {key}, Value: {value}")
    # points_2d = np.full((len(R_All), X_refined.shape[0], 2), np.nan)  # Initialize with NaNs

    # for cam_idx in range(len(R_All)):
    #     for point_idx in range(X_refined.shape[0]):
    #         if v[cam_idx, point_idx]:  # If point is visible in this camera
    #             points_2d[cam_idx, point_idx] = uv[cam_idx, point_idx]  # Assign 2D point

    # Run Bundle Adjustment
    R_optimized, C_optimized, X_optimized = bundle_adjustment(
        uv, world_points, v, R_All, C_All, len(R_All), camera_matrix
    )

    print(f"--------------------------------------Bundle Adjustment Done-------------------------------------------------------")

    # Replace original values with optimized ones
    R_All, C_All, X_refined = R_optimized, C_optimized, X_optimized

    print(f"Final Optimized 3D Points: {X_refined.shape[0]}")
    print(f"Final Optimized Cameras: {len(R_All)}")

    fig, ax = plt.subplots()
    # plt.scatter(world_points[:,0], world_points[:,2], s=1, c='r', label="Before Bundle Adjustment")
    plt.scatter(X_refined[:,0], X_refined[:,2], s=1, c='b', label="After PNP")
    for i in range(5):
        # Convert rotation matrix to Euler angles
        A = Rotation.from_matrix(R_All[i]).as_euler('XYZ')
        A_d = np.rad2deg(A)
        # print(f"Camera {label} position: {position}, orientation: {angles_deg}")
        
        ax.plot(C_All[i][0], C_All[i][2], marker=(3, 0, int(A_d[1])), markersize=15, linestyle='None', label=f'Camera {i+1}') 
        
        # Annotate camera with label
        correction = -0.1
        ax.annotate(i+1, xy=(C_All[i][0] + correction, C_All[i][2] + correction))
    
    plt.axis([-20, 20, -10, 25])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.title("Camera Poses and World Points before and after PNP")
    plt.savefig("Data/IntermediateOutputImages/Camera Poses and World Points before and after PNP.png")
    # plt.show()        


if __name__ == "__main__":
    main()
