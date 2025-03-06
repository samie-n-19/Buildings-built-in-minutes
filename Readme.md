# Structure from Motion (SfM) - Phase 1  

This repository contains the implementation of a classical Structure from Motion (SfM) pipeline, which reconstructs a 3D scene from multiple 2D images. The project follows a structured approach, including feature matching, epipolar geometry, triangulation, PnP, and Bundle Adjustment to refine the reconstruction.  

## File Structure  

```
Group11_p2.zip
│   README.md
|   ├── Phase1/
|   | ├── GetInliersRANSAC.py
|   | ├── EstimateFundamentalMatrix.py
|   | ├── EssentialMatrixFromFundamentalMatrix.py
|   | ├── ExtractCameraPose.py
|   | ├── LinearTriangulation.py
|   | ├── DisambiguateCameraPose.py
|   | ├── NonlinearTriangulation.py
|   | ├── PnPRANSAC.py
|   | ├── NonlinearPnP.py
|   | ├── BuildVisibilityMatrix.py
|   | ├── BundleAdjustment.py
|   | ├── Wrapper.py
|   | ├── utils.py
|   
└── Report.pdf
```

## Data Organization  

Please create a **directory outside the Phase1 folder** named **P2Data**, and place the required input files inside it:  

```
P2Data/
│   matching1.txt
│   matching2.txt
│   matching3.txt
│   matching4.txt
│   calibration.txt
│   1.png
│   2.png
│   3.png
│   4.png
│   5.png
```

This folder should contain the provided **feature matching files (`matching*.txt`)** and the **calibration file (`calibration.txt`)**, along with the **input images (`.png`)**.

## How to Run  

To execute the full SfM pipeline, navigate to the project directory Phase1 and run:

```bash
python3 Wrapper.py
```

This will process the images, compute camera poses, triangulate 3D points, refine them using Bundle Adjustment, and generate visualizations.

## Output  

The code generates intermediate results and outputs inside the `Data/IntermediateOutputImages/` directory. This includes:
- Epipolar geometry visualization
- Feature matching results after RANSAC
- 3D reconstruction plots before and after Bundle Adjustment
- Reprojected points for Linear and Non-Linear Triangulation & PnP

The final camera poses and 3D points are also refined through Bundle Adjustment.

## Notes  

- Ensure you have **Python 3** installed along with the required dependencies (`numpy`, `scipy`, `opencv-python`, `matplotlib`).  
- If the code fails due to missing input files, verify that `P2Data/` contains all required files.  
- The output images will be saved inside `Data/IntermediateOutputImages/`.  

For further details, refer to the **Report.pdf** included in the submission.  

