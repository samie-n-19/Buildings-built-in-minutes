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


# Phase 2: NeRF Implementation

### Code Structure
```
Phase2/
├── NeRFModel.py
├── NeRFDataset.py
└── Wrapper.py

```

### File Descriptions

- **NeRFModel.py:** Defines the neural network architecture (MLP-based) for NeRF. Includes positional encoding (PE) and a simplified model without positional encoding.

- **NeRFDataset.py:** Custom PyTorch dataset loader that reads NeRF-compatible datasets, extracting camera poses, images, and camera intrinsics from JSON files.

- **Wrapper.py:** Main training and testing script which:
  - Implements ray generation, batch processing, and sampling strategies (stratified and uniform).
  - Defines rendering and metric computation (PSNR and SSIM).
  
### File Tree
```
Group11_p2.zip
│
├── Phase2
│   ├── Wrapper.py
│   ├── NeRFModel.py
│   ├── NeRFDataset.py
│   ├── NeRF_lego_with_pe.gif
│   ├── NeRF_lego_without_pe.gif
│   ├── NeRF_ship_with_pe.gif
│   ├── NeRF_ship_without_pe.gif
|   └── NeRF_Protein_shake.gif
├── README.md
└── Report.pdf

```

## Dependencies
- Python 3.x
- PyTorch
- torchvision
- numpy
- WandB (for logging)
- imageio
- tqdm
- torchmetrics

```bash
pip install torch torchvision numpy imageio wandb torchmetrics tqdm
```

## Installation



## Dataset Structure
The datasets are from the [NeRF Synthetic Dataset](https://github.com/bmild/nerf#synthetic-data), and should have the following structure:
The Extra Credit Data set is uploaded in this link https://wpi0-my.sharepoint.com/:f:/g/personal/snaukudkar_wpi_edu/Eqqr71VI2n1AlbKXxXxYxHEBroBwyfZ1MSl3SHbY531faA?e=aGZYoC

```
nerf_synthetic/
├── lego/
│   ├── transforms_train.json
│   ├── transforms_val.json
│   └── transforms_test.json
├── ship/
│   └── ...
└── protein_shake/
    └── transforms_train.json
```

## Training
To train the NeRF model, run the following command:

```bash
python Wrapper.py --data_path "../archive/nerf_synthetic/lego" \
                  --mode "train" \
                  --checkpoint_path "./checkpoints/lego" \
                  --max_iters 500000
```

Replace `data_path`, `checkpoint_path`, and `max_iters` with your specific dataset and desired configuration.

## Testing
To test the NeRF model and generate novel view images, run:

This is for Ship without pe 

```bash

python Wrapper.py --data_path "../archive/nerf_synthetic/ship" \
                  --mode "test" \
                  --images_path "./images/ship_without_PE/" \
                  --checkpoint_path "/path/to/checkpoint/ship_withoutPE/" \
                  --max_iters 339000

```

Repeat this process similarly for other datasets (lego and protein_shake) as provided, just change the data_path, image_path, checkpoint_path, and Max Iteration
and get all the chekpoints from this onedrive link https://wpi0-my.sharepoint.com/:f:/g/personal/snaukudkar_wpi_edu/Eqqr71VI2n1AlbKXxXxYxHEBroBwyfZ1MSl3SHbY531faA?e=aGZYoC

For Ship with pe --max_iters 299000
For lego with pe --max_iters 288000
For lego without pe --max_iters 181000
For Protein shake --max_iters 351000

## Generating Results and Metrics
- Rendered test images will be stored in the specified `images_path`.
- Computes PSNR and SSIM values, which are logged on Weights & Biases.

## Logging and Monitoring
- Integrated logging with [Weights & Biases (wandb)](https://wandb.ai/). Metrics logged:
  - Training Loss
  - Test PSNR and SSIM scores

## Creating GIF Outputs
After test rendering, the GIF will be automatically generated.
