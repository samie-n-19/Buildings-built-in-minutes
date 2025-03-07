import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class NeRFDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=(256, 256)):
        """
        Args:
            root_dir (str): Path to the dataset directory (e.g., 'archive/nerf_synthetic/chair').
            split (str): Dataset split ('train', 'val', or 'test').
            img_size (tuple): Desired image size (width, height).
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        # Load transforms JSON file
        transforms_path = os.path.join(root_dir, f"transforms_{split}.json")
        if not os.path.exists(transforms_path):
            raise FileNotFoundError(f"Transforms file not found: {transforms_path}")
        
        with open(transforms_path, "r") as f:
            self.data = json.load(f)

        # Extract frames and camera parameters
        self.frames = self.data["frames"]
        self.camera_angle_x = self.data["camera_angle_x"]

        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),  # Resize images
            transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        """
        Returns:
            image (torch.Tensor): Preprocessed image tensor of shape [3, H, W].
            pose (torch.Tensor): Camera-to-world transformation matrix of shape [4, 4].
            focal_length (float): Focal length of the camera.
        """
        # Load image
        frame = self.frames[idx]
        img_path = os.path.join(self.root_dir, frame["file_path"] + ".png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert("RGB")  # Ensure 3 channels
        image = self.transform(image)  # Resize and convert to tensor

        # Load camera pose
        pose = np.array(frame["transform_matrix"], dtype=np.float32)
        pose = torch.tensor(pose, dtype=torch.float32)

        # Compute focal length
        focal_length = 0.5 * self.img_size[0] / np.tan(0.5 * self.camera_angle_x)

        return image, pose, focal_length

    def get_camera_intrinsics(self):
        """
        Returns:
            camera_matrix (torch.Tensor): Camera intrinsic matrix of shape [3, 3].
        """
        focal_length = 0.5 * self.img_size[0] / np.tan(0.5 * self.camera_angle_x)
        camera_matrix = torch.tensor([
            [focal_length, 0, self.img_size[0] / 2],
            [0, focal_length, self.img_size[1] / 2],
            [0, 0, 1]
        ], dtype=torch.float32)
        return camera_matrix

# Example Usage
if __name__ == "__main__":
    dataset = NeRFDataset(root_dir=os.path.abspath("../archive/nerf_synthetic/lego"), split="train")
    image, pose, focal_length = dataset[0]
    print("Image shape:", image.shape)  # Should output: torch.Size([3, 256, 256])
    print("Pose shape:", pose.shape)    # Should output: torch.Size([4, 4])
    print("Focal length:", focal_length)
