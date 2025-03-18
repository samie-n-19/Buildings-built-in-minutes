import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class NeRFDataset(Dataset):
    def __init__(self, root_dir, split="train"):
    def __init__(self, root_dir, split="train"):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            split (str): Dataset split ('train', 'val', or 'test').
        """
        self.root_dir = root_dir
        self.split = split

        # Load transforms JSON file
        transforms_path = os.path.join(root_dir, f"transforms_{split}.json")
        if not os.path.exists(transforms_path):
            raise FileNotFoundError(f"Transforms file not found: {transforms_path}")


        with open(transforms_path, "r") as f:
            self.data = json.load(f)

        # Extract frames and camera parameters
        self.frames = [
            frame for frame in self.data["frames"]
            if not frame["file_path"].endswith("_depth_0001")
        ]
        self.camera_angle_x = self.data["camera_angle_x"]

        self.transform = transforms.ToTensor()

        sample_img_path = os.path.join(root_dir, self.frames[0]["file_path"] + ".png")
        sample_image = Image.open(sample_img_path).convert("RGB")
        self.img_size = sample_image.size  # (width, height)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        """
        Returns:
            image (torch.Tensor): Image tensor [3, H, W].
            pose (torch.Tensor): Camera-to-world matrix [4, 4].
            camera_info (dict): Dictionary containing focal length, width, and height.
            image (torch.Tensor): Image tensor [3, H, W].
            pose (torch.Tensor): Camera-to-world matrix [4, 4].
            camera_info (dict): Dictionary containing focal length, width, and height.
        """
        frame = self.frames[idx]
        img_path = os.path.join(self.root_dir, frame["file_path"] + ".png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        pose = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        
        # Calculate focal length based on camera angle and image size
        pose = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        
        # Calculate focal length based on camera angle and image size
        focal_length = 0.5 * self.img_size[0] / np.tan(0.5 * self.camera_angle_x)
        
        # Create camera info dictionary
        camera_info = {
            'focal_length': focal_length,
            'width': self.img_size[0],
            'height': self.img_size[1]
        }

        return image, pose, camera_info
        
        # Create camera info dictionary
        camera_info = {
            'focal_length': focal_length,
            'width': self.img_size[0],
            'height': self.img_size[1]
        }

        return image, pose, camera_info

    def get_camera_intrinsics(self):
        """
        Returns:
            camera_matrix (torch.Tensor): [3, 3] intrinsic matrix.
            camera_matrix (torch.Tensor): [3, 3] intrinsic matrix.
        """
        focal_length = 0.5 * self.img_size[0] / np.tan(0.5 * self.camera_angle_x)
        camera_matrix = torch.tensor([
            [focal_length, 0, self.img_size[0] / 2],
            [0, focal_length, self.img_size[1] / 2],
            [0, 0, 1]
        ], dtype=torch.float32)
        return camera_matrix

if __name__ == "__main__":
    dataset = NeRFDataset(root_dir=os.path.abspath("../archive/nerf_synthetic/protein_shake"), split="test")
    for idx in range(len(dataset)):
        image, pose, camera_info = dataset[idx]
        print(f"Image shape: {image.shape}, Pose shape: {pose.shape}, Camera Info: {camera_info}")
        if idx == 0:
            break