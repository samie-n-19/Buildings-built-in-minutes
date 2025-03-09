import argparse
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt
import os
from NeRFDataset import *
from NeRFModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def loadDataset(data_path, mode):
    dataset = NeRFDataset(root_dir=data_path, split=mode)
    images, poses = [], []
    for i in range(len(dataset)):
        img, pose, _ = dataset[i]
        images.append(img)
        poses.append(pose)
    camera_matrix = dataset.get_camera_intrinsics()
    return camera_matrix, images, poses

def PixelToRay(camera_info, pose, pixel_position, args):
    W = camera_info[0, 2] * 2
    H = camera_info[1, 2] * 2
    ones = torch.ones(pixel_position.shape[0], 1, device=pixel_position.device)
    pixel_homo = torch.cat([pixel_position, ones], dim=1)
    cam_dir = torch.inverse(camera_info).to(pixel_position.device) @ pixel_homo.T
    cam_dir = cam_dir.T
    cam_dir = cam_dir / torch.norm(cam_dir, dim=1, keepdim=True)
    ray_directions = cam_dir @ pose[:3, :3].T
    ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)
    ray_origins = pose[:3, 3].expand_as(ray_directions)
    return ray_origins, ray_directions

def generateBatch(images, poses, camera_info, args):
    H, W = images[0].shape[1:]
    rays_origin, rays_direction, gt_colors = [], [], []
    for _ in range(args.n_rays_batch):
        img_idx = np.random.randint(0, len(images))
        i = np.random.randint(0, H)
        j = np.random.randint(0, W)
        pixel = torch.tensor([[j, i]], dtype=torch.float32)
        ray_o, ray_d = PixelToRay(camera_info, poses[img_idx], pixel, args)
        rays_origin.append(ray_o[0])
        rays_direction.append(ray_d[0])
        gt_colors.append(images[img_idx][:, i, j])
    return torch.stack(rays_origin), torch.stack(rays_direction), torch.stack(gt_colors)

def render(model, rays_origin, rays_direction, args):
    all_rgb = []
    for i in range(rays_origin.shape[0]):
        ray_o = rays_origin[i]
        ray_d = rays_direction[i]
        t_vals = torch.linspace(2.0, 6.0, steps=args.n_sample).to(ray_o.device)
        pts = ray_o[None, :] + t_vals[:, None] * ray_d[None, :]
        dirs = ray_d.expand(args.n_sample, 3)
        rgb, sigma = model(pts, dirs)
        delta = t_vals[1] - t_vals[0]
        weights = 1.0 - torch.exp(-sigma.squeeze() * delta)
        rgb_map = torch.sum(weights[:, None] * rgb, dim=0)
        all_rgb.append(rgb_map)
    return torch.stack(all_rgb)

def loss(groundtruth, prediction):
    return F.mse_loss(prediction, groundtruth)

def train(images, poses, camera_info, args):
    model = NeRFmodel(args.n_pos_freq, args.n_dirc_freq).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lrate))
    writer = SummaryWriter(args.logs_path)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    for i in tqdm(range(int(args.max_iters))):
        rays_o, rays_d, gt_colors = generateBatch(images, poses, camera_info, args)
        rays_o, rays_d, gt_colors = rays_o.to(device), rays_d.to(device), gt_colors.to(device)
        rgb_pred = render(model, rays_o, rays_d, args)
        loss_val = loss(gt_colors, rgb_pred)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        if i % 10 == 0:
            writer.add_scalar("train_loss", loss_val.item(), i)
        if i % int(args.save_ckpt_iter) == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"nerf_{i}.pth"))
    print("Training complete.")

def test(images, poses, camera_info, args):
    model = NeRFmodel(args.n_pos_freq, args.n_dirc_freq).to(device)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, f"nerf_{args.max_iters}.pth")))
    model.eval()
    os.makedirs(args.images_path, exist_ok=True)
    with torch.no_grad():
        for idx in range(len(images)):
            rendered_image = torch.zeros_like(images[idx])
            H, W = rendered_image.shape[1:]
            for i in range(H):
                for j in range(W):
                    pixel = torch.tensor([[j, i]], dtype=torch.float32)
                    ray_o, ray_d = PixelToRay(camera_info, poses[idx], pixel, args)
                    t_vals = torch.linspace(2.0, 6.0, steps=args.n_sample).to(ray_o.device)
                    pts = ray_o[0][None, :] + t_vals[:, None] * ray_d[0][None, :]
                    dirs = ray_d[0].expand(args.n_sample, 3)
                    rgb, sigma = model(pts, dirs)
                    delta = t_vals[1] - t_vals[0]
                    weights = 1.0 - torch.exp(-sigma.squeeze() * delta)
                    rgb_map = torch.sum(weights[:, None] * rgb, dim=0)
                    rendered_image[:, i, j] = rgb_map.cpu()
            output_img = (rendered_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(args.images_path, f"rendered_{idx}.png"), output_img)

def main(args):
    print("Loading data...")
    camera_info, images, poses = loadDataset(args.data_path, args.mode)
    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="../archive/nerf_synthetic/lego", help="dataset path")
    parser.add_argument('--mode', default='train', help="train/test/val")
    parser.add_argument('--lrate', default=5e-4, type=float, help="training learning rate")
    parser.add_argument('--n_pos_freq', default=10, type=int, help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq', default=4, type=int, help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch', default=32*32*4, type=int, help="number of rays per batch")
    parser.add_argument('--n_sample', default=400, type=int, help="number of sample per ray")
    parser.add_argument('--max_iters', default=10000, type=int, help="number of max iterations for training")
    parser.add_argument('--logs_path', default="./logs/", help="logs path")
    parser.add_argument('--checkpoint_path', default="./Phase2/example_checkpoint/", help="checkpoints path")
    parser.add_argument('--load_checkpoint', default=True, help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter', default=1000, type=int, help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/", help="folder to store images")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)
