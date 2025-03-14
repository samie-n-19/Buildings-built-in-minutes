import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from NeRFDataset import NeRFDataset
from NeRFModel import NeRFmodel
import wandb
import torchvision
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

wandb.init(project="nerf-Training-new1"" ", name="nerfnew")

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
def ssim(img1, img2):
    return ssim_metric(img1, img2)

def loadDataset(data_path, mode):
    dataset = NeRFDataset(root_dir=data_path, split=mode)
    images, poses = [], []
    for i in range(len(dataset)):
        img, pose, _ = dataset[i]
        images.append(img)
        poses.append(pose)
    
    # Extract camera info from the dataset
    focal_length = 0.5 * dataset.img_size[0] / np.tan(0.5 * dataset.camera_angle_x)
    camera_info = {
        'focal_length': focal_length,
        'width': dataset.img_size[0],
        'height': dataset.img_size[1]
    }
    
    return camera_info, images, poses


def PixelToRay(camera_info, pose, pixel_position, args):
    """
    Convert specific pixel coordinates to ray origins and directions.
    
    Input:
        camera_info: Dictionary containing focal length, width, height
        pose: Camera pose in world frame (single 4x4 transformation matrix)
        pixel_position: Tensor of shape [N, 2] containing pixel coordinates (x, y)
        args: Additional arguments
    
    Outputs:
        ray_origins: Origins of rays in world space
        ray_directions: Directions of rays in world space
    """
    focal_length = camera_info['focal_length']
    width = camera_info['width']
    height = camera_info['height']
    
    # Extract pixel coordinates
    pixels_x, pixels_y = pixel_position[:, 0], pixel_position[:, 1]
    
    # Calculate normalized device coordinates
    x = (pixels_x - width / 2) / focal_length
    y = -(pixels_y - height / 2) / focal_length
    
    # Create ray directions in camera space
    ray_dirs_camera = torch.stack([x, y, -torch.ones_like(x)], dim=-1).to(device)
    
    # Transform ray directions to world space
    rotation = pose[:3, :3].to(device)
    ray_directions = torch.matmul(ray_dirs_camera, rotation.T)
    
    # Normalize ray directions
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    
    # Ray origins are the camera position in world coordinates
    translation = pose[:3, 3].view(1, 3).expand_as(ray_directions).to(device)
    ray_origins = translation
    
    return ray_origins, ray_directions



def generateBatch(images, poses, camera_info, args):
    H, W = images[0].shape[1:]
    rays_origin, rays_direction, gt_colors = [], [], []
    for _ in range(args.n_rays_batch):
        img_idx = np.random.randint(0, len(images))
        i = np.random.randint(0, H)
        j = np.random.randint(0, W)
        pixel = torch.tensor([[j, i]], dtype=torch.float32).to(device)
        ray_o, ray_d = PixelToRay(camera_info, poses[img_idx], pixel, args)
        rays_origin.append(ray_o[0])
        rays_direction.append(ray_d[0])
        gt_colors.append(images[img_idx][:, i, j])
    return torch.stack(rays_origin), torch.stack(rays_direction), torch.stack(gt_colors)

def render(model, rays_origin, rays_direction, args, stratified=True):
    N_rays = rays_origin.shape[0]
    near = 2.0
    far = 6.0

    if stratified:
        # Stratified sampling (generate n_sample bins)
        t_vals = torch.linspace(near, far, steps=args.n_sample + 1, device=rays_origin.device)  # [n_sample+1]
        lower = t_vals[:-1]  # [n_sample]
        upper = t_vals[1:]   # [n_sample]
        t_rand = torch.rand((N_rays, args.n_sample), device=rays_origin.device)
        t_vals = lower[None, :] + (upper - lower)[None, :] * t_rand  # [N_rays, n_sample]
    else:
        t_vals = torch.linspace(near, far, steps=args.n_sample, device=rays_origin.device)
        t_vals = t_vals.expand(N_rays, args.n_sample)

    # Sample 3D points
    pts = rays_origin[:, None, :] + t_vals[..., None] * rays_direction[:, None, :]
    dirs = rays_direction[:, None, :].expand(-1, args.n_sample, 3)

    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)

    rgb, sigma = model(pts_flat, dirs_flat)
    rgb = rgb.view(N_rays, args.n_sample, 3)
    sigma = sigma.view(N_rays, args.n_sample)

    delta = t_vals[:, 1:] - t_vals[:, :-1]
    delta = torch.cat([delta, delta[:, -1:]], dim=1)

    alpha = 1.0 - torch.exp(-sigma * delta)
    T = torch.cumprod(torch.cat([torch.ones((N_rays, 1), device=alpha.device), 1.0 - alpha + 1e-10], dim=1)[:, :-1], dim=1)
    weights = alpha * T
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)

    return rgb_map


def loss(gt, pred):
    return F.mse_loss(pred, gt)

def train(images, poses, camera_info, args):
    model = NeRFmodel(args.n_pos_freq, args.n_dirc_freq).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)
    writer = SummaryWriter(args.logs_path)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    model.train()

    for i in tqdm(range(args.max_iters)):
        rays_o, rays_d, gt_colors = generateBatch(images, poses, camera_info, args)
        rays_o, rays_d, gt_colors = rays_o.to(device), rays_d.to(device), gt_colors.to(device)

        rgb_pred = render(model, rays_o, rays_d, args, stratified=True)
        loss_val = F.mse_loss(gt_colors, rgb_pred)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        scheduler.step()

        if i % 10 == 0:
            writer.add_scalar("train_loss", loss_val.item(), i)
            wandb.log({"train_loss": loss_val.item()}, step=i)

        if i % args.save_ckpt_iter == 0:
            ckpt_path = os.path.join(args.checkpoint_path, f"nerf_{i}.pth")
            torch.save(model.state_dict(), ckpt_path)
            wandb.save(ckpt_path)

    print("Training complete.")
 

def test(images, poses, camera_info, args):
    model = NeRFmodel(args.n_pos_freq, args.n_dirc_freq).to(device)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, f"nerf_{args.max_iters}.pth")))
    model.eval()
    os.makedirs(args.images_path, exist_ok=True)

    psnr_vals, ssim_vals = [], []

    all_ray_origins = []
    all_ray_directions = []
    image_shapes = []


    # Step 1: Precompute ray origins and directions
    for idx in range(len(images)):
        gt_img = images[idx].to(device)
        H, W = gt_img.shape[1:]
        image_shapes.append((H, W))

        i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        pixels = torch.stack([j.reshape(-1), i.reshape(-1)], dim=-1).float().to(device)
        ray_o, ray_d = PixelToRay(camera_info, poses[idx], pixels, args)

        all_ray_origins.append(ray_o)
        all_ray_directions.append(ray_d)


    # Step 2: Chunked Rendering + PSNR/SSIM Logging
    with torch.no_grad():
        for idx in range(len(images)):
            gt_img = images[idx].to(device)
            ray_o = all_ray_origins[idx]
            ray_d = all_ray_directions[idx]
            H, W = image_shapes[idx]

            rgb_chunks = []
            chunk_size = 8192

            for i in range(0, ray_o.shape[0], chunk_size):
                rgb_chunk = render(model, ray_o[i:i+chunk_size], ray_d[i:i+chunk_size], args, stratified=False)
                rgb_chunks.append(rgb_chunk)

            rgb_pred = torch.cat(rgb_chunks, dim=0)
            rgb_image = rgb_pred.reshape(H, W, 3).permute(2, 0, 1)

            # Save rendered image
            output_img = (rgb_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(args.images_path, f"rendered_{idx}.png"), output_img)

            # Compute metrics
            pred_img = rgb_image.clamp(0, 1).unsqueeze(0)
            gt_img = gt_img.unsqueeze(0)
            psnr_val = psnr(pred_img, gt_img).item()
            ssim_val = ssim(pred_img, gt_img).item()

            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val)
            wandb.log({"Test_PSNR": psnr_val, "Test_SSIM": ssim_val}, step=idx)

    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)
    print(f"Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}")
    wandb.log({"Avg PSNR": avg_psnr, "Avg SSIM": avg_ssim})
 

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="../archive/nerf_synthetic/lego", help="dataset path")
    parser.add_argument('--mode', default='train', help="train/test/val")
    parser.add_argument('--lrate', default=5e-4, type=float, help="training learning rate")
    parser.add_argument('--n_pos_freq', default=10, type=int, help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq', default=4, type=int, help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch', default=32*32, type=int, help="number of rays per batch")
    parser.add_argument('--n_sample', default=400, type=int, help="number of sample per ray")
    parser.add_argument('--max_iters', default=300000, type=int, help="number of max iterations for training")
    parser.add_argument('--logs_path', default="./logs/", help="logs path")
    parser.add_argument('--checkpoint_path', default="./Phase2/example_checkpoint/", help="checkpoints path")
    parser.add_argument('--load_checkpoint', default=True, help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter', default=1000, type=int, help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/", help="folder to store images")
    return parser

def main(args):
    print("Loading dataset...")
    camera_info, images, poses = loadDataset(args.data_path, args.mode)
    if args.mode == 'train':
        train(images, poses, camera_info, args)
    elif args.mode == 'test':
        test(images, poses, camera_info, args)

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)