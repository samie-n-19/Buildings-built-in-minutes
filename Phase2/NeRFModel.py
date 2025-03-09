import torch
import torch.nn as nn
import numpy as np


# class NeRFmodel(nn.Module):
#     def __init__(self, embed_pos_L, embed_direction_L):
#         super(NeRFmodel, self).__init__()
#         #############################
#         # network initialization
#         #############################

#     def position_encoding(self, x, L):
#         #############################
#         # Implement position encoding here
#         #############################

#         return y

#     def forward(self, pos, direction):
#         #############################
#         # network structure
#         #############################

#         return output



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L=10, embed_direction_L=4):
        super(NeRFmodel, self).__init__()
        
        # Positional Encoding dimensions
        self.pos_dim = embed_pos_L * 6  # 3D (x,y,z) * (sin,cos) * L frequencies
        self.dir_dim = embed_direction_L * 6  # 2D (θ, φ) * (sin,cos) * L frequencies
        
        # MLP layers
        self.layers1 = nn.Sequential(
            nn.Linear(self.pos_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Skip connection
        self.fc_skip = nn.Linear(self.pos_dim + 256, 256)
        
        self.layers2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Output: Density (σ) and Feature Vector
        self.fc_sigma = nn.Linear(256, 1)  # Density output
        self.fc_feature = nn.Linear(256, 256)  # Feature vector
        
        # Additional MLP for RGB prediction
        self.fc_rgb1 = nn.Linear(256 + self.dir_dim, 256)
        self.fc_rgb2 = nn.Linear(256, 128)
        self.fc_rgb3 = nn.Linear(128, 3)  # Final RGB output
        
    def position_encoding(self, x, L):
        """ Applies positional encoding to input x """
        freq = 2.0 ** torch.arange(L, dtype=torch.float32, device=x.device)
        x = x.unsqueeze(-1) * freq  # Expand for broadcasting
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return enc.view(x.shape[0], -1)
    
    def forward(self, pos, direction):
        """ NeRF forward pass """
        pos_enc = self.position_encoding(pos, self.pos_dim // 6)  # Encode position
        direction_enc = self.position_encoding(direction, self.dir_dim // 6)  # Encode direction
        
        # First MLP with skip connection
        x = self.layers1(pos_enc)
        x = torch.cat([x, pos_enc], dim=-1)  # Skip connection
        x = self.fc_skip(x)
        x = self.layers2(x)
        
        # Compute density and feature vector
        sigma = self.fc_sigma(x)  # Density output
        feature = self.fc_feature(x)
        
        # Concatenate with viewing direction
        x = torch.cat([feature, direction_enc], dim=-1)
        x = F.relu(self.fc_rgb1(x))
        x = F.relu(self.fc_rgb2(x))
        rgb = torch.sigmoid(self.fc_rgb3(x))  # RGB values in range [0,1]
        
        return rgb, sigma
