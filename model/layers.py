import torch.nn as nn
from model.pointnet2_utils import PointNetSetAbstractionMsg
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FeatureUpsampler(nn.Module):
    def __init__(self, input_channels, layers=[2,4,8]):
        super(FeatureUpsampler, self).__init__()
        
        # First conv layer to reduce channels after initial upsample
        self.conv1 = nn.Conv2d(input_channels, input_channels//layers[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(input_channels//layers[0])
        
        # Second conv layer to reduce channels after second upsample
        self.conv2 = nn.Conv2d(input_channels//layers[0], input_channels//layers[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(input_channels//layers[1])
        
        # Third conv layer to reach final target channels
        self.conv3 = nn.Conv2d(input_channels//layers[1], input_channels//layers[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(input_channels//layers[2])

    def forward(self, x):
        # Initial shape: (B, 512, 8, 8)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 512, 16, 16)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 256, 16, 16)
        
        # Step 2: Upsample to 56x56, then apply second conv layer
        x = F.interpolate(x, size=(56, 56), mode='bilinear', align_corners=False)  # (B, 256, 56, 56)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, 56, 56)
        
        # Step 3: Upsample to 112x112, then apply third conv layer
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 128, 112, 112)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 64, 112, 112)
        
        return x

class PointFeatureDownsampler(nn.Module):
    def __init__(self, input_dim=512, output_dim=64):
        super(PointFeatureDownsampler, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, input_dim//2, kernel_size=1)
        self.conv2 = nn.Conv1d(input_dim//2, output_dim, kernel_size=1)
    
    def forward(self, x):
        # Apply Conv1d transformations with ReLU activations
        x = F.relu(self.conv1(x))  # Shape: (B, 256, N)
        x = self.conv2(x)          # Shape: (B, 512, N)
        return x
            
class SmallUpsampleNet(nn.Module):
    def __init__(self, dino_dimension, resolution):
        super(SmallUpsampleNet, self).__init__()
        
        self.dino_dimension = dino_dimension
        self.h, self.w = resolution

        # Initial Conv to reduce channels from 769 to 128
        self.conv1 = nn.Conv2d(self.dino_dimension+1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        # Upsample to 32x32, followed by Conv layer
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Upsample to 64x64, followed by Conv layer
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Upsample to 128x128, followed by Conv layer
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        
        # Final upsample to 224x224, output to 1 channel
        self.conv5 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Weight initialization
        self._initialize_weights()

    def forward(self, x):
        # Initial channel reduction
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Upsampling steps with Conv + BatchNorm + ReLU
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample to 32x32
        x = self.relu(self.bn2(self.conv2(x)))
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample to 64x64
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample to 128x128
        x = self.relu(self.bn4(self.conv4(x)))
        
        x = F.interpolate(x, size=(self.h, self.w), mode='bilinear', align_corners=False)  # Final upsample to 224x224
        x = self.conv5(x)  # Output to 1 channel
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class PointEncoder(nn.Module):
    def __init__(self, emb_dim, normal_channel, additional_channel, N_p, cfg_layers=None):
        super().__init__()
        self.N_p = N_p
        self.normal_channel = normal_channel

        default_layers = [
            dict(
                npoint=emb_dim,
                radius=[0.1, 0.2, 0.4],
                nsample=[32, 64, 128],
                in_channel=3 + additional_channel,
                mlp=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            ),
            dict(
                npoint=128,
                radius=[0.4, 0.8],
                nsample=[64, 128],
                in_channel=320,
                mlp=[[128, 128, 256], [128, 196, 256]],
            ),
            dict(
                npoint=N_p,
                radius=[0.2, 0.4],
                nsample=[16, 32],
                in_channel=512,
                mlp=[[128, 128, 256], [128, 196, 256]],
            ),
        ]
        cfg_layers = cfg_layers or default_layers

        # Construct SA layers and explicitly assign sa1/sa2/sa3 for checkpoint compatibility
        self.sa1 = self._build_sa(cfg_layers[0])
        self.sa2 = self._build_sa(cfg_layers[1])
        self.sa3 = self._build_sa(cfg_layers[2])


    def _build_sa(self, cfg):
        """Helper to instantiate a PointNet++ SA layer."""
        return PointNetSetAbstractionMsg(
            cfg["npoint"],
            cfg["radius"],
            cfg["nsample"],
            cfg["in_channel"],
            cfg["mlp"],
        )


    def forward(self, xyz):

        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return [
            [l0_xyz, l0_points],
            [l1_xyz, l1_points],
            [l2_xyz, l2_points],
            [l3_xyz, l3_points],
        ]
