import torch
import torch.nn as nn
import numpy as np

# ------------------------------------------------------------------------------
# Image Structure Analyzer (Delta Selector)
# ------------------------------------------------------------------------------
class DeltaSelector(nn.Module):
    """
    CNN to predict probability of blocks having useful differentials.
    Input: Image patch (64x64 or full image?) - Plan said (1, H, W)
    For PoC we used statistical analysis, but here is the model definition.
    """
    def __init__(self):
        super(DeltaSelector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 64) # Regress to a 64-bit delta? Or class?

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)

# ------------------------------------------------------------------------------
# Neural Distinguisher: Hybrid ViT-ResNet
# ------------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class Distinguisher(nn.Module):
    """
    Input: 64-bit block difference (represented as 1x8x8 binary image)
    Output: Probability (Real vs Random)
    """
    def __init__(self, input_dim=64):
        super().__init__()
        # Input is (Batch, 1, 8, 8)
        
        # ResNet Backbone (Process local bit interactions)
        self.backbone = nn.Sequential(
            ResBlock(1, 32),
            ResBlock(32, 64),
            nn.Flatten(2) # (Batch, 64, 64) -> 8x8 flattened to 64 pixels? No.
            # Output of ResBlock(64, ch=64) on 8x8 is (B, 64, 8, 8)
            # Flatten(2) -> (B, 64, 64) ?
        )
        
        # ViT Part
        # Treat the 64 channels as embedding dim? Or 64 pixels as sequence?
        # Option A: Flatten spatial (8x8=64 tokens), embedding is channel depth (64).
        # Sequence length = 64 (pixels). Dim = 64 (channels).
        
        self.vit_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.vit_layer, num_layers=4)
        
        # Classifier
        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (Batch, 64) usually. Need to reshape to (Batch, 1, 8, 8)
        # Assuming input is float tensor of bits
        
        if x.dim() == 2:
           x = x.view(-1, 1, 8, 8)
           
        # ResNet
        # (B, 1, 8, 8) -> (B, 64, 8, 8)
        x = self.backbone(x) 
        
        # Prepare for ViT
        # We want sequence of "Patches" or pixels.
        # Let's treat each spatial position (bit) as a token.
        # (B, 64, 8, 8) -> (B, 64, 64) (Channels, Spatial)
        # Permute to (B, 64, 64) -> (B, Spatial, Channels) ?
        # d_model=64.
        
        x = x.flatten(2) # (B, 64, 64)
        x = x.permute(0, 2, 1) # (B, 64, 64) -> Sequence=64, Dim=64
        
        # Transformer
        x = self.transformer(x) # (B, 64, 64)
        
        # Pool (Mean of sequence)
        x = x.mean(dim=1) # (B, 64)
        
        return self.mlp(x)

