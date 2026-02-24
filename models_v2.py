"""
Enhanced Neural Distinguisher for ISA-NDC
Target: 5-round PRESENT with >85% accuracy
"""
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Learnable positional encoding for 64-bit positions + CLS token."""
    def __init__(self, d_model, max_len=65):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ResBlock(nn.Module):
    """Enhanced residual block with BatchNorm and Dropout."""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.GELU()  # GELU often better than ReLU for transformers
        self.dropout = nn.Dropout2d(dropout)
        
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class NibbleEmbedder(nn.Module):
    """
    Groups 64 bits into 16 nibbles (4-bits each) and embeds them.
    Explicitly aligns with block cipher S-box structure.
    """
    def __init__(self, d_model):
        super().__init__()
        self.nibble_proj = nn.Sequential(
            nn.Linear(4, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )
        
    def forward(self, x):
        # x: (B, 64)
        B = x.shape[0]
        x = x.view(B, 16, 4) # 16 nibbles
        return self.nibble_proj(x) # (B, 16, d_model)

class EnhancedDistinguisher(nn.Module):
    """
    v4 Publication-Ready Neural Distinguisher:
    - S-box aware Nibble Embedding
    - Deep ResNet-inspired bit correlation layer
    - 8-layer ViT with learnable positional encoding
    - Residual MLP classification head
    """
    def __init__(self, d_model=256, nhead=8, num_layers=8, dropout=0.1):
        super().__init__()
        
        # 1. Local Bit Pattern Extraction (ResNet on 8x8 grid)
        self.backbone = nn.Sequential(
            ResBlock(1, 32, dropout),
            ResBlock(32, 64, dropout),
            ResBlock(64, 128, dropout),
            ResBlock(128, d_model, dropout),
        )
        
        # 2. Global Diffusion Modeling (Transformer)
        # Nibble-level embedding is concatenated with ResNet features
        self.nibble_embed = NibbleEmbedder(d_model)
        
        # Grid feature projection (8x8 -> 64 tokens)
        self.feature_proj = nn.Linear(d_model, d_model)
        
        self.pos_encoding = PositionalEncoding(d_model, max_len=64 + 16 + 1) # ResNet + Nibbles + CLS
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model * 4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # 3. Residual MLP Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, 64)
        B = x.shape[0]
        
        # Branch 1: ResNet Spatial (8x8)
        x_grid = x.view(B, 1, 8, 8)
        f_grid = self.backbone(x_grid) # (B, 256, 8, 8)
        f_grid = f_grid.view(B, f_grid.shape[1], -1).permute(0, 2, 1) # (B, 64, 256)
        f_grid = self.feature_proj(f_grid)
        
        # Branch 2: Nibble (16x4)
        f_nibble = self.nibble_embed(x) # (B, 16, 256)
        
        # Combine
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls_tokens, f_grid, f_nibble], dim=1) # (B, 1+64+16, 256)
        
        x_seq = self.pos_encoding(x_seq)
        x_seq = self.transformer(x_seq)
        
        return self.classifier(x_seq[:, 0, :])


class LightDistinguisher(nn.Module):
    """
    Lighter model for faster training on CPU.
    Still effective for 5-round attack.
    """
    def __init__(self, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.backbone = nn.Sequential(
            ResBlock(1, 32, dropout),
            ResBlock(32, 64, dropout),
            ResBlock(64, d_model, dropout),
        )
        
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 8, 8)
            
        x = self.backbone(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        

class EnhancedDistinguisher6R(nn.Module):
    """
    Deeper architecture for 6-Round Distinguisher (Priority 1 Target).
    - 12 Transformer Layers (vs 8)
    - 384 dim embedding (vs 256)
    - Higher dropout (0.15) for regularization
    """
    def __init__(self, d_model=384, nhead=12, num_layers=12, dropout=0.15):
        super().__init__()
        
        # 1. Local Bit Pattern Extraction (ResNet)
        # Scaled up backbone
        self.backbone = nn.Sequential(
            ResBlock(1, 32, dropout),
            ResBlock(32, 64, dropout),
            ResBlock(64, 128, dropout),
            ResBlock(128, 256, dropout),
            ResBlock(256, d_model, dropout), # Extra layer
        )
        
        # 2. Global Diffusion Modeling (Transformer)
        self.nibble_embed = NibbleEmbedder(d_model)
        self.feature_proj = nn.Linear(d_model, d_model)
        
        self.pos_encoding = PositionalEncoding(d_model, max_len=64 + 16 + 1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model * 4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # 3. Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, 64)
        B = x.shape[0]
        
        # Branch 1: ResNet
        x_grid = x.view(B, 1, 8, 8)
        f_grid = self.backbone(x_grid) 
        f_grid = f_grid.view(B, f_grid.shape[1], -1).permute(0, 2, 1) 
        f_grid = self.feature_proj(f_grid)
        
        # Branch 2: Nibble
        f_nibble = self.nibble_embed(x)
        
        # Combine
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls_tokens, f_grid, f_nibble], dim=1)
        
        x_seq = self.pos_encoding(x_seq)
        x_seq = self.transformer(x_seq)
        
        return self.classifier(x_seq[:, 0, :])

