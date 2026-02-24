"""
train_v4_cpu.py  — CPU-feasible multi-stage training.

Uses a compact NibbleDistinguisher (d_model=128, 4 Transformer layers, 4 heads)
instead of the full EnhancedDistinguisher so each epoch finishes in ~5-15 min
on CPU.

Stage 1: Bootstrap  — 5-round PRESENT  (dataset_v3_bootstrap.pt)  target >80%
Stage 2: Transition — 6-round PRESENT  (dataset_v3.pt)            target >65%

Usage:
    python train_v4_cpu.py --stage 1
    python train_v4_cpu.py --stage 2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import json
import os
import random
import argparse

# ──────────────────────────────────────────────
# Model: compact NibbleDistinguisher
# ──────────────────────────────────────────────
class NibbleProjector(nn.Module):
    """Map 16 nibbles (4-bit each) → 16 tokens of dim d_model."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, x):          # x: (B, 64)
        B = x.shape[0]
        x = x.view(B, 16, 4)      # 16 nibbles of 4 bits
        return self.proj(x)        # (B, 16, d_model)


class BitProjector(nn.Module):
    """Project each of 64 bits independently → 64 tokens of dim d_model."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(1, d_model)

    def forward(self, x):            # x: (B, 64)
        return self.proj(x.unsqueeze(-1))   # (B, 64, d_model)


class NibbleDistinguisher(nn.Module):
    """
    Compact neural distinguisher combining:
      • Bit-level projection (64 tokens)
      • Nibble-level S-box aware projection (16 tokens)
      • Shallow Transformer (4 layers, d_model=128, 4 heads)
    """
    def __init__(self, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.bit_proj    = BitProjector(d_model)
        self.nibble_proj = NibbleProjector(d_model)

        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Learnable positional encoding: 1 CLS + 64 bit tokens + 16 nibble tokens
        self.pos_enc = nn.Parameter(torch.randn(1, 81, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation='gelu', batch_first=True,
            norm_first=True   # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):          # x: (B, 64) float
        B = x.shape[0]
        f_bit    = self.bit_proj(x)               # (B, 64, d_model)
        f_nibble = self.nibble_proj(x)            # (B, 16, d_model)
        cls      = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        seq      = torch.cat([cls, f_bit, f_nibble], dim=1)  # (B, 81, d_model)
        seq      = seq + self.pos_enc[:, :seq.size(1), :]
        seq      = self.transformer(seq)
        return self.head(seq[:, 0, :])            # CLS token → prediction


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class DistinguisherDataset(Dataset):
    def __init__(self, pt_file: str):
        loaded = torch.load(pt_file, weights_only=False)
        self.data   = loaded['data']
        self.labels = loaded['labels']

    def __len__(self):  return len(self.data)

    def __getitem__(self, idx):
        val  = int(self.data[idx])
        bits = [(val >> i) & 1 for i in range(63, -1, -1)]
        return torch.tensor(bits, dtype=torch.float32), self.labels[idx]


# ──────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        sm_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(pred, sm_target)


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train_stage(stage: int = 1):
    DATA_FILE   = "dataset_v3_bootstrap.pt" if stage == 1 else "dataset_v3.pt"
    BEST_MODEL  = f"best_model_v4cpu_s{stage}.pth"
    CKPT_FILE   = f"checkpoint_v4cpu_s{stage}.pth"
    PREV_MODEL  = "best_model_v4cpu_s1.pth"
    HISTORY_FILE = f"history_v4cpu_s{stage}.json"

    BATCH_SIZE = 512
    EPOCHS     = 20 if stage == 1 else 25
    MAX_LR     = 3e-3 if stage == 1 else 8e-4
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Stage {stage} | Device: {DEVICE} | Epochs: {EPOCHS} ===")

    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found. Aborting.")
        return

    # Data
    full_ds = DistinguisherDataset(DATA_FILE)
    n = len(full_ds)
    n_train = int(0.8 * n)
    train_ds, val_ds = random_split(
        full_ds, [n_train, n - n_train],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Batches/epoch: {len(train_loader)}")

    # Model
    model = NibbleDistinguisher(d_model=128, nhead=4, num_layers=4).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    if stage == 2 and os.path.exists(PREV_MODEL):
        print(f"Loading Stage 1 weights from {PREV_MODEL}")
        model.load_state_dict(torch.load(PREV_MODEL, map_location=DEVICE, weights_only=True))
    elif stage == 2:
        print("WARNING: Stage 1 model not found. Starting Stage 2 from scratch.")

    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=MAX_LR,
                           steps_per_epoch=len(train_loader), epochs=EPOCHS)
    criterion = LabelSmoothingBCE(smoothing=0.1)

    # Resume from checkpoint
    start_epoch = 0
    best_acc    = 0.0
    history     = {'loss': [], 'val_acc': []}

    if os.path.exists(CKPT_FILE):
        ckpt = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc    = ckpt.get('best_acc', 0.0)
        history     = ckpt.get('history', history)
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")

    print("Starting training...")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"  Ep {epoch+1}/{EPOCHS} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}", flush=True)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                targets = targets.to(DEVICE).unsqueeze(1)
                preds   = model(inputs.to(DEVICE))
                correct += ((preds > 0.5) == targets).sum().item()
        acc = correct / len(val_ds)
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={acc:.4f}", flush=True)
        history['loss'].append(avg_loss)
        history['val_acc'].append(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL)
            print(f"  -> Best model saved ({BEST_MODEL}) acc={acc:.4f}", flush=True)

        # Save checkpoint each epoch
        torch.save({
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc':  best_acc,
            'history':   history,
        }, CKPT_FILE)

    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n=== Stage {stage} Complete. Best Val Acc: {best_acc:.4f} ===", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    train_stage(stage=args.stage)
