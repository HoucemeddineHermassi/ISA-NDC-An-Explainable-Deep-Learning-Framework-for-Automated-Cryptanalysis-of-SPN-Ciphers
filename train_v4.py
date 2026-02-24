"""
Training script v4: Publication Refinement.
- Stage 1: Bootstrap on 5-round data.
- Stage 2: Fine-tune on 6-round data.
- Uses OneCycleLR and EnhancedDistinguisher (v4).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
from models_v2 import EnhancedDistinguisher
import numpy as np
import json
import os
import random

class DistinguisherDataset(Dataset):
    def __init__(self, pt_file):
        loaded = torch.load(pt_file)
        self.data = loaded['data']
        self.labels = loaded['labels']
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        val = int(self.data[idx])
        bits = [(val >> i) & 1 for i in range(63, -1, -1)]
        return torch.tensor(bits, dtype=torch.float32), self.labels[idx]

class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCELoss()
    def forward(self, pred, target):
        sm_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(pred, sm_target)

def train_refined(stage=1):
    # Stage 1: Bootstrap (5 Rounds), Stage 2: Attack (6 Rounds)
    DATA_FILE = "dataset_v3_bootstrap.pt" if stage == 1 else "dataset_v3.pt"
    BATCH_SIZE = 256
    EPOCHS = 15 if stage == 1 else 20
    MAX_LR = 0.002 if stage == 1 else 0.0005
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"--- Refinement Stage {stage} ---")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    full_dataset = DistinguisherDataset(DATA_FILE)
    n = len(full_dataset)
    train_ds, val_ds = random_split(full_dataset, [int(0.8*n), n-int(0.8*n)], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = EnhancedDistinguisher(d_model=256, nhead=8, num_layers=8).to(DEVICE)
    
    # Load weights if Stage 2 or Resume
    BEST_MODEL = f"best_model_v4_s{stage}.pth"
    PREV_STAGE_MODEL = "best_model_v4_s1.pth"
    
    if stage == 2:
        if os.path.exists(PREV_STAGE_MODEL):
            print(f"Loading bootstrap weights from {PREV_STAGE_MODEL}")
            model.load_state_dict(torch.load(PREV_STAGE_MODEL, map_location=DEVICE))
        else:
            print("Warning: Stage 1 weights not found. Starting Stage 2 from scratch.")
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR/10, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    criterion = LabelSmoothingBCE()
    
    best_acc = 0.0
    history = {'loss': [], 'acc': []}
    
    print(f"Training on {DEVICE} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"  Ep {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                targets = targets.to(DEVICE).unsqueeze(1)
                outputs = model(inputs.to(DEVICE))
                correct += ((outputs > 0.5) == targets).sum().item()
        
        acc = correct / len(val_ds)
        print(f"Epoch {epoch+1} Results: Loss={total_loss/len(train_loader):.4f}, Val Acc={acc:.4f}")
        history['loss'].append(total_loss/len(train_loader))
        history['acc'].append(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL)
            print(f"  -> Best model saved (Acc: {acc:.4f})")
            
    with open(f"history_v4_s{stage}.json", "w") as f:
        json.dump(history, f)
    print(f"Stage {stage} Complete. Best Acc: {best_acc:.4f}")

if __name__ == "__main__":
    # To run Stage 1: python train_v4.py --stage 1
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()
    train_refined(stage=args.stage)
