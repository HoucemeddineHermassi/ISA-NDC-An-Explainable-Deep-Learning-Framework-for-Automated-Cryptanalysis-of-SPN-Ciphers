"""
Training script v3: 6-round target using full EnhancedDistinguisher.
Publication-grade experiments.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
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

def train_publication_scale():
    DATA_FILE = "dataset_v3.pt"
    BATCH_SIZE = 256
    EPOCHS = 20
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(DATA_FILE):
        print(f"Waiting for {DATA_FILE}...")
        return

    full_dataset = DistinguisherDataset(DATA_FILE)
    train_size = int(0.8 * len(full_dataset))
    train_dataset, val_dataset = random_split(full_dataset, [train_size, len(full_dataset)-train_size], 
                                              generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model: Standard Publication Architecture
    model = EnhancedDistinguisher(d_model=256, nhead=8, num_layers=8, dropout=0.1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = LabelSmoothingBCE()
    
    print(f"Starting training on {DEVICE}...")
    start_epoch = 0
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}

    # Resume logic
    CHECKPOINT_FILE = "checkpoint_v3.pth"
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Resuming from {CHECKPOINT_FILE}...")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        best_acc = checkpoint['best_acc']
        # Forward scheduler to current epoch
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed at epoch {start_epoch + 1}")
    else:
        print("Starting fresh training.")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0: print(f"  Ep {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                targets = targets.to(DEVICE).unsqueeze(1)
                outputs = model(inputs.to(DEVICE))
                correct += ((outputs > 0.5) == targets).sum().item()
        
        acc = correct / len(val_dataset)
        history['train_loss'].append(total_loss/len(train_loader))
        history['val_acc'].append(acc)
        print(f"Epoch {epoch+1} Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model_v3.pth")
            print("  -> Best model saved.")
            
        # Checkpoint for resumption
        checkpoint = {
            'epoch': epoch, 'model': model.state_dict(), 'opt': optimizer.state_dict(),
            'history': history, 'best_acc': best_acc
        }
        torch.save(checkpoint, "checkpoint_v3.pth")
        scheduler.step()

    with open("history_v3.json", "w") as f: json.dump(history, f)
    print("Training finished.")

if __name__ == "__main__":
    train_publication_scale()
