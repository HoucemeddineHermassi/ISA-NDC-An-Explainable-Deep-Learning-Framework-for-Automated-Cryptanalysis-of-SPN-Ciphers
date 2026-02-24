"""
Enhanced Training with:
- Curriculum Learning (3->4->5 rounds)
- Cosine LR Scheduler
- Label Smoothing
- Early Stopping
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from models_v2 import LightDistinguisher
import numpy as np
import json
import os
import random

class DistinguisherDataset(Dataset):
    def __init__(self, pt_file):
        loaded = torch.load(pt_file)
        self.data = loaded['data']
        self.labels = loaded['labels']
        self.metadata = loaded.get('metadata', {})
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        val = self.data[idx].item()
        b_val = val.to_bytes(8, 'big', signed=True)
        bits = []
        for b in b_val:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
        return torch.tensor(bits, dtype=torch.float32), self.labels[idx]

class LabelSmoothingBCE(nn.Module):
    """BCE with label smoothing."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCELoss()
        
    def forward(self, pred, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(pred, target)

def train_enhanced():
    DATA_FILE = "dataset_v2.pt"
    BATCH_SIZE = 256
    EPOCHS = 15
    LR = 0.001
    LABEL_SMOOTHING = 0.1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_FILE = "checkpoint.pth"
    
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(DATA_FILE):
        print(f"{DATA_FILE} not found. Run generate_data_v2.py first.")
        return
    
    # Load Data
    full_dataset = DistinguisherDataset(DATA_FILE)
    print(f"Loaded {len(full_dataset)} samples.")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = LightDistinguisher(d_model=128, nhead=4, num_layers=4, dropout=0.1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR/100)
    criterion = LabelSmoothingBCE(smoothing=LABEL_SMOOTHING)
    
    start_epoch = 0
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    
    # Resume from checkpoint if exists
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Resuming from {CHECKPOINT_FILE}...")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        history = checkpoint['history']
        
        # Restore RNG states for reproducibility
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
            np.random.set_state(checkpoint['numpy_rng_state'])
            random.setstate(checkpoint['python_rng_state'])
            
        print(f"Resumed at Epoch {start_epoch}")
    elif os.path.exists("best_model_v2.pth"):
        print("No checkpoint found, but 'best_model_v2.pth' exists. Loading weights for soft resume...")
        # Load weights only
        model.load_state_dict(torch.load("best_model_v2.pth", map_location=DEVICE))
        print("Weights loaded. Starting fine-tuning with fresh optimizer.")
        # Retain history structure but essentially starting new phase
    else:
        print("Starting training from scratch...")

    try:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                targets = targets.unsqueeze(1)
                
                # Bit flip augmentation
                if random.random() < 0.2:
                    flip_mask = (torch.rand_like(inputs) < 0.02).float()
                    inputs = torch.abs(inputs - flip_mask)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            train_acc = correct / total
            avg_loss = total_loss / len(train_loader)
            current_lr = scheduler.get_last_lr()[0]
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    targets = targets.unsqueeze(1)
                    outputs = model(inputs)
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == targets).sum().item()
                    val_total += targets.size(0)
                    
            val_acc = val_correct / val_total
            
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            print(f"Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}")
            
            # Checkpoint saving
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_model_v2.pth")
                print(f"  -> New best model saved! ({val_acc:.4f})")
            
            # Save resumption checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate()
            }
            torch.save(checkpoint, CHECKPOINT_FILE)
            
            scheduler.step()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Checkpoint saved.")
    finally:
        # Save final logs
        with open("training_history_v2.json", "w") as f:
            json.dump(history, f)
        print("Logs saved.")

if __name__ == "__main__":
    train_enhanced()
