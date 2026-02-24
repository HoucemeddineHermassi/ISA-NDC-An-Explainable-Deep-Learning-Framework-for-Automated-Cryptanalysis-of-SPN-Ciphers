import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from models import Distinguisher
import numpy as np
import json

class DistinguisherDataset(Dataset):
    def __init__(self, pt_file):
        loaded = torch.load(pt_file)
        self.data = loaded['data']
        self.labels = loaded['labels']
        self.metadata = loaded['metadata']
        
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

def train_model():
    DATA_FILE = "dataset_kpa.pt"
    BATCH_SIZE = 128
    EPOCHS = 5
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    full_dataset = DistinguisherDataset(DATA_FILE)
    print(f"Loaded {len(full_dataset)} samples.")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Distinguisher().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            targets = targets.unsqueeze(1)
            
            noise = torch.randn_like(inputs) * 0.05
            noisy_inputs = inputs + noise
            
            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
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
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    torch.save(model.state_dict(), "distinguisher_model.pth")
    print("Model saved.")
    
    with open("training_history.json", "w") as f:
        json.dump(history, f)
    print("Training history saved to training_history.json")

if __name__ == "__main__":
    train_model()
