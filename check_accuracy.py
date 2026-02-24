"""
Check current accuracy of the enhanced model during training.
"""
import torch
from torch.utils.data import DataLoader, random_split
from models_v2 import LightDistinguisher
from train_v2 import DistinguisherDataset
import os

def check_current_accuracy():
    DATA_FILE = "dataset_v2.pt"
    MODEL_PATH = "best_model_v2.pth"
    BATCH_SIZE = 256
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found.")
        return

    # Load Data
    full_dataset = DistinguisherDataset(DATA_FILE)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = LightDistinguisher(d_model=128, nhead=4, num_layers=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
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
    print(f"Current Validation Accuracy (from {MODEL_PATH}): {val_acc:.4f}")

if __name__ == "__main__":
    check_current_accuracy()
