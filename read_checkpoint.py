import torch
checkpoint = torch.load("checkpoint.pth", map_location='cpu', weights_only=False)
print(f"Epoch: {checkpoint['epoch']}")
print(f"Best Val Acc: {checkpoint['best_val_acc']:.4f}")
