import torch
ckpt = torch.load("checkpoint_v3.pth", map_location='cpu', weights_only=False)
print(f"Epoch: {ckpt['epoch'] + 1}")
print(f"Best Val Acc: {ckpt['best_acc']:.4f}")
