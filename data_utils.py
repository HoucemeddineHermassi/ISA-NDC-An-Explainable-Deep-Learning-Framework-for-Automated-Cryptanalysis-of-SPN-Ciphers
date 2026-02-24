import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, limit=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): Limit the number of images to load.
        """
        self.image_paths = glob.glob(os.path.join(image_dir, "*.png")) + \
                           glob.glob(os.path.join(image_dir, "*.jpg"))
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert('L') # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return np.array(image, dtype=np.uint8)

def load_images_from_folder(folder, size=(256, 256), limit=10):
    images = []
    paths = glob.glob(os.path.join(folder, "*.png"))
    if limit:
        paths = paths[:limit]
        
    for path in paths:
        try:
            img = Image.open(path).convert('L')
            img = img.resize(size)
            images.append(np.array(img, dtype=np.uint8))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return np.array(images)

def extract_blocks(image_np):
    """
    Flatten image to 64-bit blocks.
    image_np: 2D numpy array (uint8)
    returns: 1D numpy array of uint64
    """
    flat = image_np.flatten()
    # Pad to 8 bytes
    pad_len = (8 - (len(flat) % 8)) % 8
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), 'constant')
    
    # View as 64-bit blocks
    # Ensure Big Endian interpretation matches our cipher if needed, 
    # but for differential analysis consistency relative to itself is key.
    # We used 'big' in present.py. 
    # np.view uses system endianness (usually little on x86).
    # To be safe and consistent with present.py which does manual bytes->int:
    
    # Faster approach for arrays:
    # 1. View as uint64
    # 2. Byteswap if system is little endian to match 'big' expected by cipher if we care about specific bit position meanings (like P-layer).
    # PRESENT is defined on bits. MSB is bit 63.
    # If we map byte 0 to bits 63..56, that is Big Endian.
    # x86 stores byte 0 at lowest address. 
    # If we just view as uint64 on Little Endian: Byte 0 becomes LSB (bits 7..0).
    # So we MUST Swap bytes for consistency if we want image topological structure to map roughly to high bits? 
    # Actually, for Differential Cryptanalysis, as long as we use the SAME transformation for Encryption and Analysis, it holds.
    # HOWEVER, the 'ISA' part relies on Image Structure (gradients). 
    # Gradients are between adjacent pixels.
    # Pixels are bytes.
    # Block 1: Pixels P1, P2, P3, P4, P5, P6, P7, P8.
    # If we map this to 64-bit int:
    # Big Endian: P1 is top byte, P8 is bottom byte.
    # Little Endian: P1 is bottom byte, P8 is top byte.
    # We should stick to Big Endian so P1 (leftmost pixel in block) is MSB part.
    
    blocks = flat.view(dtype=np.uint64)
    # Check system endianness
    if list(np.array([1], dtype=np.uint64).tobytes())[0] == 1:
        # Little Endian system
        blocks = blocks.byteswap()
        
    return blocks

class PresentationDifferentialDataset(Dataset):
    def __init__(self, real_diffs, random_diffs):
        """
        real_diffs: list of (input_diff, output_diff) where output_diff is form C1 XOR C2
        random_diffs: list of (input_diff, random_diff)
        """
        # We model this as: Input = Diff_C, Target = 1 (Real) or 0 (Random)
        # Wait, the Distinguisher usually takes Partial Decrypted Difference as input?
        # Or does it take just Output Difference?
        # The Proposal says: "Train a neural distinguisher on differentials ... Distinguisher inputs \Delta_C"
        # And "Label=1 for real, 0 for random".
        # Yes, standard distinguisher: Given \Delta C, is it from a pair with \Delta P = fixed?
        # In ISA-NDC, \Delta P is NOT fixed globally, but chosen per image block?
        # "Identify optimal differential inputs (\Delta) based on known plaintext ... pairs where \Delta P matches 'good' differentials".
        # So we collect pairs (P1, P2) such that P1 ^ P2 = Delta_Target.
        # Then we feed C1 ^ C2 to distinguisher.
        # So the Distinguisher is conditioned on a specific Delta_P? Or a set of good ones?
        # Usually Distinguishers are for a SPECIFIC Delta_P -> Delta_C_Prop.
        # If we have multiple good Delta_Ps, we might need multiple Distinguishers or a conditioned one.
        # The proposal says: "Predict top-K \Delta s ... Train a neural distinguisher on differentials derived...".
        # Implicitly, it seems we might focus on ONE best Delta first for the Proof of Concept.
        # Or a class of Deltas.
        
        # Let's assume for PoC we pick ONE high probability Delta found in images, e.g. 0 (identical) or low Hamming weight.
        # Actually P1=P2 is trivial (0->0).
        # We want non-zero differences.
        # Images have smooth gradients -> P_i and P_{i+1} (adjacent blocks) might differ slightly?
        # Or P_i and P_j from same region.
        
        # Let's flatten and collect inputs.
        self.data = []
        self.labels = []
        
        # Real
        for rd in real_diffs:
            self.data.append(rd) # 64-bit int
            self.labels.append(1.0)
            
        # Random
        for rand in random_diffs:
            self.data.append(rand)
            self.labels.append(0.0)
            
        self.data = np.array(self.data, dtype=np.uint64)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return as byte tensor for CNN/ViT?
        # The model expects 8x8 input presumably?
        val = self.data[idx]
        # Convert uint64 to 8 bytes (Big Endian)
        b = int(val).to_bytes(8, 'big')
        # Convert to float array [0,1] or just raw bytes
        # Model input: 1x8x8? 
        # 64 bits = 8x8 bits? Or 8 bytes?
        # Proposal: "Input \Delta_C as 1x8x8 'image' (reshape 64 bits)"
        # This implies BIT LEVEL input? 64 bits.
        # Or Byte level? "1x8x8" usually implies 8x8 pixels. If bits, it's 64 pixels.
        # If bytes, it's 8 pixels? No 8 bytes is not 8x8.
        # Clarification: "reshape 64 bits" -> 8x8 binary grid.
        
        # Let's implement conversion to 8x8 bit grid.
        bits = [(val >> i) & 1 for i in range(63, -1, -1)] # Big Endian bits
        tensor = torch.tensor(bits, dtype=torch.float32).reshape(1, 8, 8)
        
        return tensor, torch.tensor(self.labels[idx])

