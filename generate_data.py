import numpy as np
import os
import torch
import random
from data_utils import load_images_from_folder, extract_blocks
from present import present_encrypt_block

def generate_dataset():
    # Configuration
    IMAGE_DIR = r"e:\Anti-Gravity-Article\Attack-KPA-PRESENT\COVID-19_Radiography_Dataset\Normal\images"
    TARGET_DELTA = 0x0100000000000000
    NUM_IMAGES = 200
    OUTPUT_FILE = "dataset_kpa.pt"
    ROUNDS = 6 # Reduced round for initial attack/training
    KEY = 0xFFFFFFFFFFFFFFFFFFFF # Fixed undisclosed key for the "Scenario"
    # Actually, in KPA, we find pairs in the wild.
    # We simulate this by encrypting the found plaintext pairs.
    
    print(f"Loading {NUM_IMAGES} images...")
    images = load_images_from_folder(IMAGE_DIR, limit=NUM_IMAGES)
    
    print("Extracting blocks...")
    all_blocks = []
    for img in images:
        all_blocks.extend(extract_blocks(img))
    all_blocks = np.array(all_blocks, dtype=np.uint64)
    
    print(f"Total blocks: {len(all_blocks)}")
    
    # Find pairs with difference = TARGET_DELTA
    # To do this efficiently:
    # We want P_i ^ P_j = D. => P_j = P_i ^ D.
    # Put all P in a set/hashmap.
    # For each P, check if P ^ D exists.
    
    block_set = set(all_blocks)
    pairs = []
    
    seen = set()
    
    print("Finding pairs...")
    for b in all_blocks:
        target = b ^ TARGET_DELTA
        if target in block_set:
            # Found a pair (b, target)
            # Avoid duplicates (pair (a,b) and (b,a))
            pair = tuple(sorted((b, target)))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                
    print(f"Found {len(pairs)} Unique Real Pairs with Delta {TARGET_DELTA:x}")
    
    # Encrypt pairs
    print(f"Encrypting pairs with {ROUNDS} rounds...")
    X_data = [] # Delta C
    y_data = [] # Label
    
    # We use a random key for the simulation, or fixed?
    # Key should be fixed for the "Intercepted" data.
    random.seed(42)
    key = random.getrandbits(80) 
    
    for p1, p2 in pairs:
        # P1, P2 are uint64
        # Encrypt
        c1 = present_encrypt_block(int(p1), key, rounds=ROUNDS)
        c2 = present_encrypt_block(int(p2), key, rounds=ROUNDS)
        
        diff_c = c1 ^ c2
        X_data.append(diff_c)
        y_data.append(1.0)
        
    # Generate Random Data (Label 0)
    # Generate random 64-bit diffs
    # Same amount as real data
    print("Generating random samples...")
    for _ in range(len(pairs)):
        rand_diff = random.getrandbits(64)
        X_data.append(rand_diff)
        y_data.append(0.0)
        
    print(f"Total samples: {len(X_data)}")
    
    # Save
    # overflow fix: torch.tensor(int list) tries to fit in int64 signed.
    # We strip to numpy uint64 first, then view as int64 (reinterpret bits)
    X_np_uint64 = np.array(X_data, dtype=np.uint64)
    X_np_int64 = X_np_uint64.view(np.int64)
    
    dataset = {
        'data': torch.from_numpy(X_np_int64), 
        'labels': torch.tensor(y_data, dtype=torch.float32),
        'metadata': {
            'delta_p': TARGET_DELTA,
            'rounds': ROUNDS,
            'num_real': len(pairs)
        }
    }
    
    torch.save(dataset, OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()
