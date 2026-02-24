"""
Data Generation for Round 6 (Publication Target)
Generates 2M samples (1M real / 1M random) using multiple keys.
Optimized for 6-round signal detection.
"""
import numpy as np
import torch
import random
import os
import sys
from collections import Counter

# Ensure imports work relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_utils import load_images_from_folder, extract_blocks
from present import present_encrypt_block

def find_top_differentials(all_blocks, top_k=5, blocks_per_row=32):
    """Identify most common plaintext differences in image data."""
    deltas = []
    # Horizontal differences
    for i in range(len(all_blocks) - 1):
        if (i + 1) % blocks_per_row == 0: continue
        d = int(all_blocks[i]) ^ int(all_blocks[i+1])
        if d != 0: deltas.append(d)
    
    # Vertical differences
    for i in range(len(all_blocks) - blocks_per_row):
        d = int(all_blocks[i]) ^ int(all_blocks[i+blocks_per_row])
        if d != 0: deltas.append(d)
        
    c = Counter(deltas)
    return [d for d, _ in c.most_common(top_k)]

def generate_r6_dataset():
    # Configuration
    # Adjust path to point to the dataset root relative to this script or absolute
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) # Adjust as needed
    # Using absolute path from previous context to be safe
    IMAGE_DIR = r"e:\Anti-Gravity-Article\Attack-KPA-PRESENT\COVID-19_Radiography_Dataset\Normal\images"
    
    OUTPUT_FILE = r"e:\Anti-Gravity-Article\Attack-KPA-PRESENT\data\isa_ndc_r6.npy"
    
    ROUNDS = 6
    NUM_IMAGES = 1000
    SAMPLES_PER_DELTA = 200_000
    NUM_KEYS = 50 
    
    print(f"--- Generating 6-Round Dataset ---")
    print(f"Target: {OUTPUT_FILE}")
    print(f"Rounds: {ROUNDS}")
    print(f"Total Samples: {SAMPLES_PER_DELTA * 5 * 2} (Real + Random)")
    
    # 1. Load Image Data
    print(f"Loading images from {IMAGE_DIR}...")
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory {IMAGE_DIR} not found!")
        return

    images = load_images_from_folder(IMAGE_DIR, limit=NUM_IMAGES)
    all_blocks = []
    for img in images: 
        all_blocks.extend(extract_blocks(img))
    
    print(f"Extracted {len(all_blocks)} blocks.")
    all_blocks = np.array(all_blocks, dtype=np.uint64)
    
    # 2. Find Top Differentials
    top_deltas = find_top_differentials(all_blocks, top_k=5)
    print(f"Top Deltas: {[hex(d) for d in top_deltas]}")
    
    X_data = []
    y_data = []
    
    # 3. Generate Data
    # We use multiple keys to prevent overfitting to a single key
    keys = [random.getrandbits(80) for _ in range(NUM_KEYS)]
    
    block_set = set(int(b) for b in all_blocks)
    
    for delta_idx, delta in enumerate(top_deltas):
        print(f"Processing Delta {delta_idx+1}/{len(top_deltas)}: {hex(delta)}")
        
        # Find pairs with this delta
        pairs = []
        seen = set() # Avoid duplicates
        
        # We need SAMPLES_PER_DELTA pairs. 
        # Scan blocks until we have enough or run out
        for b in all_blocks:
            b_int = int(b)
            target = b_int ^ delta
            if target in block_set:
                pair = tuple(sorted((b_int, target)))
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
            
            if len(pairs) >= SAMPLES_PER_DELTA:
                break
        
        # If not enough pairs, we might need to recycle or scramble
        if len(pairs) < SAMPLES_PER_DELTA:
            print(f"Warning: Only found {len(pairs)} natural pairs for {hex(delta)}. Supplementing with random pairs.")
            while len(pairs) < SAMPLES_PER_DELTA:
                p1 = random.getrandbits(64)
                p2 = p1 ^ delta
                pairs.append((p1, p2))
        
        # Limit to exact count
        pairs = pairs[:SAMPLES_PER_DELTA]
        
        # Encrypt pairs
        # Distribute pairs across keys
        print(f"  Encrypting {len(pairs)} pairs...")
        for i, (p1, p2) in enumerate(pairs):
            k = keys[i % NUM_KEYS]
            c1 = present_encrypt_block(p1, k, rounds=ROUNDS)
            c2 = present_encrypt_block(p2, k, rounds=ROUNDS)
            X_data.append(c1 ^ c2)
            y_data.append(1.0) # Real
            
    # 4. Generate Random Data (Label 0)
    num_real = len(X_data)
    print(f"Generated {num_real} real samples. Generating matching random samples...")
    
    # Random ciphertext differences are just random 64-bit integers
    # Ideally, they should follow the distribution of random block differences, which is uniform
    for _ in range(num_real):
        X_data.append(random.getrandbits(64))
        y_data.append(0.0) # Random
        
    # 5. Save
    print("Saving to disk...")
    X_np = np.array(X_data, dtype=np.uint64)
    y_np = np.array(y_data, dtype=np.float32)
    
    # Save as numpy dictionary for consistency with train_isa_ndc.py
    data_dict = {
        'X': X_np,
        'Y': y_np,
        'metadata': {
            'rounds': ROUNDS,
            'deltas': top_deltas,
            'num_keys': NUM_KEYS
        }
    }
    
    np.save(OUTPUT_FILE, data_dict)
    print(f"Done. Saved {len(X_np)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_r6_dataset()
