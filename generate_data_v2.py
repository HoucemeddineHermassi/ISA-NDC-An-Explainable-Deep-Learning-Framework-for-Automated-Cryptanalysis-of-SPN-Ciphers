"""
Enhanced Data Generation for ISA-NDC
- 5-round PRESENT target
- Multiple differentials
- 500 images -> ~100k samples
"""
import numpy as np
import torch
import random
from collections import Counter
from data_utils import load_images_from_folder, extract_blocks
from present import present_encrypt_block

def find_top_differentials(all_blocks, top_k=5, blocks_per_row=32):
    """Find top-K most frequent non-zero differentials in image blocks."""
    deltas = []
    
    # Horizontal differentials
    for i in range(len(all_blocks) - 1):
        if (i + 1) % blocks_per_row == 0:
            continue
        d = int(all_blocks[i]) ^ int(all_blocks[i+1])
        if d != 0:
            deltas.append(d)
            
    # Vertical differentials  
    for i in range(len(all_blocks) - blocks_per_row):
        d = int(all_blocks[i]) ^ int(all_blocks[i+blocks_per_row])
        if d != 0:
            deltas.append(d)
            
    c = Counter(deltas)
    top_deltas = [d for d, _ in c.most_common(top_k)]
    print(f"Top {top_k} differentials: {[hex(d) for d in top_deltas]}")
    return top_deltas

def generate_enhanced_dataset():
    """Generate large dataset with multiple differentials."""
    # Configuration
    IMAGE_DIR = r"e:\Anti-Gravity-Article\Attack-KPA-PRESENT\COVID-19_Radiography_Dataset\Normal\images"
    NUM_IMAGES = 500  # More images
    ROUNDS = 5  # Target 5 rounds
    TOP_K_DELTAS = 5  # Use top 5 differentials
    OUTPUT_FILE = "dataset_v2.pt"
    SAMPLES_PER_DELTA = 20000  # 20k per delta = 100k total real
    
    print(f"Loading {NUM_IMAGES} images...")
    images = load_images_from_folder(IMAGE_DIR, limit=NUM_IMAGES)
    print(f"Loaded {len(images)} images.")
    
    print("Extracting blocks...")
    all_blocks = []
    for img in images:
        all_blocks.extend(extract_blocks(img))
    all_blocks = np.array(all_blocks, dtype=np.uint64)
    print(f"Total blocks: {len(all_blocks)}")
    
    # Find top differentials
    top_deltas = find_top_differentials(all_blocks, top_k=TOP_K_DELTAS)
    
    # Generate key
    random.seed(42)
    key = random.getrandbits(80)
    print(f"Key: {key:020x}")
    
    X_data = []
    y_data = []
    delta_labels = []  # Track which delta each sample came from
    
    block_set = set(int(b) for b in all_blocks)
    
    for delta_idx, delta in enumerate(top_deltas):
        print(f"Processing delta {delta_idx+1}/{TOP_K_DELTAS}: {delta:016x}")
        
        # Find pairs with this delta
        pairs = []
        seen = set()
        for b in all_blocks:
            b_int = int(b)
            target = b_int ^ delta
            if target in block_set:
                pair = tuple(sorted((b_int, target)))
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
                    
        print(f"  Found {len(pairs)} pairs")
        
        # Sample up to SAMPLES_PER_DELTA
        if len(pairs) > SAMPLES_PER_DELTA:
            pairs = random.sample(pairs, SAMPLES_PER_DELTA)
        
        # Encrypt and collect
        for p1, p2 in pairs:
            c1 = present_encrypt_block(p1, key, rounds=ROUNDS)
            c2 = present_encrypt_block(p2, key, rounds=ROUNDS)
            diff_c = c1 ^ c2
            X_data.append(diff_c)
            y_data.append(1.0)
            delta_labels.append(delta_idx)
    
    num_real = len(X_data)
    print(f"Total real samples: {num_real}")
    
    # Generate random samples (equal to real)
    print("Generating random samples...")
    for _ in range(num_real):
        rand_diff = random.getrandbits(64)
        X_data.append(rand_diff)
        y_data.append(0.0)
        delta_labels.append(-1)  # -1 for random
        
    print(f"Total samples: {len(X_data)}")
    
    # Convert and save
    X_np = np.array(X_data, dtype=np.uint64).view(np.int64)
    
    dataset = {
        'data': torch.from_numpy(X_np),
        'labels': torch.tensor(y_data, dtype=torch.float32),
        'delta_labels': torch.tensor(delta_labels, dtype=torch.int64),
        'metadata': {
            'rounds': ROUNDS,
            'num_real': num_real,
            'top_deltas': [hex(d) for d in top_deltas],
            'key_hex': hex(key)
        }
    }
    
    torch.save(dataset, OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")
    
    return dataset['metadata']

if __name__ == "__main__":
    meta = generate_enhanced_dataset()
    print("\nDataset Metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
