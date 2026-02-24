"""
Data Generation v3: 6-round target, 200k samples.
Used for publication-grade experiments.
"""
import numpy as np
import torch
import random
from collections import Counter
from data_utils import load_images_from_folder, extract_blocks
from present import present_encrypt_block

def find_top_differentials(all_blocks, top_k=5, blocks_per_row=32):
    deltas = []
    for i in range(len(all_blocks) - 1):
        if (i + 1) % blocks_per_row == 0: continue
        d = int(all_blocks[i]) ^ int(all_blocks[i+1])
        if d != 0: deltas.append(d)
    for i in range(len(all_blocks) - blocks_per_row):
        d = int(all_blocks[i]) ^ int(all_blocks[i+blocks_per_row])
        if d != 0: deltas.append(d)
    c = Counter(deltas)
    return [d for d, _ in c.most_common(top_k)]

def generate_v3_dataset():
    IMAGE_DIR = r"e:\Anti-Gravity-Article\Attack-KPA-PRESENT\COVID-19_Radiography_Dataset\Normal\images"
    NUM_IMAGES = 500
    ROUNDS = 5  # Bootstrap stage uses 5 rounds
    TOP_K_DELTAS = 5
    OUTPUT_FILE = "dataset_v3_bootstrap.pt"
    SAMPLES_PER_DELTA = 20000 # 20k * 5 = 100k real + 100k random = 200k total
    
    print(f"Generating 6-round dataset ({OUTPUT_FILE})...")
    images = load_images_from_folder(IMAGE_DIR, limit=NUM_IMAGES)
    all_blocks = []
    for img in images: all_blocks.extend(extract_blocks(img))
    all_blocks = np.array(all_blocks, dtype=np.uint64)
    
    top_deltas = find_top_differentials(all_blocks, top_k=TOP_K_DELTAS)
    random.seed(42)
    key = random.getrandbits(80)
    
    X_data = []
    y_data = []
    block_set = set(int(b) for b in all_blocks)
    
    for delta in top_deltas:
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
        if len(pairs) > SAMPLES_PER_DELTA:
            pairs = random.sample(pairs, SAMPLES_PER_DELTA)
        print(f"Delta {hex(delta)}: {len(pairs)} pairs")
        for p1, p2 in pairs:
            c1 = present_encrypt_block(p1, key, rounds=ROUNDS)
            c2 = present_encrypt_block(p2, key, rounds=ROUNDS)
            X_data.append(c1 ^ c2)
            y_data.append(1.0)
            
    num_real = len(X_data)
    print(f"Real samples: {num_real}. Generating random data...")
    for _ in range(num_real):
        X_data.append(random.getrandbits(64))
        y_data.append(0.0)
        
    X_np = np.array(X_data, dtype=np.uint64).view(np.int64)
    dataset = {
        'data': torch.from_numpy(X_np),
        'labels': torch.tensor(y_data, dtype=torch.float32),
        'metadata': {'rounds': ROUNDS, 'num_samples': len(X_data), 'key': hex(key)}
    }
    torch.save(dataset, OUTPUT_FILE)
    print(f"Success. Total: {len(X_data)} samples saved.")

if __name__ == "__main__":
    generate_v3_dataset()
