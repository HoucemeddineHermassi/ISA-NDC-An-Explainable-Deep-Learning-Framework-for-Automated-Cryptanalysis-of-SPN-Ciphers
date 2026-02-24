import numpy as np
import os
from data_utils import load_images_from_folder, extract_blocks
from collections import Counter

def analyze_deltas():
    image_dir = r"e:\Anti-Gravity-Article\Attack-KPA-PRESENT\COVID-19_Radiography_Dataset\Normal\images"
    # Load just 5 images for quick analysis
    print("Loading images...")
    images = load_images_from_folder(image_dir, limit=5)
    print(f"Loaded {len(images)} images.")
    
    all_blocks = []
    for img in images:
        blocks = extract_blocks(img)
        all_blocks.extend(blocks)
        
    all_blocks = np.array(all_blocks, dtype=np.uint64)
    print(f"Total blocks: {len(all_blocks)}")
    
    # We want to find Delta P = P_i XOR P_j that occurs frequently.
    # Naive O(N^2) is too slow if N is large.
    # N ~ 5 * (256*256/8) ~ 5 * 8192 ~ 40,000 blocks.
    # 40k^2 = 1.6 billion pairs. Too slow.
    
    # Instead, we look for LOCAL differentials (spatial correlation).
    # Compare each block with its neighbors (Right, Down).
    # Since we flattened row by row:
    # Right neighbor: index i + 1
    # Down neighbor: index i + (Width/8)? No.
    # Block width is 8 bytes = 64 bits?
    # Actually, we processed 8 BYTES linear as a block.
    # So P_i is bytes [0..7], P_{i+1} is bytes [8..15].
    # In a 256-width image, row has 256 bytes = 32 blocks.
    # So Down neighbor is i + 32.
    
    deltas = []
    
    # Horizontal differentials (adjacent blocks in row)
    # We expect smooth changes, so P_i XOR P_{i+1} might be small/sparse.
    for i in range(len(all_blocks) - 1):
        # Skip if wrapping around row?
        # 32 blocks per row.
        if (i + 1) % 32 == 0: continue
        
        d = all_blocks[i] ^ all_blocks[i+1]
        if d != 0: # Ignore identical blocks (pure black background etc)
            deltas.append(d)
            
    # Vertical differentials
    blocks_per_row = 32
    for i in range(len(all_blocks) - blocks_per_row):
        d = all_blocks[i] ^ all_blocks[i+blocks_per_row]
        if d != 0:
            deltas.append(d)
            
    # Count frequencies
    c = Counter(deltas)
    print("Top 20 most frequent non-zero differentials:")
    for d, count in c.most_common(20):
        print(f"Delta: {d:016x}, Count: {count}")
        
    return c.most_common(1)[0][0] if c else None

if __name__ == "__main__":
    best_delta = analyze_deltas()
    if best_delta:
        print(f"Recommended Delta: {best_delta:016x}")
