"""
Enhanced Key Recovery Attack for 6-round PRESENT
Uses 5-round distinguisher to rank key candidates.
"""
import torch
import numpy as np
import random
import json
from models_v2 import LightDistinguisher
from present import present_encrypt_block, inv_p_layer, inv_sbox_layer

def attack_v2():
    MODEL_PATH = "best_model_v2.pth"
    DISTINGUISHER_ROUNDS = 5  # Trained on 5 rounds
    ATTACK_ROUNDS = 6  # Attack 6 rounds
    NUM_PAIRS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = LightDistinguisher(d_model=128, nhead=4, num_layers=4).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"{MODEL_PATH} not found. Run train_v2.py first.")
        return
    model.eval()
    print("Loaded model.")
    
    # Target Setup
    TRUE_KEY = 0x1234567890ABCDEF1234
    TARGET_DELTA = 0x0100000000000000  # Use primary delta
    
    print(f"Target Key: {TRUE_KEY:020x}")
    print(f"Generating {NUM_PAIRS} pairs encrypted with {ATTACK_ROUNDS} rounds...")
    
    # Generate Round Keys
    from present import generate_round_keys
    rks = generate_round_keys(TRUE_KEY, rounds=ATTACK_ROUNDS)
    TRUE_K_LAST = rks[ATTACK_ROUNDS]  # Post-whitening key
    
    # Generate pairs
    pairs = []
    random.seed(123)
    for _ in range(NUM_PAIRS):
        p1 = random.getrandbits(64)
        p2 = p1 ^ TARGET_DELTA
        c1 = present_encrypt_block(p1, TRUE_KEY, rounds=ATTACK_ROUNDS)
        c2 = present_encrypt_block(p2, TRUE_KEY, rounds=ATTACK_ROUNDS)
        pairs.append((c1, c2))
    
    # Key Candidates
    candidates = []
    candidates.append((TRUE_K_LAST, "True Key"))
    
    # Random keys
    for i in range(20):
        candidates.append((random.getrandbits(64), f"Random Key {i}"))
    
    # Near-miss keys (flip 1-2 bits)
    for i in range(8):
        k_near = TRUE_K_LAST ^ (1 << i)
        candidates.append((k_near, f"Near Key (bit {i})"))
    
    print(f"Testing {len(candidates)} candidates...")
    
    results = []
    
    def to_bits(diff):
        b_val = diff.to_bytes(8, 'big', signed=False)
        bits = []
        for b in b_val:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
        return bits
    
    for cand_key, label in candidates:
        scores = []
        for c1, c2 in pairs:
            # Decrypt 1 round
            s1 = c1 ^ cand_key
            s2 = c2 ^ cand_key
            s1 = inv_p_layer(s1)
            s2 = inv_p_layer(s2)
            s1 = inv_sbox_layer(s1)
            s2 = inv_sbox_layer(s2)
            
            diff = s1 ^ s2
            bits = to_bits(diff)
            t_in = torch.tensor(bits, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                prob = model(t_in).item()
                scores.append(prob)
                
        avg_score = sum(scores) / len(scores)
        results.append((label, avg_score))
        
    # Sort by score (higher = more likely real)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Find True Key rank
    true_rank = next(i for i, (l, _) in enumerate(results) if "True" in l) + 1
    
    print(f"\n{'='*50}")
    print(f"RESULTS: True Key Rank = {true_rank}/{len(candidates)}")
    print(f"{'='*50}")
    print("\nTop 10 Candidates:")
    for i, (l, s) in enumerate(results[:10]):
        marker = " <-- TRUE KEY" if "True" in l else ""
        print(f"{i+1}. {l}: {s:.4f}{marker}")
    
    # Save results
    with open("attack_results_v2.json", "w") as f:
        json.dump({'results': results, 'true_key_rank': true_rank}, f)
    print("\nSaved to attack_results_v2.json")

if __name__ == "__main__":
    attack_v2()
