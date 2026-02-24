"""
7-Round Key Recovery Attack
Uses 6-round EnhancedDistinguisher6R to attack 7-round PRESENT.
"""
import torch
import numpy as np
import random
import json
import os
import sys

# Add path for imports
sys.path.append(os.path.join(os.getcwd(), 'git repos', 'ISA-NDC'))

from models_v2 import EnhancedDistinguisher6R
from present import present_encrypt_block, inv_p_layer, inv_sbox_layer, generate_round_keys

def attack_r7():
    MODEL_PATH = "models/isa_ndc_r6_best.pth"
    DISTINGUISHER_ROUNDS = 6
    ATTACK_ROUNDS = 7
    NUM_PAIRS = 500  # Increased from 100 for stronger signal
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"--- 7-Round Key Recovery Attack ---")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model {MODEL_PATH} not found. Run train_isa_ndc.py first.")
        return
        
    model = EnhancedDistinguisher6R().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")
    
    # 2. Setup Target
    TRUE_KEY = 0x1234567890ABCDEF1234 # Master Key
    TARGET_DELTA = 0x0100000000000000
    
    # Derive round keys
    rks = generate_round_keys(TRUE_KEY, rounds=ATTACK_ROUNDS)
    TRUE_K_LAST = rks[ATTACK_ROUNDS] # K_7 for 7-round encryption
    
    print(f"True Last Round Key (K_7): {hex(TRUE_K_LAST)}")
    
    # 3. Generate Ciphertext Pairs
    print(f"Generating {NUM_PAIRS} pairs encrypted with {ATTACK_ROUNDS} rounds...")
    pairs = []
    random.seed(42) # Fixed seed for reproducibility
    for _ in range(NUM_PAIRS):
        p1 = random.getrandbits(64)
        p2 = p1 ^ TARGET_DELTA
        c1 = present_encrypt_block(p1, TRUE_KEY, rounds=ATTACK_ROUNDS)
        c2 = present_encrypt_block(p2, TRUE_KEY, rounds=ATTACK_ROUNDS)
        pairs.append((c1, c2))
        
    # 4. Preparing Candidates
    # In a real attack, we would search 2^64 space.
    # Here verification implies ranking True Key vs Random Keys.
    # We test: True Key, 20 Random Keys, 8 Near Misses.
    candidates = []
    candidates.append((TRUE_K_LAST, "True Key"))
    
    for i in range(50):
        candidates.append((random.getrandbits(64), f"Random Key {i}"))
        
    for i in range(10):
        k_near = TRUE_K_LAST ^ (1 << i)
        candidates.append((k_near, f"Near Key (bit {i})"))
        
    print(f"Evaluating {len(candidates)} candidates...")
    
    # helper
    def to_bits_batch(diffs):
        # diffs: list of ints
        # return: tensor (B, 64)
        arr = np.array(diffs, dtype=np.uint64).view(np.uint8)
        bits = np.unpackbits(arr).reshape(-1, 8, 8)
        bits = bits.reshape(-1, 64).astype(np.float32)
        return torch.from_numpy(bits)

    results = []
    
    # Optimization: Process pairs in batches if possible
    # But here we loop candidates, then pairs.
    
    for cand_key, label in candidates:
        # Decrypt last round for ALL pairs with this candidate key
        decrypted_diffs = []
        for c1, c2 in pairs:
            # Decrypt 1 round (Round 7 -> Round 6 output)
            # InvPostWhitening (built into round function usually, but here assumed specific structure)
            # Standard SPN last round: XOR K -> S-layer -> P-layer -> XOR K_next (usually no P in last round?)
            # Wait, PRESENT structure:
            # Round i: AddRoundKey -> SBox -> PLayer
            # Round 31 (last): AddRoundKey -> SBox -> AddRoundKey
            # But we are simulating R rounds.
            # My `present.py` `present_encrypt_block` uses standard rounds.
            # If rounds=7, it does 7 full rounds (RK -> S -> P).
            # So decryption is: InvP -> InvS -> XOR RK
            
            s1 = inv_p_layer(c1)
            s2 = inv_p_layer(c2)
            
            s1 = inv_sbox_layer(s1)
            s2 = inv_sbox_layer(s2)
            
            x1 = s1 ^ cand_key
            x2 = s2 ^ cand_key
            
            # Now we have Output of Round 6?
            # No, `present_encrypt_block(rounds=7)` output is C7.
            # We want C6_diff.
            # C7 = P(S(C6 ^ K7))
            # So C6 = InvS(InvP(C7)) ^ K7
            # My logic in loop:
            # s1 = inv_p_layer(c1) -> InvP(C7)
            # s1 = inv_sbox_layer(s1) -> InvS(InvP(C7))
            # x1 = s1 ^ cand_key -> C6 estimate
            
            decrypted_diffs.append(x1 ^ x2)
            
        # Batch inference
        t_in = to_bits_batch(decrypted_diffs).to(DEVICE)
        
        with torch.no_grad():
            scores = model(t_in).squeeze().tolist()
            if isinstance(scores, float): scores = [scores]
            
        # Log-likelihood
        # prob p. log(p / (1-p)) or just sum(log(p))
        # We use sum(log(p)) for "Real" class.
        # Avoid log(0)
        epsilon = 1e-10
        total_score = sum([np.log(p + epsilon) for p in scores])
        
        results.append((label, total_score))
        
    # Sort
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Rank
    true_rank = next(i for i, (l, _) in enumerate(results) if "True Key" in l) + 1
    
    print(f"\nResults:")
    print(f"True Key Rank: {true_rank} / {len(candidates)}")
    print("Top 5:")
    for i, (l, s) in enumerate(results[:5]):
         print(f"{i+1}. {l}: {s:.2f}")
         
    # Save
    with open("attack_r7_results.json", "w") as f:
        json.dump({'results': results, 'true_rank': true_rank}, f)

if __name__ == "__main__":
    attack_r7()
