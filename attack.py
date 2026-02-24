import torch
import numpy as np
import random
from models import Distinguisher
from present import present_encrypt_block, present_decrypt_block, add_round_key, p_layer, sbox_layer, inv_p_layer, inv_sbox_layer
# Note: present.py functions might need to be exposed or adapted for partial decryption if needed.
# But for the "Check whole key candidate" approach, we can do full 1-round decryption.

def key_recovery_attack():
    MODEL_PATH = "distinguisher_model.pth"
    ROUNDS = 7 # Attack 7 rounds
    DISTINGUISHER_ROUNDS = 6 # Trained on 6
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = Distinguisher().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    print("Loaded model.")
    
    # 1. Setup Target
    TRUE_KEY = 0x1234567890ABCDEF1234 # 80-bit
    TARGET_DELTA = 0x0100000000000000
    NUM_PAIRS = 50 
    
    print(f"Target Key: {TRUE_KEY:x}")
    print(f"Generating {NUM_PAIRS} pairs encrypted with 7 rounds...")
    
    # Generate pairs P1, P2
    pairs = []
    # Just generate random P1, P2 = P1 ^ Delta
    for _ in range(NUM_PAIRS):
        p1 = random.getrandbits(64)
        p2 = p1 ^ TARGET_DELTA
        
        # Encrypt to 7 rounds
        c1 = present_encrypt_block(p1, TRUE_KEY, rounds=ROUNDS)
        c2 = present_encrypt_block(p2, TRUE_KEY, rounds=ROUNDS)
        pairs.append((c1, c2))
        
    # 2. Attack: Candidates
    # We simulate a brute force on the last round key.
    # The last round key K_7 (post-whitening) is 64 bits.
    # We cannot check 2^64.
    # We will check: [True Key K_7] + [N Random Keys] + [Close Keys]
    
    true_round_keys = [0] # dummy
    from present import generate_round_keys
    rks = generate_round_keys(TRUE_KEY, rounds=ROUNDS)
    TRUE_K7 = rks[ROUNDS] # The post-whitening key
    
    candidates = []
    candidates.append((TRUE_K7, "True Key"))
    
    # Add random keys
    for i in range(10):
        candidates.append((random.getrandbits(64), f"Random Key {i}"))
        
    # Add "Near Miss" keys (flip 1 bit)
    for i in range(5):
        k_near = TRUE_K7 ^ (1 << i)
        candidates.append((k_near, f"Near Key (bit {i})"))
        
    print(f"Testing {len(candidates)} candidates...")
    
    results = []
    
    for cand_key, label in candidates:
        # Decrypt 1 round using this candidate key
        # Decryption of 1 round from C_out to State_{R-1}:
        # C_out = State_R ^ K_{R+1} (Post whitening)
        # We need to invert the last round to get to State_{R-1} output?
        # Wait, the Distinguisher is trained on Output of 6 rounds.
        # Encryption 6 rounds: P -> ... -> State_6 ^ K_7 (Post whitening defined in encrypt_block for 6 rounds?)
        # My present.py:
        # encrypt(rounds=6):
        #   for i in range(6): round(i)
        #   state ^ K_6
        # So output is State_6 (after layer) XOR K_7 (index 6 0-based is 7th key K1..K32).
        
        # Attack on 7 rounds:
        # P -> ... -> State_7 ^ K_8
        # We start with C = State_7 ^ K_8.
        # We want to go back to State_6 ^ K_7.
        # Decrypt 1 round:
        # 1. Undo K_8 whitening: State_7 = C ^ K_8. (K_8 is our guess).
        # 2. Undo Round 7: InvP -> InvS -> InvAddRoundKey(K_7).
        # Wait, if we undo Round 7, we get State_6.
        # But Distinguisher expects (State_6 ^ K_7)?
        # Or does Distinguisher expect Raw State_6?
        # My generating script used `present_encrypt_block(..., rounds=6)`.
        # Which returns `State_6 ^ K_7`.
        # So Distinguisher inputs `(State_6 ^ K_7) XOR (State_6' ^ K_7)`.
        # Note: `K_7` cancels out in the difference!
        # `(A ^ K) ^ (B ^ K) = A ^ B`.
        # So the Distinguisher ACTUALLY sees `State_6 ^ State_6'`.
        # It does NOT depend on the whitening key of the previous round!
        # It only depends on the DIFFERENCE of the outputs of the previous round.
        
        # So, to attack 7 rounds:
        # We have `C = State_7 ^ K_8`.
        # Diff C = `State_7 ^ State_7'`. (K_8 cancels out).
        # Wait, if K_8 cancels out in Diff C, then we observe `Delta State_7` directly.
        # So why guess K_8?
        # Because we need `Delta State_6`.
        # To get `Delta State_6`, we need to decrypt `State_7` to `State_6`.
        # `State_7 = P(S(State_6 ^ K_7))`.
        # So `InvP(State_7) = S(State_6 ^ K_7)`.
        # `InvS(InvP(State_7)) = State_6 ^ K_7`.
        # So if we know `State_7`, we can get `State_6 ^ K_7` (and its diff).
        # BUT we don't know `State_7`! We only know `Diff State_7`?
        # No, we know `C1` and `C2`.
        # `C1 = State_7_1 ^ K_8`. `C2 = State_7_2 ^ K_8`.
        # If we DON'T know K_8, we can't get `State_7_1`.
        # We only know `C1`.
        # `State_7_1 = C1 ^ K_8`.
        # So `State_6 ^ K_7 = InvS(InvP(C1 ^ K_8))`.
        # This DEPENDS on K_8 because of the non-linear S-layer!
        # `InvS(InvP(C ^ K))`. The S-box is non-linear.
        # So `Diff State_6` depends on `K_8`.
        # Therefore, guessing `K_8` matters.
        
        scores = []
        for c1, c2 in pairs:
            # 1. Decrypt 1 round with guessed key (assumed to be post-whitening K_{R+1})
            # Wait, 1 round logic:
            # We want to recover `State_{R-1} ^ K_R`.
            # Our `present_decrypt_block` does: Add(K_post) -> InvP -> InvS -> Add(K_R-1).
            # We just want to peel the LAST layer.
            # Step A: Remove output whitening.
            s1 = c1 ^ cand_key
            s2 = c2 ^ cand_key
            
            # Step B: InvP
            s1 = inv_p_layer(s1)
            s2 = inv_p_layer(s2)
            
            # Step C: InvS
            s1 = inv_sbox_layer(s1)
            s2 = inv_sbox_layer(s2)
            
            # Now we have `State_{R-1} ^ K_R`.
            # This is exactly the output format of `present_encrypt_block(rounds=R-1)`.
            # Calculate diff
            diff = s1 ^ s2
            
            # Feed to Distinguisher
            # Convert to bits
            b_val = diff.to_bytes(8, 'big', signed=False) # python int is unsigned here usually
            # My training used signed=True for int64 view.
            # But here `diff` is uint64 (positive python int).
            # Logic: `to_bytes(8, 'big')`.
            # If I stick to unsigned:
            bits = []
            for b in b_val:
                for i in range(7, -1, -1):
                    bits.append((b >> i) & 1)
            
            t_in = torch.tensor(bits, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                prob = model(t_in).item()
                scores.append(prob)
                
        avg_score = sum(scores) / len(scores)
        results.append((label, avg_score))
        print(f"Key: {label[:15]}... Score: {avg_score:.4f}")
        
    # Sort
    results.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 Candidates:")
    for l, s in results[:5]:
        print(f"{l}: {s:.4f}")
        
    # Save results for plotting
    import json
    with open("attack_results.json", "w") as f:
        json.dump(results, f)
    print("Results saved to attack_results.json")

if __name__ == "__main__":
    key_recovery_attack()
