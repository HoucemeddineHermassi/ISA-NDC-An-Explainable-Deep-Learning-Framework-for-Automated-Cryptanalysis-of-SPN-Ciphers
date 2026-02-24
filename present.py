import numpy as np

# S-Box
SBOX = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
INVERSE_SBOX = [0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD, 0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA]

# Permutation Table (P-Layer)
# Bit i of Input is moved to Bit P_LAYER[i] of Output
P_LAYER = [
    0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
    4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
    8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
    12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63
]

# Inverse P-Layer
# Bit i of Input is moved to Bit INV_P_LAYER[i] of Output
INV_P_LAYER = [0] * 64
for i, val in enumerate(P_LAYER):
    INV_P_LAYER[val] = i

def generate_round_keys(key, rounds=31):
    """
    Generate round keys for PRESENT-80.
    Key is an 80-bit integer.
    Returns a list of 32 round keys (64-bit integers).
    """
    round_keys = []
    key_state = key
    
    for i in range(1, rounds + 2):
        # 1. Extract 64-bit round key (leftmost 64 bits of key_state)
        round_key = (key_state >> 16) & 0xFFFFFFFFFFFFFFFF
        round_keys.append(round_key)
        
        # 2. Update Key
        # Rotate left by 61 positions
        key_state = ((key_state << 61) & 0xFFFFFFFFFFFFFFFFFFFF) | (key_state >> 19)
        
        # S-Box on top 4 bits (bits 79-76)
        sbox_in = (key_state >> 76) & 0xF
        sbox_out = SBOX[sbox_in]
        key_state = (key_state & 0x0FFFFFFFFFFFFFFFFFFF) | (sbox_out << 76)
        
        # XOR with round counter (bits 19-15)
        key_state ^= (i << 15)
        
    return round_keys

def add_round_key(state, round_key):
    return state ^ round_key

def sbox_layer(state):
    output = 0
    for i in range(16): # 16 nibbles
        nibble = (state >> (i * 4)) & 0xF
        output |= (SBOX[nibble] << (i * 4))
    return output

def inv_sbox_layer(state):
    output = 0
    for i in range(16):
        nibble = (state >> (i * 4)) & 0xF
        output |= (INVERSE_SBOX[nibble] << (i * 4))
    return output

def p_layer(state):
    """
    Bit i of state is moved to bit P_LAYER[i]
    """
    output = 0
    for i in range(64):
        if (state >> i) & 1:
            output |= (1 << P_LAYER[i])
    return output

def inv_p_layer(state):
    """
    Reverse of p_layer.
    Bit i of state is moved to bit INV_P_LAYER[i]
    """
    output = 0
    for i in range(64):
        if (state >> i) & 1:
            output |= (1 << INV_P_LAYER[i])
    return output

def present_encrypt_block(plaintext, key, rounds=31):
    round_keys = generate_round_keys(key, rounds)
    state = plaintext
    
    for i in range(rounds):
        state = add_round_key(state, round_keys[i])
        state = sbox_layer(state)
        state = p_layer(state)
        
    state = add_round_key(state, round_keys[rounds])
    return state

def present_decrypt_block(ciphertext, key, rounds=31):
    round_keys = generate_round_keys(key, rounds)
    state = ciphertext
    
    state = add_round_key(state, round_keys[rounds])
    
    for i in range(rounds - 1, -1, -1):
        state = inv_p_layer(state)
        state = inv_sbox_layer(state)
        state = add_round_key(state, round_keys[i])
        
    return state

def encrypt_image(image_np, key, rounds=31):
    """
    Encrypts a grayscale image (numpy array) using PRESENT-ECB.
    """
    H, W = image_np.shape
    flat = image_np.flatten()
    
    pad_len = (8 - (len(flat) % 8)) % 8
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), 'constant')
        
    encrypted_bytes = bytearray()
    byte_data = flat.tobytes()
    
    for i in range(0, len(byte_data), 8):
        block_bytes = byte_data[i:i+8]
        block_int = int.from_bytes(block_bytes, byteorder='big')
        enc_int = present_encrypt_block(block_int, key, rounds)
        enc_bytes = enc_int.to_bytes(8, byteorder='big')
        encrypted_bytes.extend(enc_bytes)
        
    enc_flat = np.frombuffer(encrypted_bytes, dtype=np.uint8)
    return enc_flat[:H*W].reshape(H, W)
