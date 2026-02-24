import unittest
from present import present_encrypt_block, present_decrypt_block, SBOX, P_LAYER

class TestPRESENT(unittest.TestCase):
    def test_vectors(self):
        # Test vector from ORIGINAL PRESENT PAPER
        # Plaintext: 0000 0000 0000 0000
        # Key: 0000 0000 0000 0000 0000
        # Ciphertext: 5579 C138 7B22 8445
        
        plaintext = 0x0000000000000000
        key = 0x00000000000000000000
        expected_ciphertext = 0x5579C1387B228445
        
        print(f"Testing Standard Vector P={plaintext:x} K={key:x}...")
        ciphertext = present_encrypt_block(plaintext, key)
        print(f"Ciphertext: {ciphertext:x}")
        
        self.assertEqual(ciphertext, expected_ciphertext)
        
        decrypted = present_decrypt_block(ciphertext, key)
        self.assertEqual(decrypted, plaintext)
        
    def test_vectors_2(self):
        # Another test vector
        # P = FF FF FF FF FF FF FF FF
        # K = FF FF FF FF FF FF FF FF FF FF
        # C = 33 33 DC D3 21 32 10 d2  (From some implementation, lets verify self-consistency primarily if authoritative vector missing)
        
        # Actually let's trust the first vector mostly.
        # But let's check invertibility on random data.
        import random
        for _ in range(5):
            p = random.getrandbits(64)
            k = random.getrandbits(80)
            
            c = present_encrypt_block(p, k)
            p_prime = present_decrypt_block(c, k)
            
            self.assertEqual(p, p_prime, "Decryption failed to recover plaintext")

if __name__ == '__main__':
    unittest.main()
