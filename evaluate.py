import matplotlib.pyplot as plt
import json
import numpy as np

def generate_plots():
    # 1. Plot Training History (simulated or parsed from log if available)
    # Since we didn't save training history in train.py, we might want to Add that or mock it for the "Paper Artifacts" if real training isn't finished.
    # But let's assume we parse output or just show the code for plotting.
    
    # 2. Plot Attack Results
    try:
        with open("attack_results.json", "r") as f:
            results = json.load(f)
            
        # Structure: list of [label, score]
        # Separate True Key, Random Keys, Near Keys
        
        true_key_score = None
        random_scores = []
        near_scores = []
        
        for label, score in results:
            if "True Key" in label:
                true_key_score = score
            elif "Random" in label:
                random_scores.append(score)
            elif "Near" in label:
                near_scores.append(score)
                
        plt.figure(figsize=(10, 6))
        
        # Plot Random
        plt.scatter(["Random"] * len(random_scores), random_scores, color='gray', alpha=0.5, label='Random Keys')
        
        # Plot Near
        plt.scatter(["Near"] * len(near_scores), near_scores, color='blue', marker='x', label='Near Keys (1-bit flip)')
        
        # Plot True
        if true_key_score is not None:
            plt.scatter(["True"], [true_key_score], color='red', marker='*', s=200, label='Correct Key')
            
        plt.ylabel("Distinguisher Score (Probability)")
        plt.title("Key Recovery Attack: Candidate Scores")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("key_recovery_scores.png")
        print("Generated key_recovery_scores.png")
        
    except FileNotFoundError:
        print("attack_results.json not found. Run attack.py first.")

if __name__ == "__main__":
    generate_plots()
