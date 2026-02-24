"""
Generate Paper Materials (Priority 1)
Plots training dynamics and key recovery separation for R6/R7.
"""
import matplotlib.pyplot as plt
import json
import numpy as np
import os

def generate_plots():
    print("Generating Paper Figures...")
    
    # 1. Training Dynamics (Mock or Real)
    # If training history exists, use it. Else validation.
    # For now, we'll setup the structure to read 'training_log.json' if we add it, 
    # or just plotting the final key recovery.
    
    # 2. Key Recovery Evolution
    results_file = "attack_r7_results.json"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)
            
        results = data['results']
        # Extract scores
        labels = [r[0] for r in results]
        scores = [r[1] for r in results]
        
        # Identify True Key
        true_idx = next(i for i, l in enumerate(labels) if "True Key" in l)
        true_score = scores[true_idx]
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        # Plot all candidates as gray dots
        plt.scatter(range(len(scores)), scores, color='gray', alpha=0.5, label='Wrong Keys')
        
        # Highlight True Key
        plt.scatter(true_idx, true_score, color='red', s=100, label='True Key')
        
        plt.title('7-Round Key Recovery: Score Separation')
        plt.xlabel('Candidate Index (Sorted by Score)')
        plt.ylabel('Log-Likelihood Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = r"e:\Anti-Gravity-Article\Attack-KPA-PRESENT\paper\key_recovery_r7.png"
        plt.savefig(out_path)
        print(f"Saved {out_path}")
    else:
        print(f"{results_file} not found. Run attack_r7.py first.")

if __name__ == "__main__":
    generate_plots()
