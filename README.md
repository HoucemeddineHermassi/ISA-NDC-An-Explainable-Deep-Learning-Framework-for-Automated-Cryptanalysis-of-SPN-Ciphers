# ISA-NDC-An-Explainable-Deep-Learning-Framework-for-Automated-Cryptanalysis-of-SPN-Ciphers
This preprint introduces ISA-NDC, a neural differential cryptanalysis tool for SPN ciphers (PRESENT, GIFT-64). It features a hybrid ResNet-Transformer architecture + curriculum learning to distinguish differential pairs across rounds
**ISA-NDC** is a state-of-the-art framework for automated differential cryptanalysis of Substitution-Permutation Network (SPN) ciphers (particularly **PRESENT** and **GIFT-64**). Unlike traditional neural cryptanalysis which assumes uniform random distribution, ISA-NDC leverages the inherent structure of **medical images** (using the [COVID-19 Radiography Dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)) to perform automated differential mining and key recovery.

---

## 🚀 Key Innovations

### 1. Hybrid ResNet-Transformer Architecture
ISA-NDC breaks the "diffusion bottleneck" in block ciphers by combining:
- **ResNet Blocks**: For local feature extraction and handling the high non-linearity of S-Boxes.
- **Transformer Encoders**: For global diffusion modeling, capturing long-range bit dependencies that emerge across multiple encryption rounds.

### 2. Image-Structure-Aware Differential Mining
Traditional attacks often struggle with the data complexity of finding "good" differentials. ISA-NDC mines differentials directly from the structure of medical images, identifying high-probability XOR differences occurring between adjacent pixel blocks in grayscale radiographs.

### 3. Multi-Round Curriculum Learning
Training a neural distinguisher for $N$ rounds is computationally hard. ISA-NDC employs a progressive training regimen:
1. Initialize with $3$-round data.
2. Fine-tune on $4$ and $5$ rounds.
3. Converge on the target $6$-round distinguisher using the pre-learned features.

### 4. Neural-Aided Key Recovery (Bayesian Scoring)
The framework integrates the neural distinguisher into a key recovery pipeline. By evaluating subkey candidates against the neural score, ISA-NDC can recover key bits with significantly lower data complexity than classical differential cryptanalysis.

---

## 📂 Repository Structure

```text
├── cryptanalysis_present/    # Core implementations of PRESENT & GIFT-64
│   ├── src/cipher/           # PyTorch and NumPy implementations
│   └── src/attacks/          # Key recovery logic
├── COVID-19_Radiography_Dataset/ # (External) Data source for training
├── data/                     # Generated datasets (.npy)
├── models/                   # Saved model weights (.pth)
├── paper/                    # Scripts for generating figures & assets
├── train_isa_ndc.py         # Main curriculum training pipeline
└── data_miner.py             # Image-to-differential mining utility
```

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.8 or higher
- CUDA-enabled GPU (recommended for 6+ rounds)
- Requirements: `pip install torch torchvision numpy pillow matplotlib scipy`

### Installation
```bash
git clone https://github.com/[your-repo]/ISA-NDC.git
cd ISA-NDC
```

### Quick Start Workflow

1. **Mine Differentials & Generate Data**:
   This script processes the medical images, identifies top differentials, and generates the training dataset.
   ```bash
   python data_miner.py
   ```

2. **Run Curriculum Training**:
   Train the enhanced distinguisher across multiple rounds.
   ```bash
   python train_isa_ndc.py
   ```

3. **Verify Interpretability**:
   Generate saliency maps to see which bits the model is focusing on.
   ```bash
   python paper/gen_interpretability_rigor.py
   ```

4. **Execute Key Recovery**:
   Perform the final attack to recover the round keys.
   ```bash
   python paper/gen_key_evolution.py
   ```

---

## � Results Summary

| Rounds | Attack Type | Metric | Performance |
| :--- | :--- | :--- | :--- |
| 5 Rounds | Neural Distinguisher | Accuracy | ~99.2% |
| 6 Rounds | Neural Distinguisher | Accuracy | ~68.5% |
| 6 Rounds | Key Recovery | Success Rate | 100% (within $2^{14}$ pairs) |

*Note: Results are based on the standard PRESENT implementation.*
