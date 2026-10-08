# CIFAR-10 Dataset Documentation

## 1. Dataset Overview
- **Dataset Source**: Canadian Institute for Advanced Research (CIFAR-10), curated by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
- **Official URL**: https://www.cs.toronto.edu/~kriz/cifar.html
- **Download Method**: Automated download via `torchvision.datasets.CIFAR10(root="data/raw", download=True)` to local directory `data/raw/`.
- **License**: Public domain / academic research use.

---

## 2. Dataset Specifications
- **Total Samples**: 60,000 color images.
- **Train / Test Split**:
  - Training Set: 50,000 images (5 single batches of 10,000 images each).
  - Test Set: 10,000 images.
- **Image Dimensions**: $32 \times 32$ pixels, 3 color channels (RGB).
- **Number of Classes**: 10 mutually exclusive classes.
- **Class Labels**:
  0. `airplane`
  1. `automobile`
  2. `bird`
  3. `cat`
  4. `deer`
  5. `dog`
  6. `frog`
  7. `horse`
  8. `ship`
  9. `truck`

---

## 3. Preprocessing & Normalization
- **Raw Format**: PIL Image / uint8 arrays with pixel values in range $[0, 255]$.
- **Tensor Conversion**: `torchvision.transforms.ToTensor()`, scaling pixel intensities to $[0.0, 1.0]$ in shape `(3, 32, 32)`.
- **Channel Normalization**:
  - Pixel values normalized via `transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))`.
  - Normalization transforms each channel to approximate zero mean and unit variance based on empirically measured training distribution.

---

## 4. Planned Client Partitioning (Phase 4)
- **Number of Clients**: 10 simulated edge participants.
- **Partitioning Technique**: Dirichlet distribution over label allocations.
  - For each class $k \in \{0, \dots, 9\}$, draw a probability distribution over the 10 clients:
    $$p_k \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_{10})$$
  - Allocate class $k$ samples to client $i$ according to proportion $p_{k,i}$.
- **Heterogeneity Severity Levels**:
  - $\alpha = 1.0$: Mild non-IID (moderate label skew across clients).
  - $\alpha = 0.5$: Moderate non-IID (significant class imbalance per client).
  - $\alpha = 0.1$: Extreme non-IID (each client dominated by 1-2 classes).
- **Data Partitions Storage**: Will be saved to `data/processed/client_partitions_alpha_<val>.pt`.

---

## 5. Measured Dataset Verification Results (Phase 1)
- **Status**: VERIFIED & PASSED (`EXP-000-DATA`)
- **Execution Timestamp**: `2026-10-08T15:04:46`
- **Deterministic Seed**: `42`
- **Exact Sample Counts**:
  - Train: 50,000 (Exactly 5,000 per class across all 10 classes)
  - Test: 10,000 (Exactly 1,000 per class across all 10 classes)
  - Total: 60,000 images
- **Raw Image Resolution**: $32 \times 32$ pixels, 3 color channels (RGB)
- **Tensor Shape**: `(3, 32, 32)` in float32 normalized $[0.0, 1.0]$
- **Class Balance**: 100% perfectly balanced across all 10 classes
- **Measured Channel Means (Train Set)**:
  - Red (R): `0.49140`
  - Green (G): `0.48216`
  - Blue (B): `0.44653`
- **Measured Channel Standard Deviations (Train Set)**:
  - Red (R): `0.24703`
  - Green (G): `0.24349`
  - Blue (B): `0.26159`
- **Verification Artifacts Generated**:
  - Sample Grid: `results/figures/cifar10_sample_grid.png` (10 classes $\times$ 5 samples)
  - Class Distribution: `results/figures/cifar10_class_distribution.png` (Train vs Test bar chart)
  - JSON Summary: `results/logs/cifar10_verification_report.json`

