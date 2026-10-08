# Medical Dataset Documentation: PneumoniaMNIST

## 1. Primary Dataset Overview
- **Dataset Source**: **PneumoniaMNIST** from the official MedMNIST v3 collection (Yang et al., Nature Scientific Data / Zenodo).
- **Official Direct File**: `https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1`
- **Official Repository**: `https://github.com/MedMNIST/MedMNIST`
- **Clinical Origin**: Based on Kermany et al.'s pediatric chest X-ray cohort (5,856 images).
- **Download Method**: Automated via `medmnist.PneumoniaMNIST(split="train", root="data/raw", download=True)`.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 2. Why This Dataset?
> **"We use chest X-ray images because the final project is designed around a hospital scenario. The dataset gives us a real medical-image classification problem that can later be distributed among simulated hospitals."**

---

## 3. Official Dataset Specifications
Unlike generic vision datasets, **we preserve the official MedMNIST train, validation, and test splits without arbitrary re-splitting**:
- **Total Samples**: 5,856 images.
- **Official Splits**:
  - **Train split**: 4,708 images (for local/global model training).
  - **Validation split**: 524 images (for checkpoint model selection).
  - **Test split**: 624 images (strictly untouched until final model evaluation).
- **Image Dimensions**: $28 \times 28$ pixels, single grayscale channel (shape: `(1, 28, 28)`).
- **Binary Diagnosis Classes**:
  - `0`: **Normal**
  - `1`: **Pneumonia**
- **Class Breakdown Across Splits**:
  - **Train**: Normal = 1,214 (25.8%), Pneumonia = 3,494 (74.2%)
  - **Validation**: Normal = 135 (25.8%), Pneumonia = 389 (74.2%)
  - **Test**: Normal = 234 (37.5%), Pneumonia = 390 (62.5%)

---

## 4. Preprocessing & Normalization
- **Raw Format**: Grayscale uint8 pixel intensities in $[0, 255]$.
- **Tensor Conversion**: `torchvision.transforms.ToTensor()`, scaling intensities to $[0.0, 1.0]$.
- **Grayscale Normalization**:
  - Mean: `(0.57192,)`
  - Standard Deviation: `(0.16835,)`
  - Computed empirically across all 4,708 training images.
- **Training Augmentations**: `RandomHorizontalFlip()`, `RandomRotation(degrees=10)`.

---

## 5. Measured Dataset Verification Results (Phase MEDICAL-01)
- **Status**: VERIFIED & PASSED (`EXP-MED-000-DATA`)
- **Execution Timestamp**: `2026-10-08T20:24:37`
- **Deterministic Seed**: `42`
- **Empirical Statistics**:
  - Min Pixel: `0`
  - Max Pixel: `255`
  - Normalized Mean: `0.57192`
  - Normalized Std: `0.16835`
- **Generated Artifacts**:
  - Sample Grid: `results/figures/pneumonia_sample_grid.png` (Representative Normal vs Pneumonia chest X-rays)
  - Class Distribution: `results/figures/pneumonia_class_distribution.png` (Train vs Val vs Test bar chart)
  - JSON Summary: `results/logs/pneumonia_mnist_verification_report.json`

---

## 6. Planned Hospital Partitioning (Phase 4)
- **Number of Simulated Hospitals**: 10 clients.
- **Partitioning Technique**: Dirichlet distribution over label allocations ($\alpha \in \{1.0, 0.5, 0.1\}$).
- **Clinical Meaning**: Simulates realistic variations in case-mix across different hospital types (e.g., pediatric pulmonary centers having a higher proportion of pneumonia cases than general clinics).

---

## 7. Archived Previous Proof-of-Concept
The initial generic CIFAR-10 exploration is preserved in [`results/archive/cifar10/`](file:///home/suvranil/project/results/archive/cifar10/README.md).
