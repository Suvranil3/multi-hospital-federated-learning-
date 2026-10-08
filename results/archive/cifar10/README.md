# Archived CIFAR-10 Proof-of-Concept

**Status**: Archived Historical Proof-of-Concept  
**Date of Completion**: October 2026 (Phase 1 & Phase 2)  
**Role in Research**: Initial generic sanity-check before migration to the primary hospital dataset (**PneumoniaMNIST**).

---

## Why Was This Work Archived?

The professor clarified that the **complete project must be developed and presented in a hospital/medical scenario**. 

As a result:
- **PneumoniaMNIST** (Chest X-Ray classification) is designated as the **Primary Medical Project Dataset**.
- **CIFAR-10** is preserved here in full as an **Archived Previous Proof-of-Concept**.
- No CIFAR-10 results are mixed with medical results.

---

## Summary of Archived Results

### Phase 1: CIFAR-10 Verification
- **Total Images**: 60,000 (50,000 train, 10,000 test) across 10 balanced classes (6,000/class).
- **Resolution**: 32x32 RGB (3 channels).
- **Integrity**: Exact class balance verified; empirical mean and std computed.
- **Log Report**: [`logs/cifar10_verification_report.json`](file:///home/suvranil/project/results/archive/cifar10/logs/cifar10_verification_report.json)

### Phase 2: Centralized SmallCNN Baseline
- **Model**: SmallCNN (3 Conv Blocks, BatchNorm, MaxPool, Linear head, ~189,130 parameters).
- **Training Setup**: 15 Epochs, Batch Size 64, Adam (lr=0.001), Seed 42, NVIDIA RTX 3050 GPU.
- **Train Split**: 40,000 samples (80% stratified).
- **Validation Split**: 10,000 samples (20% stratified).
- **Untouched Test Split**: 10,000 samples.
- **Best Validation Accuracy**: 77.37% (Epoch 15).
- **Final Untouched Test Accuracy**: 76.94% (Test Loss: 0.6578).
- **Training Time**: 100.97 seconds (~1.7 minutes).
- **Checkpoint**: [`models/cifar10_best_centralized_cnn.pt`](file:///home/suvranil/project/results/archive/cifar10/models/cifar10_best_centralized_cnn.pt)
- **Log Report**: [`logs/cifar10_centralized_cnn_results.json`](file:///home/suvranil/project/results/archive/cifar10/logs/cifar10_centralized_cnn_results.json)

### Archived Figures
- [`figures/cifar10_sample_grid.png`](file:///home/suvranil/project/results/archive/cifar10/figures/cifar10_sample_grid.png)
- [`figures/cifar10_class_distribution.png`](file:///home/suvranil/project/results/archive/cifar10/figures/cifar10_class_distribution.png)
- [`figures/centralized_training_loss.png`](file:///home/suvranil/project/results/archive/cifar10/figures/centralized_training_loss.png)
- [`figures/centralized_training_accuracy.png`](file:///home/suvranil/project/results/archive/cifar10/figures/centralized_training_accuracy.png)
- [`figures/cifar10_confusion_matrix.png`](file:///home/suvranil/project/results/archive/cifar10/figures/cifar10_confusion_matrix.png)
