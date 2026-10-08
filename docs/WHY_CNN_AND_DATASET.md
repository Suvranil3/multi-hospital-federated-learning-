# Rationale: Why This Dataset and Why This Model?

## 1. Core Explanations in Simple Language

### Why This Dataset?
> **"We use chest X-ray images because the final project is designed around a hospital scenario. The dataset gives us a real medical-image classification problem that can later be distributed among simulated hospitals."**

### Why This Model?
> **"We use a small CNN because it is lightweight, easy to train, and suitable for running the same model across multiple simulated hospitals."**

### Why This Matters for Federated Learning?
> **"Each hospital can train on its own local medical images without directly sharing the raw patient images. The hospitals can contribute model updates to a shared global model."**

---

## 2. In-Depth Analysis: The Primary Dataset (PneumoniaMNIST)

### 2.1. Clinical Relevance vs. Generic Benchmarks
In previous iterations, the project used CIFAR-10 (natural images of vehicles and animals). While CIFAR-10 is common in generic computer science benchmarks, it lacks the characteristics of real clinical decision-making:
- **Medical Realism**: PneumoniaMNIST represents genuine pediatric chest radiography for diagnosing viral and bacterial pneumonia versus normal lungs.
- **Grayscale Radiography**: Unlike 3-channel color photographs, radiographic scans are single-channel grayscale ($1 \times 28 \times 28$) where diagnostic information relies on tissue densities, opacities, and lung field symmetry.
- **Natural Clinical Class Imbalance**: In the real world, hospital patient cohorts are rarely balanced 50/50. PneumoniaMNIST features a realistic imbalance (~74% pneumonia, ~26% normal), requiring nuanced evaluation (precision, sensitivity, F1) that mirrors healthcare evaluation standards.

### 2.2. Official Dataset Integrity
We use the official MedMNIST v3 benchmark distribution without fabricating random splits:
- **Total Images**: **5,856**
- **Train Split (Official)**: **4,708** images (1,214 Normal, 3,494 Pneumonia)
- **Validation Split (Official)**: **524** images (135 Normal, 389 Pneumonia)
- **Test Split (Official, Untouched)**: **624** images (234 Normal, 390 Pneumonia)

Preserving the official benchmark splits ensures:
1. Model selection (checkpointing) is performed purely on the official validation split.
2. The official test split remains strictly unobserved during hyperparameter tuning, preventing data leakage.
3. Our results are directly comparable to published medical AI literature.

---

## 3. In-Depth Analysis: The Model Architecture (SmallCNN)

### 3.1. Architectural Layout
The model is implemented in [`federated_learning/model.py`](file:///home/suvranil/project/federated_learning/model.py#L21-L71) as [`SmallCNN`](file:///home/suvranil/project/federated_learning/model.py#L21-L71):

```text
Input Grayscale Image: (1 × 28 × 28)
       │
       ▼
Conv Block 1: Conv2d(1 -> 32, 3×3, pad=1) + BatchNorm2d(32) + ReLU + MaxPool2d(2×2)
       │ Output: (32 × 14 × 14)
       ▼
Conv Block 2: Conv2d(32 -> 64, 3×3, pad=1) + BatchNorm2d(64) + ReLU + MaxPool2d(2×2)
       │ Output: (64 × 7 × 7)
       ▼
Flatten Layer: 64 × 7 × 7 = 3,136 features
       │
       ▼
Dense Head: Linear(3136 -> 128) + ReLU + Dropout(p=0.2)
       │ Output: 128 features
       ▼
Classification Head: Linear(128 -> 2)
       │
       ▼
Logits Output: [Normal, Pneumonia]
```

### 3.2. Detailed Parameter Count
| Layer | Weight Shape | Bias Shape | Total Parameters |
| :--- | :---: | :---: | :---: |
| `features.0 (Conv2d)` | `(32, 1, 3, 3)` | `(32,)` | 320 |
| `features.1 (BatchNorm2d)` | `(32,)` | `(32,)` | 64 |
| `features.4 (Conv2d)` | `(64, 32, 3, 3)` | `(64,)` | 18,496 |
| `features.5 (BatchNorm2d)` | `(64,)` | `(64,)` | 128 |
| `classifier.1 (Linear)` | `(128, 3136)` | `(128,)` | 401,536 |
| `classifier.4 (Linear)` | `(2, 128)` | `(2,)` | 258 |
| **Total Parameter Count** | — | — | **420,802** |

### 3.3. Why Not a Giant Model (e.g., ResNet-50 / ViT)?
1. **Simulation Scalability**: In subsequent phases, 10 simulated hospital nodes will execute local epochs in parallel across 20+ communication rounds. Giant models require massive GPU VRAM and lead to extreme training latency per round.
2. **Communication Efficiency**: In real federated learning, hospitals transmit model weights over the internet. A 420k parameter model consumes only **~1.61 MB** per transmission, whereas a 25M parameter ResNet-50 requires ~100 MB per round, creating unacceptable network bottlenecks.
3. **Prevention of Overfitting**: On a dataset of 4,708 training images, giant models rapidly overfit or memorize local data. SmallCNN possesses sufficient feature extraction capacity while generalizing effectively (measured validation accuracy: **97.90%**).

---

## 4. Why This Architecture and Dataset Matter for Federated Learning

1. **Hospital Specialization (Non-IID)**: Because chest X-ray features represent distinct physiological structures, partitioning the data by clinical sub-types or class skews (e.g., pediatric pulmonary clinics having mostly pneumonia vs. health check clinics having mostly normal scans) creates realistic gradient divergence.
2. **Clinical Security**: If an adversary compromises one hospital and flips labels (e.g., poisoning pneumonia cases to be labeled normal), the impact directly mirrors real-world healthcare sabotage (causing missed pneumonia diagnoses).
3. **Auditable Collaboration**: Hospitals can track the exact parameter updates on a lightweight blockchain ledger without risking patient privacy, proving regulatory compliance.
