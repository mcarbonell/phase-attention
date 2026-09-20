# PhaseAttention: Experimental Documentation & Technical Reports

Welcome to the technical documentation suite for **PhaseAttention**. This directory provides in-depth experimental reports, mathematical formulations, hardware cost models, and clinical/vision benchmarks validating the architecture.

---

## 📚 Index of Technical Reports

| # | Document | Focus Domain | Key Finding / Benchmark | Target Hardware |
| :---: | :--- | :--- | :--- | :--- |
| **01** | [**Associative Recall & Toroidal Scaling**](01_associative_recall_torus.md) | Theory & Synthetic Retrieval | Explains $\mathbb{R}^1$ monotonic collapse (59.5%), $U(1)$ resonance (100%), and $T^H$ toroidal capacity scaling up to $K=64$ keys. | General TinyML |
| **02** | [**Latency, Memory & Complexity Scaling**](02_latency_memory_scaling.md) | Systems & Complexity ($N=128 \dots 8192$) | **82.3x latency speedup** and **455x memory reduction** at $N=8,192$. Documents parallel $O(N)$ training vs $O(1)$ streaming state memory. | Microcontrollers / DSPs |
| **03** | [**Fixed-Point Silicon & FPGA Emulation**](03_fixed_point_silicon_emulation.md) | Digital Hardware & VLSI (INT4/8/16) | Free two's complement modulo wrap on $S^1$. Complete INT8 core requires **~95 NAND2 gates** and **0.04 pJ/op** (47.4x smaller, 92.5x less energy than FP32 MAC). | ASICs / Low-cost FPGAs |
| **04** | [**Clinical ECG Arrhythmia Detection**](04_ecg_arrhythmia_detection.md) | Biomedical Wearables (PhysioNet MIT-BIH) | **93.46% Macro F1** and **98.66% Ventricular sensitivity** under AAMI EC57 standard. Beats SOTA `cosFormer` with 25% fewer parameters and zero Softmax. | Medical Holters / Smart Patches |
| **05** | [**Micro-ViT Embedded Computer Vision**](05_micro_vit_embedded_vision.md) | Embedded Vision (Fashion-MNIST) | Micro-ViT (<10k params, 49 tokens) achieves **80.67% Top-1 Acc**, outperforming modern linear attention and microcontroller 2D CNNs with 16.8% parameter reduction. | ESP32-CAM / Cortex-M55 |

---

## 🔬 Cross-Experiment Performance Matrix

| Metric / Characteristic | Standard Vector Attention | SOTA Linear Attention (`cosFormer`) | Edge-CNN (1D / 2D) | **PhaseAttention (Linear Holographic)** | **PhaseAttention (Triangular)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Time Complexity** | $O(N^2)$ | $O(N)$ | $O(N \cdot K)$ | **$O(N)$** | $O(N^2)$ |
| **Streaming State Memory** | $O(N)$ (KV-Cache) | $O(1)$ | $O(K)$ (Buffer) | **$O(1)$ (16 floats/head)** | $O(N)$ |
| **Softmax Exponentiation** | Required | Replaced with division | Not required | **Completely Eliminated** | Required |
| **$Q \times K$ Multipliers** | $N \cdot d_k$ Float MACs | $4 \cdot d_v$ Float MACs | Kernel MACs | **0 (Prefix Scan)** | **0 (Sub + Abs Only)** |
| **MIT-BIH Arrhythmia F1** | 96.87% | 92.87% | 90.22% | **93.46%** | **92.76%** |
| **Ventricular Arrhythmia Sens.**| 99.33% | 99.33% | 98.66% | **98.66%** | **95.97%** |
| **Micro-ViT Vision Top-1 Acc** | 81.07% | 80.33% | 78.07% | **80.67%** | **80.33%** |
| **Attention Projection Overhead**| Large ($d \times (H \cdot d_k)$) | Large ($d \times (H \cdot d_v)$) | N/A | **Minimal ($d \times H$)** | **Minimal ($d \times H$)** |
| **Silicon Gate Estimate** | ~4,500 NAND2 | ~3,500 NAND2 | ~1,200 NAND2 | **~350 NAND2** | **~95 NAND2** |

---

## 🛠️ How to Reproduce All Experiments

To reproduce any benchmark, run the corresponding script from the repository root:

```bash
# 1. Associative Recall & Toroidal Scaling
python experiments/benchmark_recall.py

# 2. Complexity, Latency & Memory Scaling
python experiments/benchmark_scaling_latency.py

# 3. Multiplier-Free Fixed-Point Silicon Emulation
python experiments/benchmark_fixed_point_integer.py

# 4. Clinical ECG Arrhythmia Detection (MIT-BIH)
python experiments/benchmark_ecg_arrhythmia.py

# 5. Micro-ViT Embedded Vision (Fashion-MNIST)
python experiments/benchmark_micro_vit.py
```

All raw numerical results are recorded as JSON files in `results/`, and all publication figures are stored at 300 DPI in `assets/`.
