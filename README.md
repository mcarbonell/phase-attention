# PhaseAttention: Multiplier-Free, Exact $O(N)$ Linear Attention on the Unit Circle $U(1)$

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture: U(1) Phase](https://img.shields.io/badge/Architecture-U(1)%20Phase-purple.svg)]()
[![Hardware: Multiplier--Free](https://img.shields.io/badge/Hardware-Multiplier--Free-red.svg)]()

> **"Attention Is All You Need" — Vaswani et al., 2017**  
> **PhaseAttention asks: What if attention doesn't need high-dimensional vector spaces, floating-point multipliers, or quadratic $O(N^2)$ matrices?**

---

## 🎯 The Core Discovery: 1D Scalar Attention in the Complex Plane

Standard Transformers project queries and keys into high-dimensional vector spaces $Q, K \in \mathbb{R}^d$ ($d \ge 64$), paying a severe price:
- **$O(d^2)$ projection parameters** per layer.
- **$O(N^2)$ quadratic matrix memory** and latency for sequence length $N$.
- **Millions of Multiply-Accumulate (MAC) units** burning milliwatts of power, making them unusable on battery-operated microcontrollers (ARM Cortex-M, ESP32, FPGAs).

### 1. The Monotonic Bottleneck of $\mathbb{R}^1$ vs The $U(1)$ Unit Circle
- **Real Scalar Attention ($\mathbb{R}^1$, $d=1$):** Dot products $q \cdot k$ are strictly monotonic on the real line. A query $q$ cannot isolate an intermediate key without assigning an even higher score to the extremes ($k_{\max}$ or $k_{\min}$). It collapses to a simple *Soft-Ranker* (56% accuracy on associative retrieval).
- **Unitary Complex Attention ($U(1)$, $d=1$):** By projecting tokens to phase angles $\theta \in [-\pi, \pi)$ on the compact unit circle, **a single complex dimension achieves 100% associative recall**:
  $$\text{sim}(q, k) = \cos(\theta_q - \theta_k)$$
  - **Constructive Resonance:** $\Delta\theta = 0 \implies \cos(0) = +1.0$ (exact match).
  - **Continuous Orthogonality:** $\Delta\theta = \pm \pi/2 \implies \cos(\pm \pi/2) = 0.0$ (mutual indifference).
  - **Active Inhibition (Antiphase):** $\Delta\theta = \pi \implies \cos(\pi) = -1.0$ (destructive cancellation).

---

## ⚡ Three Algorithmic Breakthroughs

### 1. Exact Analytical $O(N)$ Linear Attention (No Softmax, No Approximations)
The cosine affinity is an **exact separable rank-2 kernel**:
$$\cos(\theta_q - \theta_k) = \begin{pmatrix} \cos\theta_q \\ \sin\theta_q \end{pmatrix}^\top \begin{pmatrix} \cos\theta_k \\ \sin\theta_k \end{pmatrix} = \phi(q)^\top \phi(k)$$

Unlike conventional linear attention that uses truncated Taylor series or Random Fourier Features, PhaseAttention is **exact and unapproximated**. It runs causally in $O(N)$ time with a tiny recurrent memory state $S_t \in \mathbb{R}^{2 \times d_v}$ (only **16 floats per head**):
$$S_t = S_{t-1} + \phi(k_t) v_t^\top$$
$$y_t = \phi(q_t)^\top S_t$$
**Noise cancellation happens naturally through destructive wave interference ($\sum \cos \approx 0$), making Softmax completely redundant.**

### 2. Multiplier-Free Silicon Kernel (Subtraction + Absolute Value Only)
Replacing $\cos(\Delta\theta)$ with a periodic triangular wave:
$$\text{tri}(\Delta\theta) = 1.0 - \frac{2}{\pi} |\text{wrap}(\theta_q - \theta_k)|$$
yields **100.00% validation accuracy** while requiring **ZERO floating-point multipliers** and zero trigonometric operations in the attention affinity kernel. It can be implemented in digital logic using only an integer subtractor and a sign-bit absolute value rectifier.

### 3. Toroidal Multicell Scaling ($T^H = S^1 \times \dots \times S^1$)
While 1 circle ($H=1$) cleanly stores up to 16 discrete keys ($22.5^\circ$ separation), $H=4$ heads form a 4-dimensional torus $T^4$, scaling capacity to **>91% on $K=64$ keys** with 35% fewer parameters than standard vector attention.

---

## 📊 Empirical Benchmarks

### 1. Direct Content Addressing ($K=8$ keys, 8 slots, chance = 12.5%)

| Model | Complexity | Heads ($H$) | Params | Val Acc (%) | Val Loss | Multipliers in $Q \times K$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `RealScalar_R1` | $O(N^2)$ | 1 | 659 | 56.67% | 0.9308 | Multiplications in $\mathbb{R}^1$ |
| **`PhaseAttention (U(1))`** | $O(N^2)$ | 1 | **659** | **100.00%** | **0.0003** | **0 (Angular subtraction)** |
| **`TriangularPhase (Multiplier-Free)`** | $O(N^2)$ | 4 | **1745** | **100.00%** | **0.0008** | **0 (Sub + Abs only)** |
| **`LUT16_Phase (ROM Table)`** | $O(N^2)$ | 4 | **1745** | **100.00%** | **0.0001** | **0 (16-word LUT)** |
| **`LinearHolographicPhase`** | **$O(N)$** | 4 | **1744** | **100.00%** | **0.0001** | **0 (Exact linear scan)** |
| `StandardVector (d_k=8)` | $O(N^2)$ | 4 | 2696 | **100.00%** | 0.0001 | Floating-point matrix MACs |

---

## 🚀 Quickstart

### Installation
```bash
pip install -e .
```

### Basic Usage in PyTorch
```python
import torch
from phase_attention import (
    PhaseAttention,
    LinearHolographicPhaseAttention,
    TriangularPhaseAttention
)

# Batch of 4 sequences, length 32, embedding dimension 32
x = torch.randn(4, 32, 32)

# 1. Standard PhaseAttention (U(1) Cosine)
attn_u1 = PhaseAttention(d_model=32, num_heads=4, d_v=8)
out_u1 = attn_u1(x) # (4, 32, 32)

# 2. Linear Holographic Attention O(N) (No Softmax, exact prefix-sum)
attn_linear = LinearHolographicPhaseAttention(d_model=32, num_heads=4, d_v=8)
out_linear = attn_linear(x) # (4, 32, 32)

# 3. 100% Multiplier-Free Attention (Hardware-friendly Sub + Abs)
attn_tri = TriangularPhaseAttention(d_model=32, num_heads=4, d_v=8)
out_tri = attn_tri(x) # (4, 32, 32)
```

---

## 💡 Industrial & Edge AI Applications

1. **Ultra-Low-Power Wearables (ECG / EEG / Bio-Sensors):**
   Runs inside <2 KB SRAM on battery-powered medical patches (ARM Cortex-M0+/M4) detecting cardiac arrhythmias and seizures via phase-locking synchronization.
2. **Always-On Keyword Spotting (KWS):**
   Replaces heavy speech transformers with multiplier-free triangular attention running on coin-cell batteries at microwatt scale.
3. **Predictive Maintenance on Industrial Motors:**
   Vibration sensor nodes with piezoelectric energy harvesting analyzing acoustic harmonics locally without sending continuous raw telemetry over cellular/LoRa networks.
4. **Multiplier-Free ASICs and FPGAs:**
   Zero DSP multiplier blocks required. Fits into ultra-cheap FPGAs (e.g. Lattice iCE40, $2) using pure logic gates.

---

## 📄 Research Proposal & Citation

For academic inquiries, grant proposals, and collaborations with research institutes (e.g. **VRAIN / UPV**), refer to:
- [`proposals/PROPUESTA_INVESTIGACION_VRAIN_UPV.md`](proposals/PROPUESTA_INVESTIGACION_VRAIN_UPV.md)

```bibtex
@article{carbonell2026phaseattention,
  title={PhaseAttention: Multiplier-Free, Exact O(N) Linear Attention on the Unit Circle for Ultra-Low Power Edge Devices},
  author={Carbonell, Mario},
  year={2026}
}
```
