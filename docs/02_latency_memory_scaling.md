# Experiment 2: Computational Complexity, Latency & Memory Scaling

## 📌 Executive Summary

This experiment measures the empirical runtime latency and peak activation memory footprint across sequence lengths ranging from $N = 128$ to $N = 8,192$ tokens. We benchmark `LinearHolographicPhaseAttention` ($O(N)$ time, $O(1)$ streaming state) against `StandardVector Attention` ($O(N^2)$ PyTorch MHA), quadratic `PhaseAttention` ($O(N^2)$), and `TriangularPhaseAttention`. At $N = 8,192$, the linear holographic formulation achieves an **82.3x speedup** and a **455x memory reduction**.

---

## 🔬 Theoretical Foundations

### 1. The Quadratic Wall of Standard Attention

Standard multi-head attention constructs an explicit $N \times N$ attention matrix:
$$A = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times H \times N \times N}$$

This imposes a severe computational burden on edge devices:
- **Time Complexity:** $\mathcal{O}(N^2 \cdot d_k + N^2 \cdot d_v)$ operations.
- **Memory Complexity:** $\mathcal{O}(B \cdot H \cdot N^2)$ floats to materialize attention weights.
- **Microcontroller Limit:** For $N = 8,192$ and $H = 4$, storing $A$ in float32 requires **2,048 MB (2 GB)** of SRAM, exceeding typical microcontroller memory (typically 64 KB to 2 MB) by several orders of magnitude.

### 2. Exact Separable Rank-2 Kernel

PhaseAttention leverages Euler's formula to rewrite circular phase differences:
$$\cos(\theta_q - \theta_k) = \cos\theta_q \cos\theta_k + \sin\theta_q \sin\theta_k = \phi(q)^\top \phi(k)$$
where $\phi(\theta) = [\cos\theta, \; \sin\theta]^\top \in \mathbb{R}^2$.

By applying associative matrix multiplication:
$$Y = \left(\phi(Q) \phi(K)^\top\right) V = \phi(Q) \left(\phi(K)^\top V\right)$$

This changes the computational order from $(N \times N) \times (N \times d_v)$ to $(N \times 2) \times (2 \times d_v)$, reducing computational complexity to strictly linear:
$$\mathcal{O}(N \cdot 2 \cdot d_v) = \mathcal{O}(N)$$

### 3. Dual Operational Regimes: Parallel Training vs. Streaming Inference

1. **Parallel Training ($O(N)$ Time & Memory):**
   The forward pass computes cumulative sums over the entire sequence via `torch.cumsum(phi(K) * V, dim=1)`, parallelizing across all $N$ tokens on GPU/CPU with $O(N)$ activation storage.
2. **Online Streaming Inference ($O(1)$ Time & Memory):**
   For real-time edge processing (biomedical sensors, audio, control loops), tokens arrive sequentially. The recurrent state updates step-by-step:
   $$S_t = S_{t-1} + \phi(k_t) v_t^\top \in \mathbb{R}^{2 \times d_v}$$
   $$y_t = \phi(q_t)^\top S_t \in \mathbb{R}^{d_v}$$
   Memory is strictly bounded to $2 \times d_v$ floats per head (**16 floats / 64 bytes per head** when $d_v=8$).

---

## ⚙️ Experimental Setup

- **Hardware Platform:** AMD Ryzen 7 8845HS (8 Cores / 16 Threads), 64 GB DDR5 RAM, Windows 11.
- **Sequence Lengths Swept:** $N \in \{128, 256, 512, 1024, 2048, 4096, 8192\}$.
- **Model Architecture:**
  - $d_{\text{model}} = 32$
  - Number of heads: $H = 4$
  - Value dimension: $d_v = 8$
  - Batch size: $B = 1$
- **Measurement Protocol:**
  - 10 warm-up runs per configuration.
  - 50 timed iterations using high-precision CPU timers (`time.perf_counter`).
  - Peak activation memory tracked via analytical tensor allocation tracking.

---

## 📊 Quantitative Results

### Detailed Latency & Memory Measurements

| Sequence Length ($N$) | LinearHolographic $O(N)$ | StandardVector $O(N^2)$ | PhaseAttention $O(N^2)$ | TriangularPhase $O(N^2)$ | Speedup (Linear vs Vector) | Memory Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 128$** | 0.23 ms (0.07 MB) | 0.13 ms (0.5 MB) | 0.40 ms (0.5 MB) | 0.53 ms (0.5 MB) | 0.56x | 7.1x |
| **$N = 256$** | 0.33 ms (0.14 MB) | 0.18 ms (2.0 MB) | 0.55 ms (2.0 MB) | 0.64 ms (2.0 MB) | 0.54x | 14.2x |
| **$N = 512$** | **0.36 ms (0.28 MB)** | 0.57 ms (8.0 MB) | 1.66 ms (8.0 MB) | 2.63 ms (8.0 MB) | **1.6x** | **28.5x** |
| **$N = 1,024$** | **0.56 ms (0.56 MB)** | 2.30 ms (32.0 MB) | 8.11 ms (32.0 MB) | 12.26 ms (32.0 MB) | **4.1x** | **56.9x** |
| **$N = 2,048$** | **0.87 ms (1.13 MB)** | 14.05 ms (128.0 MB) | 39.84 ms (128.0 MB) | 62.88 ms (128.0 MB) | **16.2x** | **113.8x** |
| **$N = 4,096$** | **1.81 ms (2.25 MB)** | 59.35 ms (512.0 MB) | 155.33 ms (512.0 MB) | 238.93 ms (512.0 MB) | **32.7x** | **227.6x** |
| **$N = 8,192$** | **3.03 ms (4.50 MB)** | 249.32 ms (2,048 MB) | 615.80 ms (2,048 MB) | 948.95 ms (2,048 MB) | **82.3x** | **455.1x** |

### Empirical Complexity Exponents

Fitting power laws $t(N) \propto N^\alpha$ over empirical measurements:
- **`LinearHolographicPhase`**: $\alpha = 0.62$ (sub-linear to linear regime governed by vectorization and memory bandwidth).
- **`StandardVector Attention`**: $\alpha = 1.94$ (quadratic $O(N^2)$ regime).
- **`PhaseAttention (U(1))`**: $\alpha = 1.88$ (quadratic $O(N^2)$ regime).
- **`TriangularPhase`**: $\alpha = 1.93$ (quadratic $O(N^2)$ regime).

---

## 📈 Visualizations

![Complexity and Scaling Curves](../assets/complexity_scaling.png)

*Figure 1: (Left) Latency vs Sequence Length on log-log scale demonstrating empirical $O(N)$ vs $O(N^2)$ behavior. (Right) Memory footprint scaling across context lengths.*

---

## 💡 Key Findings

1. **Crossover Point:** `LinearHolographicPhaseAttention` outperforms standard vector attention in runtime latency starting at sequence length $N \ge 350$. For longer sequences ($N \ge 1024$), speedups grow dramatically (4x at 1k, 16x at 2k, 82x at 8k).
2. **True $O(1)$ Embedded Memory:** In streaming inference (`step(x_t, state)`), the state footprint is independent of sequence length: exactly $2 \times d_v$ floats per head. A 4-head model requires only 64 floats (256 bytes) of state RAM, allowing infinite context streaming on microcontrollers.
3. **No Softmax Overhead:** Eliminating the row-wise exponentiation and normalization simplifies hardware implementation and eliminates numerical instability/overflow risks.

---

## 🔁 Reproduction Command

```bash
python experiments/benchmark_scaling_latency.py
```
Outputs:
- Raw metrics: `results/benchmark_scaling_latency.json`
- Scaling plot: `assets/complexity_scaling.png`
