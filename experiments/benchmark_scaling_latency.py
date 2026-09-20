#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT 2: COMPLEXITY & CONTEXT LENGTH SCALING BENCHMARK
================================================================================
Empirical validation of O(N) linear time and O(1) state memory complexity:
  1. Compares LinearHolographicPhaseAttention against Quadratic Attention models
     (PhaseAttention, TriangularPhase, Standard Multi-Head Attention).
  2. Evaluates context lengths from N = 128 to N = 8192.
  3. Measures latency (ms), activation memory footprint (MB), and speedup ratio.
  4. Fits empirical power-law scaling exponents (T ~ N^alpha).

Generates:
  - results/benchmark_scaling_latency.json
  - assets/complexity_scaling.png (3-panel publication-ready figure)
================================================================================
"""

import os
import sys
import time
import json
import math
import gc
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Ensure local package is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from phase_attention import (
    PhaseAttention,
    LinearHolographicPhaseAttention,
    TriangularPhaseAttention
)

# Standard Vector Multi-Head Attention baseline for scaling comparison
class StandardPyTorchAttention(nn.Module):
    def __init__(self, d_model: int = 32, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


def calculate_activation_memory_mb(model_type: str, B: int, L: int, H: int = 4, d_v: int = 8) -> float:
    """
    Computes analytical peak memory required by intermediate attention representations.
    - Quadratic: attention scores + softmax weights (2 * B * H * L * L * 4 bytes)
    - Linear Holographic: phi_k, phi_q, kv_pairs, state (B * H * L * (2 + 2 + 2*d_v + 2*d_v) * 4 bytes)
    """
    bytes_per_float = 4.0
    if model_type in ["quadratic_phase", "triangular_phase", "standard_vector"]:
        # Q @ K^T scores matrix + Softmax probabilities matrix
        elements = 2.0 * (B * H * L * L)
    elif model_type == "linear_holographic":
        # Cumulative prefix memory state (B, H, L, 2, d_v) + phi pairs
        elements = B * H * L * (2 * d_v) + B * H * L * (2 * d_v) + 2 * (B * H * L * 2)
    else:
        elements = B * H * L * L
    return (elements * bytes_per_float) / (1024.0 * 1024.0)


def benchmark_model_latency(model: nn.Module, x: torch.Tensor, warmup: int = 5, reps: int = 20) -> float:
    """Measures average forward execution time in milliseconds."""
    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(warmup):
            _ = model(x)
            
        # Timed runs
        timings = []
        for _ in range(reps):
            gc.disable()
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            gc.enable()
            timings.append((t1 - t0) * 1000.0) # Convert to ms
            
    # Remove outliers (trimmed mean)
    timings.sort()
    trimmed = timings[1:-1] if len(timings) > 4 else timings
    return float(np.mean(trimmed))


def plot_scaling_benchmark(results, lengths, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    
    colors = {
        "LinearHolographic O(N)": "#f39c12",   # Orange / Gold
        "PhaseAttention O(N^2)": "#3498db",    # Blue
        "TriangularPhase O(N^2)": "#2ecc71",   # Green
        "StandardVector O(N^2)": "#34495e"     # Dark Slate
    }
    markers = {
        "LinearHolographic O(N)": "o",
        "PhaseAttention O(N^2)": "s",
        "TriangularPhase O(N^2)": "^",
        "StandardVector O(N^2)": "d"
    }

    # -------------------------------------------------------------------------
    # Panel 1: Latency vs Sequence Length (Log-Log)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    for model_name, data in results.items():
        latencies = data["latency_ms"]
        ax1.plot(lengths, latencies, label=model_name,
                 color=colors.get(model_name, "#7f8c8d"),
                 marker=markers.get(model_name, "o"),
                 linewidth=2.2, markersize=6)
        
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(lengths)
    ax1.set_xticklabels([str(L) for L in lengths], fontsize=9)
    ax1.set_title("A. Inference Latency (Log-Log Scale)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Sequence Length (N tokens)", fontsize=11)
    ax1.set_ylabel("Latency per Forward Pass (ms)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

    # -------------------------------------------------------------------------
    # Panel 2: Intermediate Memory Footprint (Log-Log)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    for model_name, data in results.items():
        memories = data["memory_mb"]
        ax2.plot(lengths, memories, label=model_name,
                 color=colors.get(model_name, "#7f8c8d"),
                 marker=markers.get(model_name, "o"),
                 linewidth=2.2, markersize=6)
        
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(lengths)
    ax2.set_xticklabels([str(L) for L in lengths], fontsize=9)
    ax2.set_title("B. Attention Matrix / State Memory (MB)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Sequence Length (N tokens)", fontsize=11)
    ax2.set_ylabel("Intermediate Memory (MB)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

    # -------------------------------------------------------------------------
    # Panel 3: Speedup Factor (LinearHolographic vs Standard Attention)
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    lin_latencies = np.array(results["LinearHolographic O(N)"]["latency_ms"])
    quad_latencies = np.array(results["PhaseAttention O(N^2)"]["latency_ms"])
    std_latencies = np.array(results["StandardVector O(N^2)"]["latency_ms"])
    
    speedup_quad = quad_latencies / lin_latencies
    speedup_std = std_latencies / lin_latencies
    
    x = np.arange(len(lengths))
    w = 0.35
    b1 = ax3.bar(x - w/2, speedup_quad, width=w, label="vs PhaseAttention O(N^2)", color="#3498db", edgecolor="black")
    b2 = ax3.bar(x + w/2, speedup_std, width=w, label="vs StandardVector O(N^2)", color="#34495e", edgecolor="black")
    
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(L) for L in lengths], fontsize=9)
    ax3.set_title("C. Speedup Factor of Linear Holographic", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Sequence Length (N tokens)", fontsize=11)
    ax3.set_ylabel("Speedup Ratio (x times faster)", fontsize=11)
    ax3.legend(loc="upper left", fontsize=9.5)
    
    # Annotate top bar values on longer sequences
    for i, (sq, ss) in enumerate(zip(speedup_quad, speedup_std)):
        if lengths[i] >= 1024:
            ax3.annotate(f"{sq:.0f}x", xy=(x[i] - w/2, sq), xytext=(0, 3),
                         textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight="bold")
            ax3.annotate(f"{ss:.0f}x", xy=(x[i] + w/2, ss), xytext=(0, 3),
                         textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[+] Publication-quality scaling plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PhaseAttention Complexity & Scaling Benchmark")
    parser.add_argument("--d_model", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_v", type=int, default=8, help="Value dimension per head")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    args = parser.parse_args()

    # Sequence lengths to evaluate
    lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    
    print("=" * 80)
    print("EXPERIMENT 2: COMPUTATIONAL COMPLEXITY & CONTEXT LENGTH SCALING")
    print("=" * 80)
    print(f"Platform: Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CPU Execution")
    print(f"Config: d_model={args.d_model}, num_heads={args.num_heads}, d_v={args.d_v}, Batch={args.batch_size}")
    print(f"Sequence Lengths (N): {lengths}\n")

    models = {
        "LinearHolographic O(N)": (
            LinearHolographicPhaseAttention(d_model=args.d_model, num_heads=args.num_heads, d_v=args.d_v),
            "linear_holographic"
        ),
        "PhaseAttention O(N^2)": (
            PhaseAttention(d_model=args.d_model, num_heads=args.num_heads, d_v=args.d_v),
            "quadratic_phase"
        ),
        "TriangularPhase O(N^2)": (
            TriangularPhaseAttention(d_model=args.d_model, num_heads=args.num_heads, d_v=args.d_v),
            "triangular_phase"
        ),
        "StandardVector O(N^2)": (
            StandardPyTorchAttention(d_model=args.d_model, num_heads=args.num_heads),
            "standard_vector"
        )
    }

    results = {
        name: {"latency_ms": [], "memory_mb": []}
        for name in models.keys()
    }

    for L in lengths:
        print(f"--- Benchmarking Sequence Length N = {L:4d} ---")
        x = torch.randn(args.batch_size, L, args.d_model)
        
        # Adaptive repetitions: fewer reps for very large sequences to avoid long runtime
        reps = 30 if L <= 1024 else (15 if L <= 4096 else 8)
        
        for name, (model, mtype) in models.items():
            latency = benchmark_model_latency(model, x, warmup=3, reps=reps)
            mem_mb = calculate_activation_memory_mb(mtype, args.batch_size, L, args.num_heads, args.d_v)
            
            results[name]["latency_ms"].append(round(latency, 3))
            results[name]["memory_mb"].append(round(mem_mb, 4))
            print(f"  [{name:24s}] Latency: {latency:8.2f} ms | Attention Memory: {mem_mb:8.3f} MB")
        print()

    # Calculate empirical power law exponent alpha: Latency ~ N^alpha
    print("--- Empirical Scaling Exponent (T ~ N^alpha) ---")
    log_N = np.log(lengths)
    for name in models.keys():
        log_T = np.log(results[name]["latency_ms"])
        slope, _ = np.polyfit(log_N, log_T, 1)
        results[name]["empirical_scaling_alpha"] = round(float(slope), 2)
        print(f"  [{name:24s}] alpha = {slope:.2f} (Expected: ~1.0 for linear, ~2.0 for quadratic)")

    # Save results to JSON
    res_dir = REPO_ROOT / "results"
    assets_dir = REPO_ROOT / "assets"
    res_dir.mkdir(exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    json_path = res_dir / "benchmark_scaling_latency.json"
    plot_path = assets_dir / "complexity_scaling.png"

    payload = {
        "sequence_lengths": lengths,
        "config": {
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "d_v": args.d_v,
            "batch_size": args.batch_size
        },
        "results": results
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Raw metrics saved to: {json_path}")

    # Plot figure
    plot_scaling_benchmark(results, lengths, plot_path)
    
    print("\n" + "=" * 80)
    print("SCALING BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
