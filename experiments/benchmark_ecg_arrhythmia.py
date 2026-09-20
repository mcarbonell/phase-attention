#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT 4: REAL-WORLD MIT-BIH ECG ARRHYTHMIA CLASSIFICATION
================================================================================
Evaluates PhaseAttention on real clinical patient ECG recordings from the
official PhysioNet MIT-BIH Arrhythmia Database (Harvard-MIT Health Sciences).

AAMI EC57 Clinical Classes:
  - Class 0: Normal Beat (N)
  - Class 1: Supraventricular Ectopic Beat (S) [Atrial premature beat, PAC]
  - Class 2: Ventricular Ectopic Beat (V) [Premature ventricular contraction, PVC]

Models Compared:
  1. LinearHolographicPhaseAttention (Exact O(N), O(1) state memory, No Softmax)
  2. TriangularPhaseAttention (100% Multiplier-Free QxK kernel: Sub + Abs only)
  3. PhaseAttention U(1) (Quadratic circular cosine attention)
  4. StandardVector Attention (Standard PyTorch Transformer, O(N^2))
  5. cosFormer Linear Attention (Qin et al., 2022 - SOTA linear attention baseline)
  6. Edge-CNN 1D (Standard microcontroller convolutional baseline)

Generates:
  - results/benchmark_ecg_arrhythmia.json: Exact numerical test metrics
  - assets/ecg_arrhythmia_benchmark.png: 3-panel publication-ready figure
================================================================================
"""

import os
import sys
import time
import json
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# -----------------------------------------------------------------------------
# 1. BASELINE ARCHITECTURES (Standard Transformer, cosFormer, Edge-CNN)
# -----------------------------------------------------------------------------

class StandardVectorAttn(nn.Module):
    def __init__(self, d_model=32, num_heads=4, d_k=8, d_v=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, num_heads * d_k)
        self.w_k = nn.Linear(d_model, num_heads * d_k)
        self.w_v = nn.Linear(d_model, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, d_model)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.w_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        v = self.w_v(x).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out)


class CosFormerLinearAttn(nn.Module):
    """cosFormer Linear Attention baseline (Qin et al., 2022)."""
    def __init__(self, d_model=32, num_heads=4, d_v=8, max_len=18):
        super().__init__()
        self.num_heads = num_heads
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, num_heads * d_v)
        self.w_k = nn.Linear(d_model, num_heads * d_v)
        self.w_v = nn.Linear(d_model, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, d_model)
        
        angles = torch.arange(max_len).float() * (math.pi / (2 * max_len))
        self.register_buffer('cos_weight', torch.cos(angles))
        self.register_buffer('sin_weight', torch.sin(angles))

    def forward(self, x):
        B, L, _ = x.shape
        q = F.relu(self.w_q(x)).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        k = F.relu(self.w_k(x)).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        
        cw = self.cos_weight[:L].view(1, 1, L, 1)
        sw = self.sin_weight[:L].view(1, 1, L, 1)
        
        q_c, q_s = q * cw, q * sw
        k_c, k_s = k * cw, k * sw
        
        kv_c = torch.matmul(k_c.transpose(-2, -1), v)
        kv_s = torch.matmul(k_s.transpose(-2, -1), v)
        
        out = torch.matmul(q_c, kv_c) + torch.matmul(q_s, kv_s)
        denom = torch.matmul(q_c, k_c.sum(dim=-2, keepdim=True).transpose(-2, -1)) + \
                torch.matmul(q_s, k_s.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6
        out = out / denom
        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out)


class EdgeCNN1D(nn.Module):
    """Standard 1D Convolutional Neural Network baseline for Microcontrollers."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


# Transformer wrapper with 1D patch tokenizer for time-series ECG
class ECGTransformerClassifier(nn.Module):
    def __init__(self, attn_module, d_model=32, patch_len=10, num_classes=3):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 180 // patch_len, d_model) * 0.02)
        self.attn = attn_module
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.shape[0]
        # Tokenize 180-sample ECG signal into 18 patches of length 10
        patches = x.view(B, -1, self.patch_len)
        h = self.patch_proj(patches) + self.pos_emb
        h = h + self.attn(self.ln(h))
        h = h + self.mlp(h)
        # Global pooling across token sequence
        out = h.mean(dim=1)
        return self.head(out)


# -----------------------------------------------------------------------------
# 2. METRICS & TRAINING HARNESS
# -----------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, num_classes=3):
    """Computes Overall Accuracy, Macro F1-Score, and per-class Sensitivities."""
    acc = (y_true == y_pred).mean() * 100.0
    f1_scores = []
    sensitivities = []
    
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        prec = tp / max(tp + fp, 1e-8)
        rec = tp / max(tp + fn, 1e-8)
        f1 = 2 * (prec * rec) / max(prec + rec, 1e-8)
        
        f1_scores.append(f1 * 100.0)
        sensitivities.append(rec * 100.0)
        
    return {
        "accuracy": float(acc),
        "macro_f1": float(np.mean(f1_scores)),
        "sens_normal": float(sensitivities[0]),
        "sens_supraventricular": float(sensitivities[1]),
        "sens_ventricular": float(sensitivities[2])
    }


def train_and_eval_ecg(name, model, X_tr, y_tr, X_te, y_te, epochs=15, batch_size=64, lr=0.003):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()
    
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y_tr))
        for b in range(0, len(y_tr), batch_size):
            idx = perm[b : b + batch_size]
            opt.zero_grad()
            logits = model(X_tr[idx])
            loss = crit(logits, y_tr[idx])
            loss.backward()
            opt.step()
            
    train_time = time.time() - t0
    model.eval()
    with torch.no_grad():
        preds = model(X_te).argmax(dim=-1).cpu().numpy()
        targets = y_te.cpu().numpy()
        
    m = compute_metrics(targets, preds)
    m["name"] = name
    m["params"] = n_params
    m["train_time"] = round(train_time, 2)
    
    print(f"  [{name:30s}] Acc: {m['accuracy']:5.2f}% | Macro F1: {m['macro_f1']:5.2f}% | V-Sens: {m['sens_ventricular']:5.2f}% | Params: {n_params}")
    return m


# -----------------------------------------------------------------------------
# 3. PLOTTING FUNCTION
# -----------------------------------------------------------------------------

def plot_ecg_results(results, sample_beats, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    
    # -------------------------------------------------------------------------
    # Panel 1: Real ECG Morphologies (MIT-BIH Waveforms)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    time_ms = np.linspace(-250, 250, 180) # 180 samples at 360 Hz = 500 ms
    
    ax1.plot(time_ms, sample_beats[0], label="Normal (N) [P-QRS-T]", color="#2ecc71", linewidth=2.0)
    ax1.plot(time_ms, sample_beats[1], label="Supraventricular (S) [PAC]", color="#3498db", linewidth=2.0, linestyle="--")
    ax1.plot(time_ms, sample_beats[2], label="Ventricular (V) [Wide PVC]", color="#e74c3c", linewidth=2.2)
    
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.6, label="R-Peak Reference")
    ax1.set_title("A. Real Clinical ECG Beats (MIT-BIH Lead MLII)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Time relative to R-peak (ms)", fontsize=11)
    ax1.set_ylabel("Normalized Amplitude (Z-score)", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9.5)

    # -------------------------------------------------------------------------
    # Panel 2: Accuracy & Macro F1 Comparison
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    names = [r["name"].replace(" Attention", "").replace(" (O(N))", "\n[O(N)]").replace(" (Multiplier-Free)", "\n[No-Mult]").replace(" (Qin 2022)", "\n[cosFormer]").replace(" (Baseline)", "\n[CNN]") for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    
    x = np.arange(len(names))
    w = 0.35
    b1 = ax2.bar(x - w/2, accs, width=w, label="Accuracy (%)", color="#34495e", edgecolor="black")
    b2 = ax2.bar(x + w/2, f1s, width=w, label="Macro F1 (%)", color="#f39c12", edgecolor="black")
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8.5)
    ax2.set_ylim(80, 102)
    ax2.set_title("B. Diagnostic Performance (MIT-BIH 3-Class)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Score (%)", fontsize=11)
    ax2.legend(loc="lower right", fontsize=9.5)
    
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 3: Multipliers in QxK vs Macro F1 Score
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    # Model multipliers per forward token interaction:
    # LinearHolographic: 0 multipliers in affinity (exact separable sum)
    # Triangular: 0 multipliers in affinity (sub + abs only)
    # PhaseAttention: 0 multipliers in affinity (subtraction only)
    # cosFormer: 4*d_v multipliers in decomposition
    # StandardVector: N * d_k multipliers
    
    mults = [0, 0, 0, 18 * 8, 32, 0] # for the 6 models
    scatter_colors = ["#f39c12", "#2ecc71", "#3498db", "#34495e", "#9b59b6", "#e74c3c"]
    
    for i, r in enumerate(results):
        ax3.scatter(r["macro_f1"], r["params"], s=150, color=scatter_colors[i], edgecolors="black", linewidth=1.5, zorder=3)
        ax3.annotate(r["name"].split()[0], xy=(r["macro_f1"], r["params"]), xytext=(5, 5),
                     textcoords="offset points", fontsize=9, fontweight="bold")

    ax3.set_title("C. Parameter Efficiency vs Macro F1", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Macro F1-Score (%)", fontsize=11)
    ax3.set_ylabel("Total Parameters", fontsize=11)
    ax3.set_xlim(88, 98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[+] Publication-quality clinical benchmark plot saved to: {output_path}")


# -----------------------------------------------------------------------------
# 4. MAIN PIPELINE
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MIT-BIH ECG Arrhythmia Benchmark")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("PHASEATTENTION: REAL-WORLD MIT-BIH ECG ARRHYTHMIA BENCHMARK")
    print("=" * 80)
    print(f"Platform: Python {sys.version.split()[0]} | PyTorch {torch.__version__} | Seed {args.seed}\n")

    # Load dataset
    data_path = REPO_ROOT / "data" / "mitbih_arrhythmia_balanced.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run dataset generation first.")
        
    data = torch.load(data_path, weights_only=True)
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"], data["y_test"]
    class_names = data["class_names"]

    print(f"Dataset Loaded: Train = {len(y_tr)} beats, Test = {len(y_te)} beats (180 samples @ 360 Hz)")
    print(f"Classes: {class_names} (Balanced ~33.3% each, Chance = 33.33%)\n")

    # Sample representative beats for plotting
    sample_beats = []
    for c in range(3):
        idx = (y_te == c).nonzero()[0].item()
        sample_beats.append(X_te[idx].numpy())

    # Build model suite
    d_model = 32
    num_heads = 4
    d_v = 8
    
    models = [
        ("LinearHolographic (O(N))", ECGTransformerClassifier(LinearHolographicPhaseAttention(d_model=d_model, num_heads=num_heads, d_v=d_v))),
        ("Triangular (Multiplier-Free)", ECGTransformerClassifier(TriangularPhaseAttention(d_model=d_model, num_heads=num_heads, d_v=d_v))),
        ("PhaseAttention (U(1))", ECGTransformerClassifier(PhaseAttention(d_model=d_model, num_heads=num_heads, d_v=d_v))),
        ("StandardVector Attention", ECGTransformerClassifier(StandardVectorAttn(d_model=d_model, num_heads=num_heads, d_k=8, d_v=d_v))),
        ("cosFormer (Qin 2022)", ECGTransformerClassifier(CosFormerLinearAttn(d_model=d_model, num_heads=num_heads, d_v=d_v))),
        ("Edge-CNN 1D (Baseline)", EdgeCNN1D(num_classes=3))
    ]

    print("--- Training and Evaluating 6 Architectures on MIT-BIH ---")
    results = []
    for name, model in models:
        res = train_and_eval_ecg(name, model, X_tr, y_tr, X_te, y_te, epochs=args.epochs, batch_size=args.batch_size)
        results.append(res)

    # Save results to JSON
    res_dir = REPO_ROOT / "results"
    assets_dir = REPO_ROOT / "assets"
    res_dir.mkdir(exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    json_path = res_dir / "benchmark_ecg_arrhythmia.json"
    plot_path = assets_dir / "ecg_arrhythmia_benchmark.png"

    payload = {
        "dataset": "MIT-BIH Arrhythmia Database (PhysioNet)",
        "lead": "MLII",
        "sampling_rate_hz": 360,
        "classes": class_names,
        "sample_counts": {"train": len(y_tr), "test": len(y_te)},
        "results": results
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Raw metrics saved to: {json_path}")

    # Plot publication figure
    plot_ecg_results(results, sample_beats, plot_path)
    
    print("\n" + "=" * 80)
    print("CLINICAL ECG BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
