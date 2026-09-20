#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT 5: EMBEDDED VISION MICRO-VIT BENCHMARK (FASHION-MNIST)
================================================================================
Evaluates PhaseAttention for TinyML Computer Vision on microcontroller-sized
Vision Transformers (Micro-ViT, <10k parameters):
  - Image input: 28x28 grayscale images.
  - Patch Tokenizer: 4x4 non-overlapping patches -> 49 visual tokens.
  - Target: 10 clothing categories (T-shirt, Trouser, Pullover, Dress, Coat,
            Sandal, Shirt, Sneaker, Bag, Ankle boot).

Models Compared:
  1. LinearHolographicPhaseAttention (Exact O(N), O(1) state memory, No Softmax)
  2. TriangularPhaseAttention (100% Multiplier-Free QxK kernel: Sub + Abs only)
  3. PhaseAttention U(1) (Quadratic circular cosine attention)
  4. StandardVector Attention (Standard PyTorch ViT, O(N^2))
  5. cosFormer Linear Attention (Qin et al., 2022 - SOTA linear attention baseline)
  6. Edge-CNN 2D (Standard embedded convolutional network baseline)

Generates:
  - results/benchmark_micro_vit.json: Exact numerical test metrics
  - assets/micro_vit_benchmark.png: 3-panel publication-ready figure
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
import torchvision
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
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
# 1. BASELINE ARCHITECTURES
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


class CosFormerAttn(nn.Module):
    def __init__(self, d_model=32, num_heads=4, d_v=8, max_len=49):
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


class EdgeCNN2D(nn.Module):
    """Standard 2D Tiny ConvNet for Microcontrollers (CMSIS-NN reference)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class MicroViT(nn.Module):
    """Micro Vision Transformer (<10k parameters) with 4x4 patch tokenization."""
    def __init__(self, attn_module, patch_size=4, in_ch=1, d_model=32, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = in_ch * patch_size * patch_size # 16
        self.patch_proj = nn.Linear(patch_dim, d_model)
        num_patches = (28 // patch_size) ** 2 # 49
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        self.attn = attn_module
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        # Unfold 2D image into sequence of 49 patches (B, 49, 16)
        patches = x.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, -1, C * p * p)
        
        h = self.patch_proj(patches) + self.pos_emb
        h = h + self.attn(self.ln1(h))
        h = h + self.mlp(self.ln2(h))
        out = h.mean(dim=1)
        return self.head(out)


# -----------------------------------------------------------------------------
# 2. TRAINING & EVALUATION
# -----------------------------------------------------------------------------

def train_and_eval_vit(name, model, loader_tr, loader_te, epochs=12, lr=0.003):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()
    
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for bx, by in loader_tr:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            
    train_time = time.time() - t0
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in loader_te:
            preds = model(bx).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(by.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = (all_preds == all_targets).mean() * 100.0
    
    # Compute Macro F1
    f1s = []
    for c in range(10):
        tp = np.sum((all_targets == c) & (all_preds == c))
        fp = np.sum((all_targets != c) & (all_preds == c))
        fn = np.sum((all_targets == c) & (all_preds != c))
        p = tp / max(tp + fp, 1e-8)
        r = tp / max(tp + fn, 1e-8)
        f1 = 2 * (p * r) / max(p + r, 1e-8)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) * 100.0
    
    print(f"  [{name:30s}] Acc: {acc:5.2f}% | Macro F1: {macro_f1:5.2f}% | Params: {n_params:5d} | Time: {train_time:.1f}s")
    return {
        "name": name,
        "accuracy": round(float(acc), 2),
        "macro_f1": round(float(macro_f1), 2),
        "params": n_params,
        "train_time": round(float(train_time), 2)
    }


# -----------------------------------------------------------------------------
# 3. PLOTTING
# -----------------------------------------------------------------------------

def plot_micro_vit_results(results, sample_img, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    
    # -------------------------------------------------------------------------
    # Panel 1: Patch Tokenization Decomposition
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.imshow(sample_img, cmap="inferno")
    # Draw 4x4 patch grid lines
    for i in range(0, 29, 4):
        ax1.axhline(i - 0.5, color="cyan", linewidth=1.0, alpha=0.7)
        ax1.axvline(i - 0.5, color="cyan", linewidth=1.0, alpha=0.7)
        
    ax1.set_title("A. Micro-ViT Patch Tokenization\n(28x28 -> 49 Tokens of 4x4 Patches)", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(0, 29, 7))
    ax1.set_yticks(range(0, 29, 7))
    ax1.grid(False)

    # -------------------------------------------------------------------------
    # Panel 2: Accuracy & Macro F1 Bar Chart
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    clean_names = [
        "Linear-Holo\n[O(N)]",
        "Triangular\n[No-Mult]",
        "PhaseAttn\n[U(1)]",
        "Standard\n[Softmax]",
        "cosFormer\n[Linear]",
        "Edge-CNN\n[Conv]"
    ]
    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    
    x = np.arange(len(clean_names))
    w = 0.35
    b1 = ax2.bar(x - w/2, accs, width=w, label="Top-1 Accuracy (%)", color="#2980b9", edgecolor="black")
    b2 = ax2.bar(x + w/2, f1s, width=w, label="Macro F1 (%)", color="#e67e22", edgecolor="black")
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(clean_names, fontsize=9.5)
    ax2.set_ylim(70, 88)
    ax2.set_title("B. Classification Performance (Fashion-MNIST 10-Class)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Score (%)", fontsize=11)
    ax2.legend(loc="lower right", fontsize=9.5)
    
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 3: Parameter Efficiency vs Accuracy
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    colors = ["#f39c12", "#2ecc71", "#3498db", "#34495e", "#9b59b6", "#e74c3c"]
    
    labels = [
        "Linear-Holo (9.1k)",
        "Triangular (9.1k)",
        "PhaseAttn (9.1k)",
        "Standard (11.0k)",
        "cosFormer (11.0k)",
        "Edge-CNN (5.2k)"
    ]
    offsets = [
        (8, -6),     # Linear-Holo
        (8, 6),      # Triangular
        (-95, -6),   # PhaseAttn
        (8, -6),     # Standard
        (-95, 6),    # cosFormer
        (8, -3),     # Edge-CNN
    ]
    
    for i, r in enumerate(results):
        ax3.scatter(r["accuracy"], r["params"], s=160, color=colors[i], edgecolors="black", linewidth=1.5, zorder=3)
        lbl = labels[i] if i < len(labels) else r["name"].split()[0]
        off = offsets[i] if i < len(offsets) else (5, 5)
        ax3.annotate(lbl, xy=(r["accuracy"], r["params"]), xytext=off,
                     textcoords="offset points", fontsize=9, fontweight="bold")

    ax3.set_title("C. Model Size vs Vision Accuracy", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Top-1 Accuracy (%)", fontsize=11)
    ax3.set_ylabel("Total Parameters", fontsize=11)
    ax3.set_xlim(76.5, 83.5)
    ax3.set_ylim(4500, 12000)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[+] Publication-quality Micro-ViT plot saved to: {output_path}")


# -----------------------------------------------------------------------------
# 4. MAIN PIPELINE
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Micro-ViT Embedded Vision Benchmark")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_samples", type=int, default=6000, help="Train samples subset")
    parser.add_argument("--test_samples", type=int, default=1500, help="Test samples subset")
    parser.add_argument("--plot_only", action="store_true", help="Only regenerate plot from cached results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("PHASEATTENTION: EMBEDDED VISION MICRO-VIT BENCHMARK (FASHION-MNIST)")
    print("=" * 80)
    print(f"Platform: Python {sys.version.split()[0]} | PyTorch {torch.__version__} | Seed {args.seed}")

    # Dataset location
    data_dir = Path("c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/data")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    ds_tr = FashionMNIST(root=str(data_dir), train=True, download=False, transform=transform)
    ds_te = FashionMNIST(root=str(data_dir), train=False, download=False, transform=transform)

    g = torch.Generator().manual_seed(args.seed)
    idx_tr = torch.randperm(len(ds_tr), generator=g)[:args.train_samples]
    idx_te = torch.randperm(len(ds_te), generator=g)[:args.test_samples]

    loader_tr = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_tr, idx_tr), batch_size=args.batch_size, shuffle=True)
    loader_te = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_te, idx_te), batch_size=128, shuffle=False)

    print(f"Data Splits: Train = {len(idx_tr)} images, Test = {len(idx_te)} images")
    print(f"Patch Configuration: 4x4 patches -> 49 visual tokens per 28x28 image\n")

    # Sample image for visualization
    sample_img = ds_te[0][0].squeeze().numpy()

    res_dir = REPO_ROOT / "results"
    assets_dir = REPO_ROOT / "assets"
    res_dir.mkdir(exist_ok=True)
    assets_dir.mkdir(exist_ok=True)
    json_path = res_dir / "benchmark_micro_vit.json"
    plot_path = assets_dir / "micro_vit_benchmark.png"

    if args.plot_only and json_path.exists():
        print(f"Loading existing metrics from {json_path} for plotting...")
        with open(json_path, "r") as f:
            payload = json.load(f)
        plot_micro_vit_results(payload["results"], sample_img, plot_path)
        print("Done!")
        return

    # Build model suite
    d_model = 32
    num_heads = 4
    d_v = 8
    
    models = [
        ("LinearHolographic (O(N))", MicroViT(LinearHolographicPhaseAttention(d_model=d_model, num_heads=num_heads, d_v=d_v))),
        ("Triangular (Multiplier-Free)", MicroViT(TriangularPhaseAttention(d_model=d_model, num_heads=num_heads, d_v=d_v))),
        ("PhaseAttention (U(1))", MicroViT(PhaseAttention(d_model=d_model, num_heads=num_heads, d_v=d_v))),
        ("StandardVector Attention", MicroViT(StandardVectorAttn(d_model=d_model, num_heads=num_heads, d_k=8, d_v=d_v))),
        ("cosFormer (Qin 2022)", MicroViT(CosFormerAttn(d_model=d_model, num_heads=num_heads, d_v=d_v))),
        ("Edge-CNN 2D (Baseline)", EdgeCNN2D())
    ]

    print("--- Training and Evaluating 6 Architectures on Micro-ViT ---")
    results = []
    for name, model in models:
        res = train_and_eval_vit(name, model, loader_tr, loader_te, epochs=args.epochs)
        results.append(res)

    # Save results to JSON
    res_dir = REPO_ROOT / "results"
    assets_dir = REPO_ROOT / "assets"
    res_dir.mkdir(exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    json_path = res_dir / "benchmark_micro_vit.json"
    plot_path = assets_dir / "micro_vit_benchmark.png"

    payload = {
        "dataset": "Fashion-MNIST",
        "input_resolution": [28, 28],
        "patch_size": [4, 4],
        "num_tokens": 49,
        "classes": 10,
        "sample_counts": {"train": len(idx_tr), "test": len(idx_te)},
        "results": results
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Raw metrics saved to: {json_path}")

    # Plot publication figure
    plot_micro_vit_results(results, sample_img, plot_path)

    print("\n" + "=" * 80)
    print("MICRO-VIT BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
