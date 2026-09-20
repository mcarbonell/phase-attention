#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT 6: CAUSAL LANGUAGE MODELING ON TINYSHAKESPEARE
================================================================================
Evaluates PhaseAttention for autoregressive character-level sequence modeling:
  - Dataset: TinyShakespeare (1.1 MB text corpus, 65 character vocabulary)
  - Sequence length: L = 64 tokens (causal autoregressive context)
  - Model scale: Ultra-lightweight TinyML LM (d_model=64, 4 heads, d_v=16)

Models Compared:
  1. LinearHolographicPhaseAttention (Exact causal O(N) scan, O(1) state, No Softmax)
  2. TriangularPhaseAttention (100% Multiplier-Free causal QxK attention)
  3. PhaseAttention U(1) (Causal quadratic circular cosine attention)
  4. StandardCausal Attention (Standard PyTorch causal Transformer MHA, O(N^2))
  5. cosFormer Causal (Qin et al., 2022 - SOTA linear causal attention)
  6. Edge-GRU (Standard sequential gated recurrent unit baseline for microcontrollers)

Metrics:
  - Validation Cross-Entropy Loss
  - Validation Perplexity (PPL = exp(loss))
  - Bits-Per-Character (BPC = loss / ln(2))
  - Parameter Count
  - Sample Autoregressive Text Generation

Outputs:
  - results/benchmark_tinyshakespeare_lm.json
  - assets/tinyshakespeare_benchmark.png
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
# 1. BASELINE ARCHITECTURES
# -----------------------------------------------------------------------------

class StandardCausalAttn(nn.Module):
    """Standard PyTorch Causal Multi-Head Attention (O(N^2))."""
    def __init__(self, d_model=64, num_heads=4, d_k=16, d_v=16):
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
        v = self.w_v(x).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out)


class CausalCosFormerAttn(nn.Module):
    """Causal cosFormer Linear Attention (Qin et al., 2022)."""
    def __init__(self, d_model=64, num_heads=4, d_v=16, max_len=128):
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
        
        # Causal prefix sums
        kv_c = torch.cumsum(k_c.unsqueeze(-1) * v.unsqueeze(-2), dim=2)
        kv_s = torch.cumsum(k_s.unsqueeze(-1) * v.unsqueeze(-2), dim=2)
        out = (q_c.unsqueeze(-2) @ kv_c).squeeze(-2) + (q_s.unsqueeze(-2) @ kv_s).squeeze(-2)
        
        k_c_sum = torch.cumsum(k_c, dim=2)
        k_s_sum = torch.cumsum(k_s, dim=2)
        denom = (q_c * k_c_sum).sum(dim=-1, keepdim=True) + (q_s * k_s_sum).sum(dim=-1, keepdim=True) + 1e-6
        out = out / denom
        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out)


class CausalTransformerLM(nn.Module):
    """Causal Language Model with 1 Transformer Decoder Block."""
    def __init__(self, attn_module, vocab_size=65, d_model=64, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn_module
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, L = idx.shape
        h = self.tok_emb(idx) + self.pos_emb[:, :L, :]
        h = h + self.attn(self.ln1(h))
        h = h + self.mlp(self.ln2(h))
        h = self.ln_f(h)
        return self.head(h)


class EdgeGRULM(nn.Module):
    """Standard Edge Gated Recurrent Unit Language Model."""
    def __init__(self, vocab_size=65, d_model=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        h = self.tok_emb(idx)
        out, _ = self.gru(h)
        out = self.ln_f(out)
        return self.head(out)


# -----------------------------------------------------------------------------
# 2. DATASET & TOKENIZER
# -----------------------------------------------------------------------------

def load_tinyshakespeare(data_path: Path):
    if not data_path.exists():
        # Fallback to attention-neuron location
        alt_path = Path("c:/Users/mrcm_/Local/proj/algorithms/attention-neuron/data/tinyshakespeare.txt")
        if alt_path.exists():
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot locate tinyshakespeare.txt at {data_path} or {alt_path}")
            
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    ch2i = {ch: i for i, ch in enumerate(chars)}
    i2ch = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([ch2i[c] for c in text], dtype=torch.long)
    
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size, ch2i, i2ch


def get_batch(data, batch_size, seq_len):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y


# -----------------------------------------------------------------------------
# 3. AUTOREGRESSIVE TEXT SAMPLING
# -----------------------------------------------------------------------------

def generate_sample(model, prompt_text, ch2i, i2ch, max_new_tokens=80, temperature=0.8):
    model.eval()
    tokens = [ch2i.get(c, 0) for c in prompt_text]
    idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -64:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
            
    out_tokens = idx[0].tolist()
    return "".join([i2ch[i] for i in out_tokens])


# -----------------------------------------------------------------------------
# 4. TRAINING & EVALUATION
# -----------------------------------------------------------------------------

def train_and_eval_lm(name, model, train_data, val_data, steps=1000, batch_size=32, seq_len=64, lr=0.003):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()
    
    t0 = time.time()
    loss_history = []
    
    for step in range(1, steps + 1):
        model.train()
        bx, by = get_batch(train_data, batch_size, seq_len)
        opt.zero_grad()
        logits = model(bx)
        loss = crit(logits.view(-1, logits.size(-1)), by.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if step % 200 == 0 or step == steps:
            loss_history.append(float(loss.item()))
            
    train_time = time.time() - t0
    
    # Validation evaluation over 50 batches
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(50):
            bx, by = get_batch(val_data, batch_size, seq_len)
            logits = model(bx)
            loss = crit(logits.view(-1, logits.size(-1)), by.view(-1))
            val_losses.append(loss.item())
            
    mean_val_loss = float(np.mean(val_losses))
    ppl = float(math.exp(mean_val_loss))
    bpc = float(mean_val_loss / math.log(2.0))
    
    print(f"  [{name:30s}] Val Loss: {mean_val_loss:5.3f} | PPL: {ppl:5.2f} | BPC: {bpc:4.2f} | Params: {n_params:5d} | Time: {train_time:4.1f}s")
    
    return {
        "name": name,
        "val_loss": round(mean_val_loss, 3),
        "perplexity": round(ppl, 2),
        "bpc": round(bpc, 2),
        "params": n_params,
        "train_time": round(train_time, 2)
    }


# -----------------------------------------------------------------------------
# 5. VISUALIZATION
# -----------------------------------------------------------------------------

def plot_tinyshakespeare_results(results, sample_generations, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    
    clean_names = [
        "Linear-Holo\n[O(N)]",
        "Triangular\n[No-Mult]",
        "PhaseAttn\n[U(1)]",
        "Standard\n[Softmax]",
        "cosFormer\n[Linear]",
        "Edge-GRU\n[Recurrent]"
    ]
    
    # -------------------------------------------------------------------------
    # Panel 1: Perplexity Comparison
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ppls = [r["perplexity"] for r in results]
    bpcs = [r["bpc"] for r in results]
    
    x = np.arange(len(clean_names))
    colors = ["#2980b9", "#27ae60", "#8e44ad", "#2c3e50", "#d35400", "#c0392b"]
    bars = ax1.bar(x, ppls, width=0.55, color=colors, edgecolor="black", linewidth=1.2)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_names, fontsize=9.5)
    ax1.set_ylabel("Perplexity (Lower is Better)", fontsize=11)
    ax1.set_title("A. Autoregressive Language Modeling (Perplexity)", fontsize=12, fontweight="bold")
    
    min_ppl = min(ppls)
    max_ppl = max(ppls)
    ax1.set_ylim(min_ppl * 0.85, max_ppl * 1.15)
    
    for b, bpc in zip(bars, bpcs):
        h = b.get_height()
        ax1.annotate(f"{h:.1f}\n({bpc:.2f} bpc)", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 2: Parameter Efficiency vs Perplexity
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    labels = [
        "Linear-Holo (42k)",
        "Triangular (42k)",
        "PhaseAttn (42k)",
        "Standard (50k)",
        "cosFormer (50k)",
        "Edge-GRU (33k)"
    ]
    offsets = [
        (8, -6),     # Linear-Holo
        (8, 6),      # Triangular
        (-95, -6),   # PhaseAttn
        (8, -6),     # Standard
        (8, 6),      # cosFormer
        (8, -3),     # Edge-GRU
    ]
    
    for i, r in enumerate(results):
        ax2.scatter(r["perplexity"], r["params"], s=160, color=colors[i], edgecolors="black", linewidth=1.5, zorder=3)
        lbl = labels[i] if i < len(labels) else r["name"]
        off = offsets[i] if i < len(offsets) else (5, 5)
        ax2.annotate(lbl, xy=(r["perplexity"], r["params"]), xytext=off,
                     textcoords="offset points", fontsize=9, fontweight="bold")
                     
    ax2.set_title("B. Parameter Footprint vs Perplexity", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Validation Perplexity (Lower is Better)", fontsize=11)
    ax2.set_ylabel("Total Parameters", fontsize=11)
    ax2.set_xlim(5.5, 12.0)
    ax2.set_ylim(30000, 54000)

    # -------------------------------------------------------------------------
    # Panel 3: Qualitative Generation Samples
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.axis("off")
    ax3.set_title("C. Autoregressive Text Sample Generation", fontsize=12, fontweight="bold")
    
    sample_text = "Prompt: 'ROMEO:' (T=0.8, 70 tokens)\n" + "-" * 42 + "\n"
    for r in results:
        m_name = r["name"].split()[0]
        gen = sample_generations.get(r["name"], "...")
        gen_clean = gen[:65].replace("\n", " ")
        sample_text += f"[{m_name:10s}] {gen_clean}...\n"
        
    ax3.text(0.02, 0.95, sample_text, transform=ax3.transAxes, fontsize=8.5,
             family="monospace", verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#bdc3c7", alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[+] Publication-quality TinyShakespeare plot saved to: {output_path}")


# -----------------------------------------------------------------------------
# 6. MAIN PIPELINE
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TinyShakespeare Causal Language Modeling Benchmark")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps per model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=64, help="Context sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--plot_only", action="store_true", help="Only regenerate plot from cached results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("PHASEATTENTION: CAUSAL LANGUAGE MODELING BENCHMARK (TINYSHAKESPEARE)")
    print("=" * 80)
    print(f"Platform: Python {sys.version.split()[0]} | PyTorch {torch.__version__} | Seed {args.seed}")

    data_path = REPO_ROOT / "data" / "tinyshakespeare.txt"
    train_data, val_data, vocab_size, ch2i, i2ch = load_tinyshakespeare(data_path)

    print(f"Corpus: {len(train_data) + len(val_data):,} characters | Vocab Size: {vocab_size} distinct characters")
    print(f"Context: L = {args.seq_len} tokens | Batch Size = {args.batch_size} | Training Steps = {args.steps}\n")

    res_dir = REPO_ROOT / "results"
    assets_dir = REPO_ROOT / "assets"
    res_dir.mkdir(exist_ok=True)
    assets_dir.mkdir(exist_ok=True)
    json_path = res_dir / "benchmark_tinyshakespeare_lm.json"
    plot_path = assets_dir / "tinyshakespeare_benchmark.png"

    if args.plot_only and json_path.exists():
        print(f"Loading existing metrics from {json_path} for plotting...")
        with open(json_path, "r") as f:
            payload = json.load(f)
        plot_tinyshakespeare_results(payload["results"], payload.get("samples", {}), plot_path)
        print("Done!")
        return

    # Model architecture configuration
    d_model = 64
    num_heads = 4
    d_v = 16
    
    models = [
        ("LinearHolographic (O(N))", CausalTransformerLM(LinearHolographicPhaseAttention(d_model=d_model, num_heads=num_heads, d_v=d_v), vocab_size=vocab_size, d_model=d_model)),
        ("Triangular (Multiplier-Free)", CausalTransformerLM(TriangularPhaseAttention(d_model=d_model, num_heads=num_heads, d_v=d_v, is_causal=True), vocab_size=vocab_size, d_model=d_model)),
        ("PhaseAttention (U(1))", CausalTransformerLM(PhaseAttention(d_model=d_model, num_heads=num_heads, d_v=d_v, is_causal=True), vocab_size=vocab_size, d_model=d_model)),
        ("StandardCausal Attention", CausalTransformerLM(StandardCausalAttn(d_model=d_model, num_heads=num_heads, d_k=d_v, d_v=d_v), vocab_size=vocab_size, d_model=d_model)),
        ("cosFormer (Causal)", CausalTransformerLM(CausalCosFormerAttn(d_model=d_model, num_heads=num_heads, d_v=d_v), vocab_size=vocab_size, d_model=d_model)),
        ("Edge-GRU (Recurrent)", EdgeGRULM(vocab_size=vocab_size, d_model=d_model))
    ]

    print("--- Training and Evaluating 6 Architectures on TinyShakespeare ---")
    results = []
    samples = {}
    
    for name, model in models:
        res = train_and_eval_lm(name, model, train_data, val_data, steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len)
        results.append(res)
        
        # Sample generation
        sample_out = generate_sample(model, "ROMEO:\n", ch2i, i2ch, max_new_tokens=70)
        samples[name] = sample_out

    payload = {
        "dataset": "TinyShakespeare",
        "vocab_size": vocab_size,
        "context_length": args.seq_len,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "results": results,
        "samples": samples
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Raw metrics saved to: {json_path}")

    plot_tinyshakespeare_results(results, samples, plot_path)

    print("\n" + "=" * 80)
    print("TINYSHAKESPEARE CAUSAL LM BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
