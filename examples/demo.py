#!/usr/bin/env python3
"""
Quickstart Demo: PhaseAttention vs Linear Holographic Attention
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from phase_attention import PhaseAttention, LinearHolographicPhaseAttention, TriangularPhaseAttention


def main():
    print("=" * 60)
    print("PhaseAttention: Quickstart Demo")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 16
    d_model = 32
    num_heads = 4
    d_v = 8
    
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input Tensor Shape: {x.shape} (B={batch_size}, L={seq_len}, d={d_model})\n")
    
    # 1. Standard PhaseAttention (U(1) Cosine)
    m1 = PhaseAttention(d_model=d_model, num_heads=num_heads, d_v=d_v)
    out1 = m1(x)
    params1 = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    print(f"[1] PhaseAttention (U(1)):           Output {out1.shape}, Params: {params1}")
    
    # 2. Linear Holographic Phase Attention (Exact O(N), No Softmax)
    m2 = LinearHolographicPhaseAttention(d_model=d_model, num_heads=num_heads, d_v=d_v)
    out2 = m2(x)
    params2 = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    print(f"[2] LinearHolographicPhase O(N):      Output {out2.shape}, Params: {params2}")
    
    # 3. Triangular Multiplier-Free Attention (Resta + Abs only)
    m3 = TriangularPhaseAttention(d_model=d_model, num_heads=num_heads, d_v=d_v)
    out3 = m3(x)
    params3 = sum(p.numel() for p in m3.parameters() if p.requires_grad)
    print(f"[3] Triangular Multiplier-Free:      Output {out3.shape}, Params: {params3}")
    
    print("\nAll forward passes executed successfully!")

if __name__ == "__main__":
    main()
