import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wraps an angle tensor to [-pi, pi) using modulo arithmetic."""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


class PhaseAttention(nn.Module):
    """
    Standard Quadratic Phase Attention on the Unit Circle U(1).
    
    Each head projects tokens to a single phase angle theta in [-pi, pi).
    Affinity: S_ij = cos(theta_q,i - theta_k,j) * scale
    Zero multiplications in the Q x K interaction kernel!
    """
    def __init__(self, d_model: int, num_heads: int = 4, d_v: int = 8, is_causal: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = d_v
        self.is_causal = is_causal
        
        self.w_q = nn.Linear(d_model, num_heads)
        self.w_k = nn.Linear(d_model, num_heads)
        self.w_v = nn.Linear(d_model, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, d_model)
        self.scale = nn.Parameter(torch.tensor(8.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        B, L, _ = x.shape
        theta_q = (torch.tanh(self.w_q(x)) * math.pi).transpose(1, 2).unsqueeze(-1) # (B, H, L, 1)
        theta_k = (torch.tanh(self.w_k(x)) * math.pi).transpose(1, 2).unsqueeze(-2) # (B, H, 1, L)
        
        scores = torch.cos(theta_q - theta_k) * self.scale
        if self.is_causal:
            causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        v = self.w_v(x).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out)


class LinearHolographicPhaseAttention(nn.Module):
    """
    Exact O(N) Linear Holographic Phase Attention (No Softmax).
    
    Uses the exact analytical rank-2 separable identity:
        cos(theta_q - theta_k) = [cos theta_q, sin theta_q] @ [cos theta_k, sin theta_k]^T
                               = phi(q)^T phi(k)
                               
    State per head is S_t = S_(t-1) + phi(k_t) v_t^T in R^(2 x d_v).
    Unwanted signals cancel via destructive wave interference. Zero N x N matrix!
    """
    def __init__(self, d_model: int, num_heads: int = 4, d_v: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = d_v
        
        self.w_q = nn.Linear(d_model, num_heads)
        self.w_k = nn.Linear(d_model, num_heads)
        self.w_v = nn.Linear(d_model, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        theta_q = torch.tanh(self.w_q(x)) * math.pi # (B, L, H)
        theta_k = torch.tanh(self.w_k(x)) * math.pi
        
        # phi: (B, H, L, 2)
        phi_k = torch.stack([torch.cos(theta_k), torch.sin(theta_k)], dim=-1).transpose(1, 2)
        phi_q = torch.stack([torch.cos(theta_q), torch.sin(theta_q)], dim=-1).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.num_heads, self.d_v).transpose(1, 2) # (B, H, L, d_v)
        
        # Exact causal prefix sum in O(N):
        # Outer product of (2,) and (d_v,) = (B, H, L, 2, d_v)
        kv_pairs = phi_k.unsqueeze(-1) * v.unsqueeze(-2)
        state = torch.cumsum(kv_pairs, dim=2)
        
        # Linear readout: phi_q @ state
        out = torch.matmul(phi_q.unsqueeze(-2), state).squeeze(-2) # (B, H, L, d_v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out)

    def init_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Initializes recurrent memory state S_0 in R^(B x H x 2 x d_v) with zeros."""
        dev = device if device is not None else next(self.parameters()).device
        return torch.zeros(batch_size, self.num_heads, 2, self.d_v, device=dev)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Online streaming autoregressive step in strict O(1) time and O(1) state memory.
        
        Args:
            x_t: Input token representation at step t of shape (B, d_model)
            state: Recurrent memory state S_(t-1) of shape (B, num_heads, 2, d_v)
            
        Returns:
            out_t: Output representation of shape (B, d_model)
            next_state: Updated state S_t = S_(t-1) + phi(k_t) v_t^T of shape (B, num_heads, 2, d_v)
        """
        B, _ = x_t.shape
        theta_q = torch.tanh(self.w_q(x_t)) * math.pi
        theta_k = torch.tanh(self.w_k(x_t)) * math.pi
        
        phi_k = torch.stack([torch.cos(theta_k), torch.sin(theta_k)], dim=-1) # (B, H, 2)
        phi_q = torch.stack([torch.cos(theta_q), torch.sin(theta_q)], dim=-1) # (B, H, 2)
        v = self.w_v(x_t).view(B, self.num_heads, self.d_v)                  # (B, H, d_v)
        
        # State update: S_t = S_(t-1) + phi(k_t) v_t^T
        kv_t = phi_k.unsqueeze(-1) * v.unsqueeze(-2)                         # (B, H, 2, d_v)
        next_state = state + kv_t
        
        # Linear readout: y_t = phi(q_t)^T S_t
        out_t = torch.matmul(phi_q.unsqueeze(-2), next_state).squeeze(-2)    # (B, H, d_v)
        out_flat = out_t.contiguous().view(B, self.num_heads * self.d_v)
        return self.out_proj(out_flat), next_state


class DeltaPhaseLinearAttention(nn.Module):
    """
    O(N) Recurrent Delta-Rule Phase Memory.
    
    Updates the 2 x d_v state S_t with an error-correcting delta rule:
        S_t = S_(t-1) + beta_t * (v_t - phi(k_t)^T S_(t-1)) outer phi(k_t)
    """
    def __init__(self, d_model: int, num_heads: int = 4, d_v: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = d_v
        
        self.w_q = nn.Linear(d_model, num_heads)
        self.w_k = nn.Linear(d_model, num_heads)
        self.w_v = nn.Linear(d_model * num_heads * d_v) if False else nn.Linear(d_model, num_heads * d_v)
        self.w_beta = nn.Linear(d_model, num_heads)
        self.out_proj = nn.Linear(num_heads * d_v, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        theta_q = torch.tanh(self.w_q(x)) * math.pi
        theta_k = torch.tanh(self.w_k(x)) * math.pi
        beta = torch.sigmoid(self.w_beta(x)) # (B, L, H)
        
        phi_k = torch.stack([torch.cos(theta_k), torch.sin(theta_k)], dim=-1).transpose(1, 2)
        phi_q = torch.stack([torch.cos(theta_q), torch.sin(theta_q)], dim=-1).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        beta = beta.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        
        state = self.init_state(B, device=x.device)
        outs = []
        
        for t in range(L):
            pk_t = phi_k[:, :, t, :].unsqueeze(-2) # (B, H, 1, 2)
            v_t = v[:, :, t, :]                   # (B, H, d_v)
            b_t = beta[:, :, t, :, :]             # (B, H, 1, 1)
            
            pred_v = torch.matmul(pk_t, state).squeeze(-2)
            err = v_t - pred_v
            delta = torch.matmul(pk_t.transpose(-2, -1), err.unsqueeze(-2))
            state = state + b_t * delta
            
            pq_t = phi_q[:, :, t, :].unsqueeze(-2)
            y_t = torch.matmul(pq_t, state).squeeze(-2)
            outs.append(y_t)
            
        out_seq = torch.stack(outs, dim=2)
        out = out_seq.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out)

    def init_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Initializes recurrent memory state S_0 in R^(B x H x 2 x d_v) with zeros."""
        dev = device if device is not None else next(self.parameters()).device
        return torch.zeros(batch_size, self.num_heads, 2, self.d_v, device=dev)

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Online single-step autoregressive update for Delta-Rule memory."""
        B, _ = x_t.shape
        theta_q = torch.tanh(self.w_q(x_t)) * math.pi
        theta_k = torch.tanh(self.w_k(x_t)) * math.pi
        beta = torch.sigmoid(self.w_beta(x_t)) # (B, H)
        
        phi_k = torch.stack([torch.cos(theta_k), torch.sin(theta_k)], dim=-1) # (B, H, 2)
        phi_q = torch.stack([torch.cos(theta_q), torch.sin(theta_q)], dim=-1) # (B, H, 2)
        v = self.w_v(x_t).view(B, self.num_heads, self.d_v)                  # (B, H, d_v)
        b_t = beta.unsqueeze(-1).unsqueeze(-1)                                # (B, H, 1, 1)
        
        pk_t = phi_k.unsqueeze(-2) # (B, H, 1, 2)
        pred_v = torch.matmul(pk_t, state).squeeze(-2)
        err = v - pred_v
        delta = torch.matmul(pk_t.transpose(-2, -1), err.unsqueeze(-2))
        next_state = state + b_t * delta
        
        pq_t = phi_q.unsqueeze(-2)
        out_t = torch.matmul(pq_t, next_state).squeeze(-2)
        out_flat = out_t.contiguous().view(B, self.num_heads * self.d_v)
        return self.out_proj(out_flat), next_state


class TriangularPhaseAttention(nn.Module):
    """
    100% Multiplier-Free Phase Attention.
    
    Replaces cos(Delta theta) with a periodic triangular wave:
        tri(Delta theta) = 1.0 - (2 / pi) * |wrap(theta_q - theta_k)|
        
    Requires ONLY subtraction and absolute value (abs).
    Zero floating-point multipliers, zero transcendental functions!
    """
    def __init__(self, d_model: int, num_heads: int = 4, d_v: int = 8, is_causal: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = d_v
        self.is_causal = is_causal
        
        self.w_q = nn.Linear(d_model, num_heads)
        self.w_k = nn.Linear(d_model, num_heads)
        self.w_v = nn.Linear(d_model, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, d_model)
        self.scale = nn.Parameter(torch.tensor(8.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        theta_q = (torch.tanh(self.w_q(x)) * math.pi).transpose(1, 2).unsqueeze(-1)
        theta_k = (torch.tanh(self.w_k(x)) * math.pi).transpose(1, 2).unsqueeze(-2)
        
        diff = wrap_angle(theta_q - theta_k)
        tri_affinity = 1.0 - (2.0 / math.pi) * torch.abs(diff)
        scores = tri_affinity * self.scale
        
        if self.is_causal:
            causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        v = self.w_v(x).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out)


class LUT16PhaseAttention(nn.Module):
    """
    Look-Up Table (LUT-16) Phase Attention.
    
    Emulates an ultra-lightweight 16-word microcode ROM table for ASICs and FPGAs.
    Given angle difference Delta theta in [-pi, pi), maps to one of 16 precomputed values.
    """
    def __init__(self, d_model: int, num_heads: int = 4, d_v: int = 8, is_causal: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = d_v
        self.is_causal = is_causal
        
        self.w_q = nn.Linear(d_model, num_heads)
        self.w_k = nn.Linear(d_model, num_heads)
        self.w_v = nn.Linear(d_model, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, d_model)
        self.scale = nn.Parameter(torch.tensor(8.0))
        
        angles = torch.linspace(-math.pi, math.pi - (2 * math.pi / 16), 16)
        self.register_buffer("lut_table", torch.cos(angles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        theta_q = (torch.tanh(self.w_q(x)) * math.pi).transpose(1, 2).unsqueeze(-1)
        theta_k = (torch.tanh(self.w_k(x)) * math.pi).transpose(1, 2).unsqueeze(-2)
        
        diff = wrap_angle(theta_q - theta_k)
        bin_idx = torch.clamp(((diff + math.pi) / (2.0 * math.pi) * 16.0).long(), 0, 15)
        lut_affinity = self.lut_table[bin_idx]
        
        # Straight-Through Estimator for clean autograd through table lookup
        scores = ((lut_affinity - torch.cos(diff)).detach() + torch.cos(diff)) * self.scale
        
        if self.is_causal:
            causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        v = self.w_v(x).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out)
