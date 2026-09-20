import math
import torch
import pytest

from phase_attention import (
    PhaseAttention,
    LinearHolographicPhaseAttention,
    DeltaPhaseLinearAttention,
    TriangularPhaseAttention,
    LUT16PhaseAttention,
    wrap_angle,
)


def test_wrap_angle():
    theta = torch.tensor([-3.0 * math.pi, -math.pi, 0.0, math.pi, 3.0 * math.pi])
    wrapped = wrap_angle(theta)
    assert torch.all(wrapped >= -math.pi)
    assert torch.all(wrapped < math.pi)
    assert torch.isclose(wrapped[2], torch.tensor(0.0), atol=1e-5)


@pytest.mark.parametrize("model_cls", [
    PhaseAttention,
    LinearHolographicPhaseAttention,
    DeltaPhaseLinearAttention,
    TriangularPhaseAttention,
    LUT16PhaseAttention,
])
def test_forward_backward(model_cls):
    B, L, d_model = 2, 8, 16
    num_heads, d_v = 2, 4
    x = torch.randn(B, L, d_model, requires_grad=True)
    
    model = model_cls(d_model=d_model, num_heads=num_heads, d_v=d_v)
    out = model(x)
    
    assert out.shape == (B, L, d_model)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_causal_masking():
    B, L, d_model = 2, 6, 16
    x1 = torch.randn(B, L, d_model)
    x2 = x1.clone()
    # Change the last token in x2
    x2[:, -1, :] = torch.randn(B, d_model)
    
    model = PhaseAttention(d_model=d_model, num_heads=2, d_v=4, is_causal=True)
    model.eval()
    
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
        
    # In causal attention, tokens 0 to L-2 must have identical outputs
    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)


def test_linear_holographic_step_matches_forward():
    B, L, d_model = 2, 10, 16
    num_heads, d_v = 4, 8
    x = torch.randn(B, L, d_model)
    
    model = LinearHolographicPhaseAttention(d_model=d_model, num_heads=num_heads, d_v=d_v)
    model.eval()
    
    with torch.no_grad():
        out_parallel = model(x)
        
        # Step-by-step O(1) recurrent streaming
        state = model.init_state(B, device=x.device)
        step_outs = []
        for t in range(L):
            out_t, state = model.step(x[:, t, :], state)
            step_outs.append(out_t)
            
        out_streaming = torch.stack(step_outs, dim=1)
        
    assert torch.allclose(out_parallel, out_streaming, atol=1e-5)


def test_delta_phase_step_matches_forward():
    B, L, d_model = 2, 8, 16
    num_heads, d_v = 4, 8
    x = torch.randn(B, L, d_model)
    
    model = DeltaPhaseLinearAttention(d_model=d_model, num_heads=num_heads, d_v=d_v)
    model.eval()
    
    with torch.no_grad():
        out_parallel = model(x)
        
        state = model.init_state(B, device=x.device)
        step_outs = []
        for t in range(L):
            out_t, state = model.step(x[:, t, :], state)
            step_outs.append(out_t)
            
        out_streaming = torch.stack(step_outs, dim=1)
        
    assert torch.allclose(out_parallel, out_streaming, atol=1e-5)
