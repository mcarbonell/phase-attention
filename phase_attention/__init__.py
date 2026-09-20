"""
PhaseAttention: Multiplier-Free, Exact O(N) Linear Attention on the Unit Circle U(1).
"""

from .core import (
    PhaseAttention,
    LinearHolographicPhaseAttention,
    DeltaPhaseLinearAttention,
    TriangularPhaseAttention,
    LUT16PhaseAttention,
    wrap_angle
)

__all__ = [
    "PhaseAttention",
    "LinearHolographicPhaseAttention",
    "DeltaPhaseLinearAttention",
    "TriangularPhaseAttention",
    "LUT16PhaseAttention",
    "wrap_angle"
]

__version__ = "0.1.0"
