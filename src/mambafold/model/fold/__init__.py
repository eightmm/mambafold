"""coarse-to-fine 2-stage MambaFold.

Stage 1: CA-only flow matching with Linear Triangle Attention pair stack.
Stage 2: All-atom flow matching conditioned on Stage 1 CA + trunk latents.

See:
  * docs/architecture.md      — main spec
  * docs/pair_module.md       — pair block (LinearTriAttn + MultUpdate)
  * docs/architecture.md    — file-level roadmap

Modules are added incrementally per the I0–I7 plan. Imports stay light
during I0 (skeleton) — heavier symbols become available as I1–I4 land.
"""

from mambafold.model.fold.conditioning import (
    CAAnchoredFourier,
    Stage1LatentBroadcast,
)
from mambafold.model.fold.linear_tri_attn import LinearTriangleAttention
from mambafold.model.fold.multiplicative_update import TriangleMultiplicativeUpdate
from mambafold.model.fold.pair_blocks import PairBlock, PairTransition, pair_to_single
from mambafold.model.fold.stage1_ca import MambaFoldStage1
from mambafold.model.fold.stage2_atom import MambaFoldStage2
from mambafold.model.fold.two_stage import TwoStageMambaFold

__all__ = [
    "LinearTriangleAttention",
    "TriangleMultiplicativeUpdate",
    "PairBlock",
    "PairTransition",
    "pair_to_single",
    "CAAnchoredFourier",
    "Stage1LatentBroadcast",
    "MambaFoldStage1",
    "MambaFoldStage2",
    "TwoStageMambaFold",
]
