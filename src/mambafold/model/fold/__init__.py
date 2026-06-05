"""coarse-to-fine 2-stage MambaFold.

Stage 1: CA-only flow matching with a Linear Triangle Attention pair stack.
Stage 2: all-atom flow matching conditioned on the Stage 1 CA + trunk latent.

See:
  * docs/architecture.md  — main spec
  * docs/pair_module.md   — pair block (LinearTriAttn + MultUpdate)
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
