"""v3a mixed-crop schedule: deterministic per-step crop length picker.

The schedule is a list of phases. Each phase is dict with:
  - `until`: step at which this phase ends (None / omitted = last phase, no upper bound)
  - `weights`: {length: probability} mapping; probabilities normalized internally

Lookup walks phases in order, picks the first one whose `until` exceeds `step`,
and samples a length from that phase's weights. The sampler uses a torch
Generator seeded with `step`, so all DDP ranks pick the same length on the same
step without any cross-rank communication.

Example YAML:
    crop_schedule:
      - until: 10000
        weights: {512: 0.7, 768: 0.3}
      - until: 30000
        weights: {512: 0.4, 768: 0.4, 1024: 0.2}
      - weights: {768: 0.4, 1024: 0.6}      # last phase: no `until`
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch


def pick_crop_length(
    step: int,
    schedule: Optional[Iterable[dict]],
    default_max: int,
) -> int:
    """Return the crop length for `step`. Returns `default_max` if no schedule."""
    if not schedule:
        return default_max

    weights: Optional[dict] = None
    for phase in schedule:
        until = phase.get("until")
        if until is None or step < int(until):
            weights = phase.get("weights")
            break
    if weights is None:
        # Past the last bounded phase → use the final phase's weights
        weights = list(schedule)[-1].get("weights")
    if not weights:
        return default_max

    # YAML may parse keys as either int or str; normalize.
    items = sorted(((int(L), float(p)) for L, p in weights.items()), key=lambda x: x[0])
    lengths = [L for L, _ in items]
    probs = [p for _, p in items]
    total = sum(probs)
    if total <= 0:
        return default_max

    g = torch.Generator()
    g.manual_seed(int(step))
    r = float(torch.rand(1, generator=g).item()) * total
    cum = 0.0
    for L, p in zip(lengths, probs):
        cum += p
        if r < cum:
            return int(L)
    return int(lengths[-1])
