# configs/

Active config: `direct_allatom_puremamba_attn6_geo_adaln_sf360_esmc6b.yaml`.

The ESM3 variant is retained for reproducibility only. New mainline training
uses pinned sequence-only ESMC-6B embeddings and must not resume an ESM3
checkpoint.

YAML keys map 1:1 to `scripts/train.py` arguments. CLI flags override YAML.
