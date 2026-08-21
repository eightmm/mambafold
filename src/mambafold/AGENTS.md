# src/mambafold

Active single-chain code path only.

| Area | Files |
|---|---|
| Data | `data/dataset.py`, `data/collate.py`, `data/transforms.py`, `data/types.py`, `data/loader.py` |
| Model | `model/fold/`, `model/bimamba3.py`, `model/embeddings.py` |
| Loss | `losses/lddt.py`; flow matching is composed in `train/engine.py` |
| Training | `train/config.py`, `train/engine.py`, `train/trainer.py`, `train/distributed.py` |
| Sampling | `sampling/samplers.py` |

Keep new code on this path unless `PROJECT.md` is updated first.
