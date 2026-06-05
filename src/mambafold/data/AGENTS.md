# data/ — Dataset and Batch Plumbing

`ProteinBatch` is built for flow matching:

| Field | Meaning |
|---|---|
| `x_clean` | normalized ground truth coords |
| `x_t` | corrupted coords, model input |
| `eps` | sampled noise |
| `t` | interpolation time |
| `esm` | optional ESM3 conditioning |

Current corruption:

```text
x_t = t * x_clean + (1 - t) * eps
```

`RCSBDataset` preserves `chain_id`, `entity_id`, `sym_id`, chain-local `res_seq_nums`, and original terminus flags. `ProteinCollator` pads to the configured `max_length` and creates `ProteinBatch` tensors.
