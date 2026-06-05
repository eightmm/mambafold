# train/ — Training Runtime

Active config: `configs/stage1.yaml` or later stage configs.

## Key Functions

| Function | Role |
|---|---|
| `config.parse_args` | YAML + CLI config parsing |
| `trainer.build_model` | instantiate `MambaFold` |
| `engine.forward_and_loss` | FM forward, aux losses, metrics |
| `engine.eval_step` | validation metrics |
| `trainer.save_checkpoint` | model/EMA/optimizer/scheduler checkpoint |
| `distributed.setup_dist` | NCCL DDP init |

The main loss metric is named `main` and corresponds to flow-matching MSE.
