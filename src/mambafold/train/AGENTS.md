# train/

Active config: `configs/direct_allatom_360m.yaml`.

Key functions:

- `config.parse_args`
- `trainer.build_model`
- `engine.allatom_forward_and_loss`
- `engine.allatom_eval_step`
- `trainer.save_checkpoint`
- `distributed.setup_dist`
