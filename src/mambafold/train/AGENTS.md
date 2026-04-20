# train/ — Training Pipeline

## `distributed.py`
**setup_dist()** — Initialize DDP (from SLURM env or torch.launch).

```python
is_dist, rank, world_size, device = setup_dist()
```

Returns DDP status and device for current rank.

**all_reduce_mean(tensor, world_size)** — Synchronize metric across ranks.

**GPUMonitor** — Background thread polling nvidia-smi, logs GPU util/memory.

## `trainer.py`
**build_model(cfg, device)** — Instantiate MambaFoldEqM from config dict.

**cosine_warmup_lr(optimizer, warmup_steps, total_steps)** — LR scheduler.
```
Linear warmup (0 → 1) then cosine decay (1 → 0.5) over total_steps
```

**save_checkpoint(out_dir, step, model, ema, optimizer, scheduler, args)** — DDP-aware checkpoint save.
- Saves raw model weights (extract from DDP wrapper)
- Creates symlink `ckpt_latest.pt` pointing to latest

**load_checkpoint(path, model, ema, optimizer, scheduler, device)** — DDP-aware resume.

## `engine.py`
**train_step(model, batch, optimizer, ...)**

```python
pred = model(batch)                       # [B, L, 14, 3]
loss_eqm = eqm_loss(pred, ...)
x_hat = x_gamma - scale * pred            # 1-step reconstruction
loss_lddt = soft_lddt_ca_loss(x_hat, ...)

alpha = 1.0 (pretrain) or 1+8*ReLU(γ-0.5) (finetune)
loss = loss_eqm + alpha * loss_lddt

optimizer.zero_grad(set_to_none=True)     # Memory efficient
loss.backward()
clip_grad_norm_(model.parameters(), grad_clip)
optimizer.step()
scheduler.step()
ema.update(model)

return {"loss", "eqm", "lddt", "grad_rms"}
```

Precision: bfloat16 AMP for forward, fp32 loss computation.

**eval_step(model_ema, batch)** — Evaluation (no grad).

Computes:
- `loss_eqm`
- `loss_lddt`
- `grad_rms` = convergence indicator

## `ema.py`
**EMA** — Exponential Moving Average of model weights.

```python
ema = EMA(model, decay=0.999)
ema.update(model)                # shadow = decay*shadow + (1-decay)*params
ema.switch_to_shadow()           # Use shadow weights (for inference)
ema.switch_to_params()           # Use original weights (for training)
```

Improves inference stability without changing model weights during training.

## Configuration (pretrain_256.yaml)

| Setting | Value |
|---------|-------|
| optimizer | AdamW, lr=1e-4, warmup_steps=2000 |
| precision | bfloat16 AMP |
| grad_clip | 1.0 |
| ema_decay | 0.999 |
| ckpt_interval | 5000 steps |
| eval_interval | Optional |
