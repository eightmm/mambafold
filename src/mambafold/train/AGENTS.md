# train/ — 훈련 파이프라인

## `engine.py`

### `train_step(model, batch, optimizer, ...)`
단일 훈련 스텝. AMP(bf16) + grad clip 포함.

```python
pred = model(batch)                          # [B, L, A, 3]
loss_eqm  = eqm_loss(pred, x_clean, eps, gamma, valid_mask)
x_hat     = x_gamma - scale * pred          # 1-step 구조 복원
loss_lddt = soft_lddt_ca_loss(x_hat, x_clean, ca_mask)

alpha = 1.0 (pretrain) 또는 1+8·ReLU(γ-0.5) (finetune)
loss  = loss_eqm + alpha * loss_lddt

optimizer.zero_grad()
loss.backward()
clip_grad_norm_(model.parameters(), grad_clip)
optimizer.step()

returns: {"loss", "eqm", "lddt", "alpha"}
```

**alpha_mode="ramp"**: finetune 시 γ>0.5인 배치(clean에 가까운 구조)에서 LDDT 가중치를
최대 9배까지 높임. 구조 품질 파인튜닝에 사용 (현재 overfit에서는 "const" 사용).

### `eval_step(model, batch, ...)`
`@torch.no_grad()` 평가 스텝. 추가로 `grad_rms` 계산:

```python
grad_rms = sqrt(||pred||² / n_valid_atoms / 3)
```

grad_rms: 샘플링 수렴의 척도. 완벽 학습 시 real data에서 → 0.

returns: {"eqm", "lddt", "grad_rms"}

## `ema.py` — EMA (Exponential Moving Average)

```python
EMA(model, decay=0.999)
    .update(model)       # shadow = decay*shadow + (1-decay)*param
    .state_dict()        # shadow weights dict (체크포인트 저장용)
    .load_state_dict()   # shadow 복원
```

추론 시에는 EMA weights 사용 → 더 안정적인 구조 생성.
overfit에서는 훈련 완료 후 EMA 저장만 하고 아직 sampling에서 미사용.

## 훈련 설정 (overfit 기준)

| 항목 | 값 |
|------|-----|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-2) |
| Precision | bf16 AMP (CUDA) |
| Grad clip | 1.0 |
| Gamma grid | 50개 균일 격자 (0.01, 0.03, ..., 0.99) |
| Gamma 순환 | 매 50 step마다 전체 gamma grid 한 바퀴 |
| EMA decay | 0.999 |

## 전체 학습 시 예상 설정 (미구현)

| 항목 | Pretrain | Finetune |
|------|---------|---------|
| lr | 1e-4 | 1e-4 |
| Warmup | 5K steps | 없음 |
| Schedule | cosine | cosine |
| alpha_mode | "const" | "ramp" |
| Max length | 256 | 512 |
| Batch | 32K atoms/step | 절반 |
