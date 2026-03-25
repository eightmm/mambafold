# model/ssm/ — Mamba-3 SSM 레이어

## 파일별 역할

### `mamba3.py` — Mamba3Layer

공식 `mamba_ssm.modules.mamba3.Mamba3`의 thin wrapper.
주요 역할: **패딩 마스크 지원** + **chunk_size 패딩 자동 처리**.

```python
Mamba3Layer(d_model, d_state, expand, headdim, mimo_rank)
    forward(x [B,L,D], mask [B,L]) → [B,L,D]
```

**MIMO (Multiple Input Multiple Output)**:
- `mimo_rank > 1`일 때 MIMO 모드 활성화
- `chunk_size = 32 // mimo_rank`
  - rank=1: chunk=64 (SISO)
  - rank=2: chunk=16
  - rank=4: chunk=8
- Triton/TileLang 커널: `seq_len % chunk_size == 0` 필요 → 자동 패딩 후 slicing

**패딩 처리**:
1. `x = x * mask` (패딩 위치를 0으로)
2. `seq_len` → chunk_size의 배수로 패드
3. `y = ssm(x)`
4. 원래 길이로 slice + 다시 mask

**GPU 필수**: 공식 mamba_ssm 커널은 CUDA Triton 커널. CPU fallback 없음.

### `bimamba3.py` — BiMamba3Block & MambaStack

**Mamba3Block** (단방향, causal):
```python
x → pre-norm(RMSNorm) → Mamba3Layer(forward) → residual
  → pre-norm → SwiGLU → residual
```
단백질 trunk에서는 미사용. 필요 시 `bidirectional=False`로 MambaStack에서 선택 가능.

**BiMamba3Block** (양방향, 핵심):
```python
h = RMSNorm(x)
y_f = Mamba3Layer_forward(h, mask)               # N→C 방향
y_b = flip → Mamba3Layer_backward(flip(h)) → flip  # C→N 방향

gate = sigmoid(Linear(concat([y_f, y_b])))       # [B, L, d_model]
fused = gate * y_f + (1-gate) * y_b             # 소프트 gated fusion
x = x + Linear(fused)                           # residual
x = x + SwiGLU(RMSNorm(x))                     # FFN residual
```

**gated fusion 선택 이유**: 단백질은 N→C 방향성이 없으므로 forward/backward 양쪽
문맥이 모두 필요. 단순 평균보다 gate 학습이 각 위치에서 더 적합한 방향을 선택할 수 있음.

**_flip_by_mask**: 유효 위치만 뒤집고 패딩은 끝에 유지.
```python
# [a, b, c, PAD] → [c, b, a, PAD]  (길이 3)
lengths = mask.sum(1)
rev_idx = (lengths - 1 - arange).clamp(0)
out = gather(x, rev_idx) * mask
```

**MambaStack**: N개의 Mamba3Block 또는 BiMamba3Block을 쌓는 재사용 가능 스택.
```python
MambaStack(d_model, n_layers, bidirectional=True, ...)
    → atom encoder, residue trunk, atom decoder 모두 이 클래스 사용
```

## 파라미터별 메모리/속도 영향

| 파라미터 | 설명 | 현재값 (overfit) |
|---------|------|-----------------|
| d_state | SSM state 크기 (메모리 ∝ d_state) | 32 |
| mimo_rank | MIMO rank (표현력 ↑, chunk↓) | 2 |
| expand | d_inner = d_model * expand | 2 |
| headdim | head 차원 (d_inner % headdim == 0 필요) | 64 |

## 주의사항

- `d_inner = d_model * expand` 가 `headdim`의 배수여야 함
- `d_inner / headdim` = 헤드 수
- MIMO rank=2, d_model=256, expand=2 → d_inner=512, n_heads=8, chunk_size=16
- 시퀀스 길이가 chunk_size보다 짧을 때도 자동 패딩으로 처리됨
