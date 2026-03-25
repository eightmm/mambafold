# model/ — 모델 아키텍처

## 파일별 역할

### `mambafold.py` — MambaFoldEqM (진입점)
전체 forward pass를 조립하는 최상위 모듈.

```
x_γ [B,L,A,3]
  → AtomFeatureEmbedder           # 원자/잔기/좌표 임베딩
  → MambaStack(atom_encoder)      # [B*L, A, d_atom]  per-residue
  → group_atoms_to_residues()     # [B, L, d_atom]    masked avg pool
  → (PLM concat → trunk_proj)     # [B, L, d_res]
  → MambaStack(residue_trunk)     # [B, L, d_res]     BiMamba-3 heavy stack
  → ResidueToAtomBroadcast        # [B, L, A, d_atom] + skip 연결
  → AtomDecoder(MambaStack+head)  # [B*L, A, d_atom]  per-residue
  → GradHead                      # [B, L, A, 3]      EqM gradient
```

**EqM 설계 원칙**: 모델은 γ를 입력으로 받지 않는다 (time-unconditional).
`batch.gamma`는 ProteinBatch에 있지만 forward에서 무시됨.

**주요 파라미터** (overfit config):

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| d_atom | 256 | atom token 차원 |
| d_res | 256 | residue token 차원 |
| d_state | 32 | Mamba SSM state 크기 |
| mimo_rank | 2 | MIMO rank (chunk_size=16) |
| headdim | 64 | Mamba head dimension |
| n_atom_enc | 2 | atom encoder layers |
| n_trunk | 6 | residue trunk layers |
| n_atom_dec | 2 | atom decoder layers |

### `embeddings.py` — 임베딩 레이어

**CoordinateFourierEmbedder**:
- 3D 좌표 → Fourier 특성 (sin/cos × 16 주파수 대역)
- 공식: `[x,y,z, sin(2^k·x), cos(2^k·x), ...]` k=0..15
- 출력: 학습된 proj → d_fourier=128
- 목적: 좌표의 연속성과 위치 정보를 주파수 도메인으로 표현

**AtomFeatureEmbedder**:
세 가지 임베딩의 합 + 좌표 임베딩:
1. `res_type_embed`: 아미노산 종류 (21종) → d_atom, 잔기 내 모든 원자에 broadcast
2. `atom_type_embed`: 원자 종류 (N, CA, C, O, CB, ...) → d_atom
3. `pair_embed`: (잔기, 원자) 화학적 정체성 → d_atom (167가지 unique pair)
4. `coord_embed`: CoordinateFourierEmbedder(coords) → d_fourier

최종: `proj(concat(sum_1_2_3, coord_feat))` → d_atom

### `blocks.py` — 공통 빌딩 블록

**RMSNorm**: Layer norm 대신 사용. scale만 있고 bias 없음.
```python
out = x / rms(x) * weight
```

**SwiGLU**: Transformer FFN 대체. 더 나은 성능/효율.
```python
out = w2(silu(w1(x)) * w3(x))
d_ff = d_model * 8/3 (8의 배수로 올림)
```

### `grouping.py` — Atom ↔ Residue 변환

**group_atoms_to_residues**: masked average pool
```python
# [B, L, A, D] → [B, L, D]
pooled = (atom_tok * mask).sum(dim=2) / mask.sum(dim=2).clamp(1)
```

**ResidueToAtomBroadcast**: Linear projection + broadcast
```python
# [B, L, d_res] → [B, L, A, d_atom]
proj(res_tok).unsqueeze(2).expand(A) * atom_mask
```
AtomDecoder의 skip connection에서 atom 표현에 residue 문맥 추가.

### `atom_decoder.py` — AtomDecoder

MambaStack (bidirectional) + GradHead:
```python
GradHead: LayerNorm → Linear(d_atom, d_atom//2) → GELU → Linear(d_atom//2, 3)
```
입력: `[B*L, A, d_atom]` (per-residue로 reshape해서 처리)
출력: `[B*L, A, 3]` → reshape → `[B, L, A, 3]`

## ssm/ 서브패키지

→ `ssm/AGENTS.md` 참조

## 설계 결정

1. **Atom encoder/decoder에 MambaStack 사용**: 원래 lightweight attention으로 설계했으나
   Mamba-3로 통일. 원자 수가 ≤15로 작아 per-residue로 처리하면 실질적 길이 15의 시퀀스.
2. **Bidirectional for atoms**: 원자에도 순서 의존성 없으므로 BiMamba3 사용.
3. **Skip connection**: atom encoder 출력 + residue broadcast → atom decoder 입력.
   residue-level 문맥(BiMamba-3 trunk)이 atom 표현에 직접 주입됨.
