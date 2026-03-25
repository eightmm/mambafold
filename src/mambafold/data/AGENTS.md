# data/ — 데이터 파이프라인

## 파일별 역할

### `constants.py`
단백질 구조의 모든 어휘(vocabulary) 정의. 다른 모든 모듈이 참조.

```python
NUM_AA_TYPES = 21          # 20종 표준 AA + UNK
MAX_ATOMS_PER_RES = 15     # atom14(14) + OXT(1) = 15 slots
CA_ATOM_ID = 1             # Cα는 슬롯 인덱스 1 (N=0, CA=1, C=2, O=3)
COORD_SCALE = 10.0         # Å → normalized (÷10), 역방향 ×10
NUM_ATOM_TYPES             # 유니크 원자 이름 수 (PAD 포함)
NUM_PAIR_TYPES             # (잔기, 원자) 페어 수 + PAD
PAIR_PAD_ID                # 비어있는 atom slot의 pair_type 값
```

**atom14 슬롯 레이아웃**: N(0), CA(1), C(2), O(3), CB(4), 잔기별 사슬 원자(5~13), OXT(14)
- GLY는 CB 없으므로 슬롯 4가 비어 atom_mask=False

### `types.py`
데이터 컨테이너 dataclass.

```python
ProteinExample:            # 단일 단백질 (배치 전)
    res_type   [L]         # AA 타입 인덱스
    atom_type  [L, A]      # 원자 타입 인덱스 (A=15)
    pair_type  [L, A]      # (잔기, 원자) 페어 인덱스
    coords     [L, A, 3]   # ground truth 좌표 (정규화 전)
    atom_mask  [L, A]      # 유효한 atom slot
    observed_mask [L, A]   # 실험적으로 관측된 원자
    res_seq_nums  [L]      # 원래 residue 번호

ProteinBatch:              # 배치 (패딩 포함)
    # 시퀀스
    res_type   [B, L]
    atom_type  [B, L, A]
    pair_type  [B, L, A]
    res_mask   [B, L]      # 유효 잔기 (padding mask)
    atom_mask  [B, L, A]
    valid_mask [B, L, A]   # atom_mask & observed_mask (loss 계산용)
    ca_mask    [B, L]      # Cα 보유 여부 (LDDT용)
    # 좌표
    x_clean    [B, L, A, 3]   # 정규화된 ground truth
    x_gamma    [B, L, A, 3]   # noisy coords (모델 입력)
    eps        [B, L, A, 3]   # 노이즈
    gamma      [B, 1, 1, 1]   # 보간 인수 γ
    # PLM
    esm        [B, L, d_plm]  # ESM 임베딩 (None이면 on-the-fly)
```

`batch.with_coords(new_x)` — 샘플링 루프에서 x_gamma만 교체할 때 사용.

### `dataset.py`
AFDB `.pt` 파일 로딩. `.pt`는 전처리된 `ProteinExample` (또는 dict).

```python
AFDBDataset(data_dir, max_length=128)
    - 표준 20종 AA만 수용 (비표준 → skip)
    - atom14 canonical slot 매핑
    - 로딩 실패 시 None 반환 (collate에서 skip)
```

### `transforms.py`
`center_and_scale(example)`:
- centroid 제거 (translation equivariance)
- `/COORD_SCALE(10.0)` 정규화

훈련 중 EqM corruption은 `overfit.py::make_batch` 또는 `engine.py::train_step` 에서 인라인으로 수행:
```python
eps = randn_like(coords)
x_gamma = gamma * coords + (1 - gamma) * eps
```

### `collate.py`
배치 패딩 및 마스킹. `ProteinExample` 리스트 → `ProteinBatch`.

### `esm.py`
`EvolutionaryScalePLM`: ESM3/ESMC 임베딩 제공.
- overfit에서는 `use_plm=False` → 사용 안 함
- 전체 학습 시 on-the-fly 또는 precompute 캐시 활용 예정

## 주요 흐름

```
afdb_data/train/*.pt
  → AFDBDataset.__getitem__() → ProteinExample
  → center_and_scale() → 정규화된 ProteinExample
  → make_batch() → ProteinBatch (γ corruption 포함)
  → model(batch) → gradient prediction
```

## 주의사항

- `valid_mask` = `atom_mask & observed_mask`: loss 계산 시 관측되지 않은 원자 제외
- `ca_mask`: Cα 위치만 True. LDDT 계산에 사용
- `PAIR_PAD_ID`: 빈 슬롯의 pair_type 기본값 (embedding에서 패딩으로 처리)
