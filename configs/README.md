# configs/ - 설정 파일

## 용도
- 모델 하이퍼파라미터 (hidden dim, num layers, SSM params 등)
- 학습 설정 (lr, batch size, scheduler, epochs)
- 데이터 설정 (crop size, feature 선택, augmentation)

## 파일 형식
- YAML 기반 설정 파일 사용
- 실험별 config 파일 분리 (e.g., `base.yaml`, `debug.yaml`)
