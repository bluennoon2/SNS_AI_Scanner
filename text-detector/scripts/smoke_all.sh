#!/usr/bin/env bash
# 전체 스모크 검증 스크립트 (더미 data/sample.csv — 동작 확인 전용)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
if [[ -x .venv/Scripts/python.exe ]]; then
  PY=".venv/Scripts/python.exe"
elif [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
fi

echo "== 0) 회귀 학습/추론 =="
$PY train.py --data_path data/sample.csv --output_dir outputs/smoke2 --epochs 1 --batch_size 4 --max_length 64 --split_strategy random
$PY predict.py --checkpoint outputs/smoke2/model --text "ㅋㅋㅋ 오늘 날씨 미쳤다 #일상"

echo "== 1) 구버전 체크포인트 =="
$PY predict.py --checkpoint outputs/smoke/model --text "테스트 문장입니다"

echo "== 2) group 분할 =="
$PY train.py --data_path data/sample.csv --output_dir outputs/s_group --epochs 1 --batch_size 4 --max_length 64 --split_strategy group

echo "== 3) 임계값 비교 =="
$PY train.py --data_path data/sample.csv --output_dir outputs/s_thr --epochs 1 --batch_size 4 --max_length 64 --threshold_mode target_fpr --target_fpr 0.01 0.05 --compare_thresholds
$PY evaluate.py --checkpoint outputs/s_thr/model --data_path data/sample.csv --eval_all --compare_thresholds --error_top_k 5 --output_dir outputs/s_thr_eval

echo "== 4) 토크나이저 점검 =="
$PY inspect_tokenizer.py --data_path data/sample.csv --max_length 128 --n_samples 20

echo "== 5) focal loss =="
$PY train.py --data_path data/sample.csv --output_dir outputs/s_focal --epochs 1 --batch_size 4 --max_length 64 --loss focal --focal_gamma 2.0

echo "== 6) 배치 추론 =="
$PY predict.py --checkpoint outputs/smoke2/model --input_file data/sample.csv --output_file outputs/smoke2/preds.csv

echo "== 7) pytest =="
$PY -m pytest tests -q

echo "ALL SMOKE PASSED"
