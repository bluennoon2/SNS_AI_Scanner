#!/usr/bin/env bash
# 파일럿: 데이터 구축 → 학습 → 평가 → 추론 (끝까지 완주 검증)
# 성능 지표는 해석하지 않는다.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROVIDER="${PROVIDER:-auto}"

echo "=== [1/4] 데이터 구축 (pilot) provider=${PROVIDER} ==="
if ! python -m data_build.build_dataset \
  --mode pilot \
  --provider "${PROVIDER}" \
  --resume \
  --output data/dataset_pilot.csv; then
  code=$?
  echo "실패 단계: 데이터 구축 (exit=${code})"
  if [[ "${PROVIDER}" != "mock" ]]; then
    echo "API 키가 없다면: PROVIDER=mock bash scripts/run_pilot.sh"
    echo "또는: python -m data_build.build_dataset --mode pilot --provider mock --resume --output data/dataset_pilot_mock.csv"
  fi
  exit "${code}"
fi

# mock이면 파일명이 _mock 으로 바뀌었을 수 있음
DATA="data/dataset_pilot.csv"
if [[ ! -f "${DATA}" ]]; then
  if [[ -f "data/dataset_pilot_mock.csv" ]]; then
    DATA="data/dataset_pilot_mock.csv"
  else
    echo "실패 단계: 데이터 구축 — 출력 CSV를 찾을 수 없음"
    exit 1
  fi
fi

echo "=== [2/4] 학습 (group + fail_on_leak) data=${DATA} ==="
if ! python train.py --data_path "${DATA}" --output_dir outputs/pilot \
  --split_strategy group --fail_on_leak --epochs 1 --batch_size 8 --max_length 128 \
  --threshold_mode target_fpr --target_fpr 0.05 --compare_thresholds; then
  echo "실패 단계: 학습"
  exit 1
fi

echo "=== [3/4] 평가 + 오류 분석 ==="
if ! python evaluate.py --checkpoint outputs/pilot/model --data_path "${DATA}" \
  --eval_all --error_top_k 10; then
  echo "실패 단계: 평가"
  exit 1
fi

echo "=== [4/4] 추론 ==="
if ! python predict.py --checkpoint outputs/pilot/model \
  --text "ㅋㅋㅋ 오늘 날씨 미쳤다 #일상"; then
  echo "실패 단계: 추론"
  exit 1
fi

echo "파일럿 파이프라인 완주 완료 (성능 해석 없음)."
