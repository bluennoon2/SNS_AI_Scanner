#!/usr/bin/env bash
# 본 학습용 데이터 구축 + 학습. 시작 전 --dry_run 으로 호출 수를 확인하세요.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROVIDER="${PROVIDER:-auto}"
TARGET="${TARGET:-20000}"

echo "============================================================"
echo "본 실행 안내"
echo "  1) 먼저 호출 수를 확인하세요:"
echo "       python -m data_build.build_dataset --mode full --target_total ${TARGET} --provider ${PROVIDER} --dry_run"
echo "  2) API 키·쿼터·비용을 확인한 뒤 이 스크립트를 실행하세요."
echo "  3) 중단 시 같은 명령에 --resume 을 붙여 이어갈 수 있습니다."
echo "============================================================"

echo "=== [1/2] 데이터 구축 (full, target=${TARGET}) ==="
if ! python -m data_build.build_dataset \
  --mode full \
  --target_total "${TARGET}" \
  --provider "${PROVIDER}" \
  --resume \
  --output data/dataset_full.csv; then
  code=$?
  echo "실패 단계: 데이터 구축 (exit=${code})"
  if [[ "${PROVIDER}" != "mock" ]]; then
    echo "키가 없으면 --provider mock 또는 PROVIDER=mock 을 사용하세요 (실성능 평가 금지)."
  fi
  exit "${code}"
fi

DATA="data/dataset_full.csv"
if [[ ! -f "${DATA}" ]]; then
  if [[ -f "data/dataset_full_mock.csv" ]]; then
    DATA="data/dataset_full_mock.csv"
  else
    echo "실패 단계: 출력 CSV 없음"
    exit 1
  fi
fi

echo "=== [2/2] 학습 (group + fail_on_leak, epochs=3) ==="
if ! python train.py --data_path "${DATA}" --output_dir outputs/full \
  --split_strategy group --fail_on_leak --epochs 3 --max_length 128 \
  --threshold_mode target_fpr --target_fpr 0.05 --compare_thresholds; then
  echo "실패 단계: 학습"
  exit 1
fi

echo "본 파이프라인 구축+학습 완료. 이어서 evaluate.py / predict.py 를 실행하세요."
