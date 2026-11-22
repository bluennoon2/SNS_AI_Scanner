#!/usr/bin/env python
"""실데이터셋 가드레일 점검. mock 혼입·group 위반·교락 징후가 있으면 학습 전 중단."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_build.quality import length_bucket, shortcut_baseline_auc  # noqa: E402
from data_build.schema import assert_group_consistency  # noqa: E402
from dataset import split_group  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--report_json", default=str(ROOT / "data" / "dataset_report.json"))
    p.add_argument("--fail_on_offline", action="store_true", default=False)
    args = p.parse_args()

    path = Path(args.data_path)
    errors = []
    warnings = []

    if "_mock" in path.stem:
        errors.append(f"파일명에 _mock 포함: {path.name}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    if "generator" in df.columns and (df["generator"].astype(str) == "mock").any():
        errors.append("generator 컬럼에 mock 이 있습니다.")

    report = {}
    rj = Path(args.report_json)
    if rj.exists():
        report = json.loads(rj.read_text(encoding="utf-8"))
        if report.get("mock_used"):
            errors.append("dataset_report.json mock_used=true")
        if report.get("offline_used") or report.get("file_ai_used"):
            msg = (
                "offline/file(cursor) 경로 사용 — 외부 LLM API 결과가 아닙니다. "
                f"disclaimer={report.get('disclaimer_offline')}"
            )
            if args.fail_on_offline:
                errors.append(msg)
            else:
                warnings.append(msg)

    try:
        assert_group_consistency(df)
        print("[ok] assert_group_consistency")
    except ValueError as exc:
        errors.append(f"group 일관성 실패: {exc}")

    # group 분할 사전 검증
    if "group_id" in df.columns and df["label"].nunique() >= 2 and len(df) >= 20:
        try:
            tr, va, te, method = split_group(df, 0.15, 0.15, seed=42)
            gtr, gva, gte = (
                set(tr["group_id"].astype(str)),
                set(va["group_id"].astype(str)),
                set(te["group_id"].astype(str)),
            )
            leak = (gtr & gva) | (gtr & gte) | (gva & gte)
            if leak:
                errors.append(f"가상 group 분할 누수 {len(leak)}개 group")
            else:
                print(f"[ok] 가상 group 분할 누수 0 (method={method})")
        except Exception as exc:
            warnings.append(f"가상 분할 생략: {type(exc).__name__}: {exc}")

    # 분포
    print("--- label ---")
    print(df["label"].value_counts().to_dict())
    if "origin" in df.columns:
        print("--- origin ---")
        print(df["origin"].astype(str).value_counts().to_dict())
    if "generator" in df.columns:
        print("--- generator ---")
        print(df["generator"].astype(str).value_counts().to_dict())
    if "source_type" in df.columns:
        print("--- source_type ---")
        print(df["source_type"].astype(str).value_counts().to_dict())
    df = df.copy()
    df["_lb"] = df["text"].astype(str).str.len().map(length_bucket)
    print("--- length_bucket ---")
    print(df["_lb"].value_counts().to_dict())
    if "source_type" in df.columns:
        print("--- source_type x length ---")
        print(pd.crosstab(df["source_type"], df["_lb"]).to_string())

    # 라벨-소스 교락 간단 점검: AI/사람이 source에만 치우침
    if "source_type" in df.columns and df["label"].nunique() == 2:
        for st, part in df.groupby(df["source_type"].astype(str)):
            if len(part) < 5:
                continue
            rate = float(part["label"].mean())
            if rate <= 0.05 or rate >= 0.95:
                warnings.append(
                    f"source_type={st} 라벨 비율 pos={rate:.2f} — 교락 가능"
                )

    conflicts = ROOT / "data" / "label_conflicts.csv"
    if conflicts.exists():
        n = len(pd.read_csv(conflicts, encoding="utf-8-sig"))
        print(f"label_conflicts.csv: {n}행")
    else:
        print("label_conflicts.csv: 없음(충돌 0으로 간주)")

    shortcut = shortcut_baseline_auc(df, seed=42)
    print("--- shortcut ---")
    print(shortcut)
    if shortcut.get("meta_auc") is not None and shortcut["meta_auc"] >= 0.8:
        warnings.append(
            f"메타 특징 AUC={shortcut['meta_auc']:.3f} ≥ 0.8 — 분포 재조정 필요 경고"
        )

    for w in warnings:
        print(f"[warn] {w}")
    if errors:
        print("\n[FAIL] 학습으로 넘어가지 마세요:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\n[PASS] check_dataset_real 통과 (경고는 위 참고)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
