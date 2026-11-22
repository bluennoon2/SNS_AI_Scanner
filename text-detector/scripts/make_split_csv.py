#!/usr/bin/env python
"""outputs/<run>/splits/ 의 train/val/test CSV를 그대로 복사·병합하거나,
원본 데이터와 row 단위로 맞춰 별도 평가용 CSV를 만듭니다.

공식 성능은 held-out test만 사용하세요. --eval_all 은 이 test CSV에만 쓰세요.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--splits_dir",
        type=str,
        required=True,
        help="예: outputs/real_v1/splits",
    )
    p.add_argument(
        "--which",
        choices=["train", "val", "test", "all"],
        default="test",
    )
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    splits = Path(args.splits_dir)
    mapping = {
        "train": splits / "train.csv",
        "val": splits / "val.csv",
        "test": splits / "test.csv",
    }
    if args.which == "all":
        frames = [pd.read_csv(mapping[k], encoding="utf-8-sig") for k in mapping]
        out = pd.concat(frames, ignore_index=True)
    else:
        path = mapping[args.which]
        if not path.exists():
            raise SystemExit(f"없음: {path}")
        out = pd.read_csv(path, encoding="utf-8-sig")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved {out_path} n={len(out)} which={args.which}")


if __name__ == "__main__":
    main()
