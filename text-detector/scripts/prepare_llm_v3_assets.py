#!/usr/bin/env python
"""llm_v3용 XDAC AI CSV + OOD 벤치마크 CSV 준비."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_build.external_hf import prepare_v3_external_assets  # noqa: E402
from utils import setup_logging  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xdac_n", type=int, default=500)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    args = p.parse_args()
    setup_logging()
    report = prepare_v3_external_assets(
        Path(args.data_dir), xdac_n=args.xdac_n, seed=args.seed
    )
    out = Path(args.data_dir) / "ood_benchmark" / "prepare_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
