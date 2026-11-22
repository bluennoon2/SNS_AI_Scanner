# dataset_report.json/md, inspect_samples.csv 생성.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from data_build.quality import length_bucket
from utils import save_json

logger = logging.getLogger("sns_ai_detector")


def build_dataset_report(
    df: pd.DataFrame,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    work = df.copy()
    if "length_bucket" not in work.columns:
        work["length_bucket"] = work["text"].astype(str).str.len().map(length_bucket)

    report: Dict[str, Any] = {
        "n_total": int(len(work)),
        "by_label": work["label"].value_counts().to_dict() if len(work) else {},
        "by_source_type": work["source_type"].astype(str).value_counts().to_dict()
        if len(work)
        else {},
        "by_origin": work["origin"].astype(str).value_counts().to_dict()
        if len(work)
        else {},
        "by_generator": work["generator"].astype(str).value_counts().to_dict()
        if len(work)
        else {},
        "by_length_bucket": work["length_bucket"].value_counts().to_dict()
        if len(work)
        else {},
        "cross_origin_length": {},
        "group_stats": {},
        "mock_used": False,
        "disclaimer_mock": None,
    }

    if len(work):
        ct = pd.crosstab(
            work["origin"].astype(str), work["length_bucket"].astype(str)
        )
        report["cross_origin_length"] = {
            str(i): {str(c): int(ct.loc[i, c]) for c in ct.columns}
            for i in ct.index
        }

        gsize = work.groupby("group_id").size()
        mixed = 0
        for gid, part in work.groupby("group_id"):
            origins = set(part["origin"].astype(str))
            has_human = any(o.startswith("human") for o in origins)
            has_ai = any(o.startswith("ai_") for o in origins)
            if has_human and has_ai:
                mixed += 1
        report["group_stats"] = {
            "n_groups": int(work["group_id"].nunique()),
            "group_size_mean": float(gsize.mean()),
            "group_size_max": int(gsize.max()),
            "max_group_share": float(gsize.max() / len(work)),
            "mixed_human_ai_group_ratio": float(mixed / max(work["group_id"].nunique(), 1)),
        }
        gens = set(work["generator"].astype(str))
        if "mock" in gens:
            report["mock_used"] = True
            report["disclaimer_mock"] = (
                "⚠ mock 데이터는 파이프라인 동작 확인 전용이며 "
                "실제 성능 평가에 사용하면 안 됩니다."
            )

    if extras:
        report.update(extras)
    return report


def report_to_markdown(report: Dict[str, Any]) -> str:
    lines = ["# Dataset Report", ""]
    if report.get("mock_used"):
        lines.append(f"> {report.get('disclaimer_mock')}")
        lines.append("")
    lines.append(f"- n_total: {report.get('n_total')}")
    lines.append(f"- by_label: `{report.get('by_label')}`")
    lines.append(f"- by_source_type: `{report.get('by_source_type')}`")
    lines.append(f"- by_origin: `{report.get('by_origin')}`")
    lines.append(f"- by_generator: `{report.get('by_generator')}`")
    lines.append(f"- by_length_bucket: `{report.get('by_length_bucket')}`")
    lines.append("")
    lines.append("## Group stats")
    lines.append(f"```\n{report.get('group_stats')}\n```")
    lines.append("")
    if report.get("removal_stats"):
        lines.append("## Removal stats")
        lines.append(f"```\n{report['removal_stats']}\n```")
        lines.append("")
    if report.get("hf_sources"):
        lines.append("## HF sources / licenses")
        lines.append("| hf_id | status | license | n |")
        lines.append("|---|---|---|---:|")
        for s in report["hf_sources"]:
            lines.append(
                f"| {s.get('hf_id')} | {s.get('status')} | {s.get('license')} | {s.get('n_loaded', 0)} |"
            )
        lines.append("")
    if report.get("shortcut"):
        lines.append("## Shortcut baseline")
        lines.append(f"```\n{report['shortcut']}\n```")
        lines.append("")
    return "\n".join(lines)


def write_reports(
    df: pd.DataFrame,
    output_dir: Path,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_dataset_report(df, extras=extras)
    save_json(report, output_dir / "dataset_report.json")
    (output_dir / "dataset_report.md").write_text(
        report_to_markdown(report), encoding="utf-8"
    )
    logger.info("dataset_report 저장: %s", output_dir)
    return report


def write_inspect_samples(
    df: pd.DataFrame,
    path: Path,
    n: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """origin × 길이 버킷 층화 샘플. 파생은 parent_text 포함."""
    work = df.copy()
    work["length"] = work["text"].astype(str).str.len()
    work["length_bucket"] = work["length"].map(length_bucket)
    by_row = work.set_index("row_id", drop=False)

    rng = np.random.default_rng(seed)
    parts = []
    groups = list(work.groupby(["origin", "length_bucket"]))
    per = max(1, n // max(len(groups), 1))
    for (origin, bucket), part in groups:
        take = min(per, len(part))
        idx = rng.choice(part.index.to_numpy(), size=take, replace=False)
        parts.append(work.loc[idx])
    sample = pd.concat(parts, ignore_index=True) if parts else work.head(0)

    parent_texts = []
    for _, row in sample.iterrows():
        pid = str(row.get("parent_id") or "")
        if pid and pid in by_row.index:
            parent_texts.append(str(by_row.loc[pid]["text"]))
        else:
            parent_texts.append("")
    out = pd.DataFrame(
        {
            "row_id": sample["row_id"],
            "origin": sample["origin"],
            "label": sample["label"],
            "source_type": sample["source_type"],
            "generator": sample["generator"],
            "length": sample["length"],
            "text": sample["text"],
            "parent_text": parent_texts,
            "review_ok": "",
            "review_note": "",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("inspect_samples 저장: %s (%d행)", path, len(out))
    return out
