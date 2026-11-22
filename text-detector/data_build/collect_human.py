# 사람(label=0) 데이터 수집·정규화.
# HF 후보 실패는 해당 소스만 건너뛴다. trust_remote_code 미사용.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

from data_build.quality import length_bucket, mask_pii, quota_sample
from data_build.schema import (
    empty_row,
    make_human_group_id,
    make_row_id,
    merge_near_duplicate_human_groups,
)
from preprocess import clean_text

logger = logging.getLogger("sns_ai_detector")


def load_human_sources_config(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("sources") or [])


def verify_hf_source(hf_id: str) -> Dict[str, Any]:
    """HfApi로 존재/접근성 확인. 실패해도 예외를 밖으로 안 던짐."""
    info: Dict[str, Any] = {
        "hf_id": hf_id,
        "ok": False,
        "error": None,
        "license": None,
    }
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        ds = api.dataset_info(hf_id)
        info["ok"] = True
        # license 필드가 있으면 기록
        card = getattr(ds, "cardData", None) or getattr(ds, "card_data", None) or {}
        if isinstance(card, dict):
            info["license"] = card.get("license")
        elif hasattr(card, "get"):
            info["license"] = card.get("license")
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("HF 소스 확인 실패 %s — %s", hf_id, info["error"])
    return info


def _normalize_filter_value(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v).strip().lower()
    if s in {"1", "1.0", "true", "yes"}:
        return "true"
    if s in {"0", "0.0", "false", "no"}:
        return "false"
    return s


def apply_row_filters(
    df: pd.DataFrame,
    filters: Optional[Sequence[Dict[str, Any]]],
) -> Tuple[pd.DataFrame, int]:
    """YAML row_filters 적용. allow 목록에 있는 값만 남긴다.

    예:
      row_filters:
        - column: label
          allow: ["normal"]
        - column: clean
          allow: [1, true]
    """
    if not filters:
        return df, 0
    before = len(df)
    work = df
    for spec in filters:
        raw_col = spec.get("column")
        # YAML 1.1 에서 OFF/Yes 등이 bool 이 될 수 있음 → 문자열로 복원
        if isinstance(raw_col, bool):
            col = "OFF" if raw_col is False else "ON"
        else:
            col = str(raw_col).strip() if raw_col is not None else ""
        if not col or col not in work.columns:
            logger.warning("row_filter 컬럼 없음/스킵: %r", raw_col)
            continue
        allow_raw = spec.get("allow")
        if allow_raw is None and "equals" in spec:
            allow_raw = [spec.get("equals")]
        allow = {_normalize_filter_value(x) for x in (allow_raw or [])}
        if not allow:
            continue
        mask = work[col].map(_normalize_filter_value).isin(allow)
        work = work.loc[mask]
    removed = before - len(work)
    return work.reset_index(drop=True), int(removed)


def _load_hf_split(
    hf_id: str,
    split: str,
    text_field: str,
    max_rows: Optional[int] = None,
    row_filters: Optional[Sequence[Dict[str, Any]]] = None,
    load_cap: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """HF split 로드. 필터 후 부족할 수 있어 load_cap으로 여유 로드."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets 미설치 — HF 소스 건너뜀: %s", hf_id)
        return None
    try:
        # trust_remote_code 사용 금지
        ds = load_dataset(hf_id, split=split)
        # 필터 전 여유 로드 (기본: max_rows*4 또는 20000)
        if load_cap is None:
            if max_rows is not None:
                load_cap = min(len(ds), max(max_rows * 4, max_rows + 500))
            else:
                load_cap = None
        if load_cap is not None:
            ds = ds.select(range(min(load_cap, len(ds))))
        df = ds.to_pandas()
        if text_field not in df.columns:
            logger.warning(
                "HF %s: text_field=%s 없음. 컬럼=%s — 건너뜀",
                hf_id,
                text_field,
                list(df.columns),
            )
            return None
        df, n_filt = apply_row_filters(df, row_filters)
        if n_filt:
            logger.info("HF %s row_filters 제거 %d행 → 남음 %d", hf_id, n_filt, len(df))
        if max_rows is not None and len(df) > max_rows:
            df = df.head(max_rows).reset_index(drop=True)
        return df
    except Exception as exc:
        logger.warning("HF 로드 실패 %s — %s (건너뜀)", hf_id, exc)
        return None


def rows_from_dataframe(
    df: pd.DataFrame,
    text_col: str,
    source_type: str,
    origin: str,
    source_key: str,
    group_col: Optional[str] = None,
    apply_mask_pii: bool = True,
    min_chars: int = 3,
    min_hangul_ratio: float = 0.2,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """DataFrame → 스키마 행. 필터 사유별 카운트."""
    from data_build.quality import hangul_ratio

    removed = {
        "empty_short": 0,
        "low_hangul": 0,
        "ad_repeat": 0,
    }
    rows: List[Dict[str, Any]] = []
    for i, rec in df.iterrows():
        raw = rec.get(text_col)
        if pd.isna(raw):
            removed["empty_short"] += 1
            continue
        text = clean_text(str(raw))
        if apply_mask_pii:
            text = mask_pii(text)
        if len(text) < min_chars:
            removed["empty_short"] += 1
            continue
        if hangul_ratio(text) < min_hangul_ratio:
            removed["low_hangul"] += 1
            continue
        # 광고성 반복 (같은 문자 과다)
        if len(set(text)) <= 2 and len(text) >= 8:
            removed["ad_repeat"] += 1
            continue

        orig_id = None
        if group_col and group_col in df.columns and pd.notna(rec.get(group_col)):
            orig_id = str(rec.get(group_col))
        gid = make_human_group_id(source_key, original_id=orig_id, text=text)
        rid = make_row_id(text, salt=source_key)
        rows.append(
            empty_row(
                text=text,
                label=0,
                source_type=source_type,
                group_id=gid,
                row_id=rid,
                origin=origin,
                parent_id="",
                generator="human",
                prompt_version="",
                mod_ratio="",
            )
        )
    return rows, removed


def collect_from_hf_sources(
    config_path: Path,
    max_per_source: Optional[int] = 5000,
    apply_mask_pii: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """YAML 후보에서 사람 데이터 수집."""
    sources = load_human_sources_config(config_path)
    all_rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"sources": [], "removed": {}}
    removed_total: Dict[str, int] = {}

    for src in sources:
        entry = {
            "hf_id": src.get("hf_id"),
            "enabled": bool(src.get("enabled", True)),
            "allow_toxic": bool(src.get("allow_toxic", False)),
            "verified": bool(src.get("verified", False)),
            "collected_before": src.get("collected_before"),
            "status": "skipped",
            "n_loaded": 0,
            "license": src.get("license"),
            "error": None,
        }
        if not src.get("enabled", True):
            entry["status"] = "disabled"
            report["sources"].append(entry)
            continue
        if src.get("allow_toxic") and not src.get("allow_toxic_forced"):
            # allow_toxic true여도 enabled면 사용하되 리포트에 표시
            entry["toxic_warning"] = True

        verify = verify_hf_source(str(src["hf_id"]))
        entry["license"] = verify.get("license") or src.get("license")
        if not verify["ok"]:
            entry["status"] = "verify_failed"
            entry["error"] = verify.get("error")
            report["sources"].append(entry)
            continue

        filters = src.get("row_filters") or []
        raw = _load_hf_split(
            str(src["hf_id"]),
            str(src.get("split", "train")),
            str(src.get("text_field", "text")),
            max_rows=max_per_source,
            row_filters=filters if isinstance(filters, list) else [],
        )
        if raw is None:
            entry["status"] = "load_failed"
            entry["error"] = "load_dataset 실패 또는 필드 불일치"
            report["sources"].append(entry)
            continue

        rows, removed = rows_from_dataframe(
            raw,
            text_col=str(src.get("text_field", "text")),
            source_type=str(src.get("source_type", "post")),
            origin="human_public",
            source_key=f"hf:{src['hf_id']}",
            group_col=src.get("group_col"),
            apply_mask_pii=apply_mask_pii,
        )
        for k, v in removed.items():
            removed_total[k] = removed_total.get(k, 0) + v
        all_rows.extend(rows)
        entry["status"] = "ok"
        entry["n_loaded"] = len(rows)
        entry["row_filters"] = filters
        report["sources"].append(entry)
        logger.info("HF %s → %d행", src["hf_id"], len(rows))

    report["removed"] = removed_total
    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    return df, report


def collect_from_local_files(
    files: Sequence[str],
    text_col: str = "text",
    source_type: str = "post",
    group_col: Optional[str] = None,
    apply_mask_pii: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """사용자가 보증한 로컬 사람 글."""
    logger.info(
        "로컬 파일은 사용자가 직접 확인한 사람 작성 글로 간주합니다: %s",
        list(files),
    )
    all_rows: List[Dict[str, Any]] = []
    removed_total: Dict[str, int] = {}
    file_stats = []

    for fp in files:
        path = Path(fp)
        if not path.exists():
            logger.warning("로컬 파일 없음 — 건너뜀: %s", path)
            file_stats.append({"path": str(path), "status": "missing", "n": 0})
            continue
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, encoding="utf-8-sig")
        elif path.suffix.lower() in {".jsonl", ".json"}:
            df = pd.read_json(path, lines=True, encoding="utf-8")
        else:
            logger.warning("지원하지 않는 확장자 — 건너뜀: %s", path)
            file_stats.append({"path": str(path), "status": "bad_ext", "n": 0})
            continue
        if text_col not in df.columns:
            raise ValueError(
                f"{path}: '{text_col}' 컬럼이 없습니다. 현재={list(df.columns)}"
            )
        rows, removed = rows_from_dataframe(
            df,
            text_col=text_col,
            source_type=source_type,
            origin="human_local",
            source_key=f"local:{path.name}",
            group_col=group_col,
            apply_mask_pii=apply_mask_pii,
        )
        for k, v in removed.items():
            removed_total[k] = removed_total.get(k, 0) + v
        all_rows.extend(rows)
        file_stats.append({"path": str(path), "status": "ok", "n": len(rows)})

    df_out = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    return df_out, {"files": file_stats, "removed": removed_total}


def collect_human(
    config_path: Path,
    extra_files: Optional[Sequence[str]] = None,
    extra_text_col: str = "text",
    extra_source_type: str = "post",
    extra_group_col: Optional[str] = None,
    target_n: Optional[int] = None,
    length_mix: Sequence[float] = (0.3, 0.4, 0.3),
    apply_mask_pii: bool = True,
    seed: int = 42,
    max_per_hf_source: Optional[int] = 5000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """HF + 로컬 수집 → 병합 → 쿼터 샘플링."""
    hf_df, hf_report = collect_from_hf_sources(
        config_path,
        max_per_source=max_per_hf_source,
        apply_mask_pii=apply_mask_pii,
    )
    local_df = pd.DataFrame()
    local_report: Dict[str, Any] = {}
    if extra_files:
        local_df, local_report = collect_from_local_files(
            extra_files,
            text_col=extra_text_col,
            source_type=extra_source_type,
            group_col=extra_group_col,
            apply_mask_pii=apply_mask_pii,
        )

    frames = [f for f in [hf_df, local_df] if f is not None and len(f) > 0]
    if not frames:
        # 실패 표
        lines = []
        for s in hf_report.get("sources", []):
            lines.append(
                f"  - {s.get('hf_id')}: status={s.get('status')} error={s.get('error')}"
            )
        msg = (
            "사용 가능한 사람 데이터가 없습니다.\n"
            "실패한 HF 소스:\n" + ("\n".join(lines) if lines else "  (없음)") + "\n"
            "해결: --extra_human_files 로 로컬 CSV/JSONL을 추가하거나 "
            "configs/human_sources.yaml 의 hf_id를 검증·교체하세요."
        )
        raise RuntimeError(msg)

    merged = pd.concat(frames, ignore_index=True)
    merged = merge_near_duplicate_human_groups(merged)
    merged["length_bucket"] = merged["text"].astype(str).str.len().map(length_bucket)

    # 쿼터: source_type × bucket
    if target_n is not None and target_n > 0:
        stypes = sorted(merged["source_type"].astype(str).unique())
        buckets = ["<20", "20~100", ">100"]
        mix = list(length_mix)
        if len(mix) != 3:
            mix = [0.3, 0.4, 0.3]
        # source별 균등, 버킷은 mix
        per_src = max(1, target_n // max(len(stypes), 1))
        per_cell = {}
        for st in stypes:
            for b, ratio in zip(buckets, mix):
                per_cell[(st, b)] = max(1, int(round(per_src * ratio)))
        sampled, quota_rep = quota_sample(merged, per_cell, seed=seed)
        # 부족하면 남은 풀에서 채움
        if len(sampled) < target_n:
            remain = merged.loc[~merged["row_id"].isin(sampled["row_id"])]
            need = target_n - len(sampled)
            if len(remain) > 0:
                extra = remain.sample(
                    n=min(need, len(remain)), random_state=seed
                )
                sampled = pd.concat([sampled, extra], ignore_index=True)
        merged = sampled.head(target_n).reset_index(drop=True)
    else:
        quota_rep = {"shortfalls": {}, "n_sampled": len(merged)}

    report = {
        "hf": hf_report,
        "local": local_report,
        "quota": quota_rep,
        "n_human": len(merged),
        "by_source_type": merged["source_type"].value_counts().to_dict()
        if len(merged)
        else {},
        "by_length_bucket": merged["length_bucket"].value_counts().to_dict()
        if len(merged)
        else {},
    }
    return merged, report
