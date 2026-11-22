# 외부 HF AI/OOD 자산 로드·스키마 변환 (XDAC, LGC, Synthetic, WEBTEXT).
# trust_remote_code 미사용. 학습 CSV와 OOD CSV를 분리 생성.

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from data_build.quality import hangul_ratio, length_bucket, mask_pii
from data_build.schema import (
    SCHEMA_COLUMNS,
    empty_row,
    make_gen_group_id,
    make_human_group_id,
    make_row_id,
)
from preprocess import clean_text

logger = logging.getLogger("sns_ai_detector")


def _load_split(hf_id: str, split: str = "train", max_rows: Optional[int] = None) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(hf_id, split=split, trust_remote_code=False)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    return ds.to_pandas()


def export_xdac_ai(
    n: int = 500,
    seed: int = 43,
    out_path: Optional[Path] = None,
    apply_mask_pii: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """keepsteady/XDAC_LLM_comments_kor → ai_generated 스키마 행.

    원본 사람 댓글 컬럼은 없음(생성댓글만). group_id는 XDAC id 기반.
    동일 id의 사람 원문이 나중에 추가되면 같은 group 키를 쓸 수 있도록
    source 키를 xdac:{id} 로 고정한다.
    """
    raw = _load_split("keepsteady/XDAC_LLM_comments_kor", "train")
    if "생성댓글" not in raw.columns:
        raise RuntimeError(f"XDAC에 '생성댓글' 없음. cols={list(raw.columns)}")

    # 한국어만
    if "댓글언어" in raw.columns:
        raw = raw[raw["댓글언어"].astype(str).str.lower().str.contains("korean", na=False)]
    raw = raw.dropna(subset=["생성댓글"]).copy()
    raw = raw.sample(n=min(n, len(raw)), random_state=seed)

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for _, rec in raw.iterrows():
        text = clean_text(str(rec["생성댓글"]))
        if apply_mask_pii:
            text = mask_pii(text)
        if len(text) < 5 or hangul_ratio(text) < 0.25:
            skipped += 1
            continue
        xid = str(rec.get("id") or "")
        # 사람 원문이 있다면 동일 키로 group 공유 가능하도록 결정적 키 사용
        gid = make_human_group_id("xdac", original_id=xid or None, text=text)
        model = str(rec.get("config_model_name") or "xdac-llm")
        rows.append(
            empty_row(
                text=text,
                label=1,
                source_type="comment",
                group_id=gid,
                row_id=make_row_id(text, salt=f"xdac-{xid}"),
                origin="ai_generated",
                parent_id="",
                generator=f"xdac:{model}",
                prompt_version="xdac-v1",
                mod_ratio="",
            )
        )

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=SCHEMA_COLUMNS)
    stats = {"n_raw": int(len(raw)), "n_out": len(df), "skipped": skipped, "human_parent_col": False}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info("XDAC AI 저장: %s (%d)", out_path, len(df))
    return df, stats


def _chunk_text(text: str, min_len: int = 80, max_len: int = 280) -> List[str]:
    t = re.sub(r"\s+", " ", str(text)).strip()
    if len(t) < min_len:
        return []
    chunks = []
    i = 0
    while i < len(t) and len(chunks) < 8:
        piece = t[i : i + max_len].strip()
        if len(piece) >= min_len:
            chunks.append(piece)
        i += max_len
    return chunks


def build_ood_comment_benchmark(
    n_ai: int = 250,
    n_human: int = 250,
    seed: int = 43,
    exclude_texts: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """LGC_test AI + APEACH 사람 댓글 (학습 소스와 분리)."""
    exclude = {str(t).strip() for t in (exclude_texts or [])}

    lgc = _load_split("keepsteady/LGC_test", "train")
    ai_rows: List[Dict[str, Any]] = []
    for _, rec in lgc.iterrows():
        text = clean_text(str(rec.get("generated_comment") or ""))
        if len(text) < 5 or hangul_ratio(text) < 0.2:
            continue
        if text in exclude:
            continue
        model = str(rec.get("llm_model_selection") or "lgc")
        idx = str(rec.get("idx") or "")
        ai_rows.append(
            empty_row(
                text=text,
                label=1,
                source_type="comment",
                group_id=make_gen_group_id(f"lgc|{idx}|{model}"),
                row_id=make_row_id(text, salt=f"lgc-{idx}"),
                origin="ai_generated",
                parent_id="",
                generator=f"lgc:{model}",
                prompt_version="ood-lgc",
                mod_ratio="",
            )
        )
    ai_df = pd.DataFrame(ai_rows)
    if len(ai_df) > n_ai:
        ai_df = ai_df.sample(n=n_ai, random_state=seed)

    apeach = _load_split("nayohan/APEACH", "train")
    hum_rows: List[Dict[str, Any]] = []
    for _, rec in apeach.iterrows():
        text = clean_text(str(rec.get("text") or ""))
        if len(text) < 5 or hangul_ratio(text) < 0.2:
            continue
        if text in exclude:
            continue
        hum_rows.append(
            empty_row(
                text=text,
                label=0,
                source_type="comment",
                group_id=make_human_group_id("ood:apeach", text=text),
                row_id=make_row_id(text, salt="ood-apeach"),
                origin="human_public",
                parent_id="",
                generator="human",
                prompt_version="",
                mod_ratio="",
            )
        )
    hum_df = pd.DataFrame(hum_rows)
    if len(hum_df) > n_human:
        hum_df = hum_df.sample(n=n_human, random_state=seed)

    out = pd.concat([hum_df, ai_df], ignore_index=True)
    out["length_bucket"] = out["text"].astype(str).str.len().map(length_bucket)
    stats = {
        "n_human": int((out["label"] == 0).sum()),
        "n_ai": int((out["label"] == 1).sum()),
        "human_source": "nayohan/APEACH",
        "ai_source": "keepsteady/LGC_test",
    }
    return out, stats


def build_ood_longform_benchmark(
    n_ai: int = 150,
    n_human: int = 150,
    seed: int = 43,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """SyntheticText(AI) + WEBTEXT(사람)를 청킹한 장문/도메인 OOD."""
    syn = _load_split("HAERAE-HUB/KOREAN-SyntheticText-1.5B", "train", max_rows=800)
    web = _load_split("HAERAE-HUB/KOREAN-WEBTEXT", "train", max_rows=400)

    ai_rows: List[Dict[str, Any]] = []
    for i, rec in syn.iterrows():
        for j, chunk in enumerate(_chunk_text(str(rec.get("text") or ""))):
            if hangul_ratio(chunk) < 0.25:
                continue
            ai_rows.append(
                empty_row(
                    text=chunk,
                    label=1,
                    source_type="post",
                    group_id=make_gen_group_id(f"syn|{i}|{j}"),
                    row_id=make_row_id(chunk, salt=f"syn-{i}-{j}"),
                    origin="ai_generated",
                    parent_id="",
                    generator="haerae:synthetic-1.5b",
                    prompt_version="ood-long-ai",
                    mod_ratio="",
                )
            )
            if len(ai_rows) >= n_ai * 3:
                break
        if len(ai_rows) >= n_ai * 3:
            break
    ai_df = pd.DataFrame(ai_rows)
    if len(ai_df) > n_ai:
        ai_df = ai_df.sample(n=n_ai, random_state=seed)

    hum_rows: List[Dict[str, Any]] = []
    for i, rec in web.iterrows():
        for j, chunk in enumerate(_chunk_text(str(rec.get("text") or ""))):
            if hangul_ratio(chunk) < 0.25:
                continue
            hum_rows.append(
                empty_row(
                    text=chunk,
                    label=0,
                    source_type="post",
                    group_id=make_human_group_id("ood:webtext", text=chunk),
                    row_id=make_row_id(chunk, salt=f"web-{i}-{j}"),
                    origin="human_public",
                    parent_id="",
                    generator="human",
                    prompt_version="",
                    mod_ratio="",
                )
            )
            if len(hum_rows) >= n_human * 3:
                break
        if len(hum_rows) >= n_human * 3:
            break
    hum_df = pd.DataFrame(hum_rows)
    if len(hum_df) > n_human:
        hum_df = hum_df.sample(n=n_human, random_state=seed)

    out = pd.concat([hum_df, ai_df], ignore_index=True)
    out["length_bucket"] = out["text"].astype(str).str.len().map(length_bucket)
    stats = {
        "n_human": int((out["label"] == 0).sum()),
        "n_ai": int((out["label"] == 1).sum()),
        "human_source": "HAERAE-HUB/KOREAN-WEBTEXT (chunked)",
        "ai_source": "HAERAE-HUB/KOREAN-SyntheticText-1.5B (chunked)",
    }
    return out, stats


def prepare_v3_external_assets(
    data_dir: Path,
    xdac_n: int = 500,
    seed: int = 43,
) -> Dict[str, Any]:
    """XDAC 학습용 CSV + OOD 벤치마크 CSV 생성."""
    ext = data_dir / "external"
    ood = data_dir / "ood_benchmark"
    ext.mkdir(parents=True, exist_ok=True)
    ood.mkdir(parents=True, exist_ok=True)

    xdac_path = ext / "xdac_ai_generated.csv"
    xdac_df, xdac_stats = export_xdac_ai(n=xdac_n, seed=seed, out_path=xdac_path)

    # OOD는 학습 XDAC 텍스트와 겹치지 않게
    exclude = set(xdac_df["text"].astype(str)) if len(xdac_df) else set()
    comment_df, comment_stats = build_ood_comment_benchmark(
        n_ai=250, n_human=250, seed=seed, exclude_texts=exclude
    )
    comment_path = ood / "ood_comment.csv"
    comment_df.to_csv(comment_path, index=False, encoding="utf-8-sig")

    long_df, long_stats = build_ood_longform_benchmark(n_ai=150, n_human=150, seed=seed)
    long_path = ood / "ood_longform.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    # 합본 (선택 평가용)
    both = pd.concat([comment_df, long_df], ignore_index=True)
    both_path = ood / "ood_all.csv"
    both.to_csv(both_path, index=False, encoding="utf-8-sig")

    report = {
        "xdac": {"path": str(xdac_path), **xdac_stats},
        "ood_comment": {"path": str(comment_path), **comment_stats},
        "ood_longform": {"path": str(long_path), **long_stats},
        "ood_all": {"path": str(both_path), "n": len(both)},
    }
    return report
