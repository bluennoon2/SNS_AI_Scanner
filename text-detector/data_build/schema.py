# 통합 스키마, row_id/group_id 생성, group 일관성 검증.
# group_id는 반드시 결정적 해시(sha1) — hash()/행순서를 쓰지 않는다.

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

SCHEMA_COLUMNS = [
    "text",
    "label",
    "source_type",
    "group_id",
    "row_id",
    "origin",
    "parent_id",
    "generator",
    "prompt_version",
    "mod_ratio",
]

VALID_ORIGINS = {
    "human_public",
    "human_local",
    "ai_generated",
    "ai_paraphrased",
    "ai_hybrid",
}

DERIVED_ORIGINS = {"ai_paraphrased", "ai_hybrid"}

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U00002700-\U000027BF"
    "\U0001F600-\U0001F64F"
    "]+",
    flags=re.UNICODE,
)


def sha1_hex(text: str) -> str:
    """결정적 sha1 헥스."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_for_hash(text: str) -> str:
    """중복 병합용 정규화: NFC, 소문자, 공백 축약, 이모지 제거."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = _EMOJI_RE.sub("", t)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def make_row_id(text: str, salt: str = "") -> str:
    """내용 해시 기반 행 고유 ID."""
    payload = normalize_for_hash(text) + "|" + salt
    return "row_" + sha1_hex(payload)[:16]


def make_human_group_id(
    source: str,
    original_id: Optional[str] = None,
    text: Optional[str] = None,
) -> str:
    """사람 원문 group_id: grp_<sha1(source|원본ID 또는 정규화 text)[:12]>"""
    if original_id:
        key = f"{source}|{original_id}"
    else:
        key = f"{source}|{normalize_for_hash(text or '')}"
    return "grp_" + sha1_hex(key)[:12]


def make_gen_group_id(task_hash: str) -> str:
    """ai_generated 작업 단위 group_id."""
    return "grp_gen_" + sha1_hex(task_hash)[:12]


def make_task_id(
    parent_id: str,
    origin: str,
    prompt_version: str,
    generator: str,
    variant_seed: str,
) -> str:
    """생성 작업 결정적 task_id."""
    raw = f"{parent_id}|{origin}|{prompt_version}|{generator}|{variant_seed}"
    return sha1_hex(raw)


def empty_row(**overrides: Any) -> Dict[str, Any]:
    """스키마 기본 행."""
    row = {
        "text": "",
        "label": 0,
        "source_type": "post",
        "group_id": "",
        "row_id": "",
        "origin": "human_local",
        "parent_id": "",
        "generator": "human",
        "prompt_version": "",
        "mod_ratio": "",
    }
    row.update(overrides)
    return row


def ensure_dataframe(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """행 리스트를 스키마 컬럼 순 DataFrame으로."""
    df = pd.DataFrame(list(rows))
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[SCHEMA_COLUMNS].copy()


def assert_group_consistency(df: pd.DataFrame) -> None:
    """group_id 규칙 위반 시 ValueError.

    위반 예:
    - 파생 샘플의 group_id ≠ 부모의 group_id
    - parent_id가 데이터에 없음
    - 서로 무관한 부모가 같은 group_id를 공유 (서로 다른 사람 원문이 한 group)
    - 서로 다른 group에 동일 text가 존재
    """
    if df is None or len(df) == 0:
        return

    required = {"text", "label", "group_id", "row_id", "origin", "parent_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"스키마 컬럼 누락: {sorted(missing)}")

    work = df.copy()
    work["row_id"] = work["row_id"].astype(str)
    work["group_id"] = work["group_id"].astype(str)
    work["parent_id"] = work["parent_id"].fillna("").astype(str)
    work["origin"] = work["origin"].astype(str)

    by_row = work.set_index("row_id", drop=False)
    errors: List[str] = []

    # 1) parent_id 존재 + 파생 group == 부모 group
    derived = work[work["origin"].isin(DERIVED_ORIGINS)]
    for idx, row in derived.iterrows():
        pid = row["parent_id"]
        if not pid:
            errors.append(f"row_id={row['row_id']}: 파생 origin인데 parent_id 비어 있음")
            continue
        if pid not in by_row.index:
            errors.append(
                f"row_id={row['row_id']}: parent_id={pid} 가 데이터에 없음"
            )
            continue
        parent = by_row.loc[pid]
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]
        if str(row["group_id"]) != str(parent["group_id"]):
            errors.append(
                f"row_id={row['row_id']}: group_id={row['group_id']} ≠ "
                f"부모 group_id={parent['group_id']}"
            )

    # 2) 동일 text가 서로 다른 group에 존재
    text_groups = (
        work.groupby(work["text"].astype(str))["group_id"]
        .nunique()
    )
    multi = text_groups[text_groups > 1]
    for text, n in multi.items():
        errors.append(
            f"동일 text가 {n}개 group에 존재: {text[:40]!r}..."
        )

    # 3) 한 group 안에 서로 다른 사람 원문(부모 없는 human)이 2개 이상이면 경고성 에러
    #    (정규화 병합으로 의도적으로 합친 경우는 동일 normalize key를 쓰므로 OK —
    #     여기서는 '서로 다른 row_id의 human이 같은 group인데 정규화 text가 다른' 경우)
    humans = work[work["origin"].astype(str).str.startswith("human")]
    for gid, part in humans.groupby("group_id"):
        norms = part["text"].map(normalize_for_hash).unique().tolist()
        if len(norms) > 1:
            errors.append(
                f"group_id={gid}: 서로 다른 정규화 텍스트의 사람 원문이 한 group에 있음 "
                f"(n_norms={len(norms)})"
            )

    if errors:
        preview = "\n".join(f"- {e}" for e in errors[:20])
        more = f"\n... 외 {len(errors) - 20}건" if len(errors) > 20 else ""
        raise ValueError(
            f"group 일관성 위반 {len(errors)}건:\n{preview}{more}"
        )


def merge_near_duplicate_human_groups(df: pd.DataFrame) -> pd.DataFrame:
    """정규화 기준 동일/근접 중복 사람 글을 하나의 group으로 병합."""
    work = df.copy()
    human_mask = work["origin"].astype(str).str.startswith("human")
    if not human_mask.any():
        return work

    # 정규화 키 → 대표 group_id (첫 등장)
    key_to_gid: Dict[str, str] = {}
    new_gids: List[str] = []
    for i, row in work.iterrows():
        if not str(row["origin"]).startswith("human"):
            new_gids.append(str(row["group_id"]))
            continue
        key = normalize_for_hash(str(row["text"]))
        if key not in key_to_gid:
            key_to_gid[key] = str(row["group_id"])
        new_gids.append(key_to_gid[key])
    work["group_id"] = new_gids

    # 파생 샘플도 부모 group에 맞춤
    by_row = {
        str(r["row_id"]): str(r["group_id"])
        for _, r in work.iterrows()
    }
    for i, row in work.iterrows():
        if str(row["origin"]) in DERIVED_ORIGINS and row["parent_id"]:
            pid = str(row["parent_id"])
            if pid in by_row:
                work.at[i, "group_id"] = by_row[pid]
    return work
