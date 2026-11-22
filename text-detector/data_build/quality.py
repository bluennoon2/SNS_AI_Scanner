# 생성물 검증 필터, 중복/라벨 충돌 제거, 지름길 베이스라인.

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("sns_ai_detector")

_HANGUL_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
_CODE_FENCE_RE = re.compile(r"^```[\s\S]*?```$", re.MULTILINE)
_QUOTE_WRAP_RE = re.compile(r'^[\"“‘\']+|["”’\']+$')

DEFAULT_META_PREFIXES = [
    "물론입니다",
    "다음은",
    "아래와 같이",
    "요청하신",
    "네,",
    "알겠습니다",
    "여기",
]

DEFAULT_REFUSALS = [
    "죄송하지만",
    "도와드릴 수 없습니다",
    "제공할 수 없습니다",
    "i can't assist",
    "i'm unable to",
]


def hangul_ratio(text: str) -> float:
    if not text:
        return 0.0
    n = sum(1 for ch in text if _HANGUL_RE.match(ch))
    return n / max(len(text), 1)


def strip_meta_wrappers(text: str, prefixes: Optional[Sequence[str]] = None) -> str:
    """서두/코드펜스/따옴표 감싸기 제거."""
    t = (text or "").strip()
    t = _CODE_FENCE_RE.sub("", t).strip()
    # 첫 줄이 메타 서두면 제거
    prefs = list(prefixes or DEFAULT_META_PREFIXES)
    lines = t.splitlines()
    if lines:
        first = lines[0].strip()
        if any(first.startswith(p) for p in prefs):
            lines = lines[1:]
            t = "\n".join(lines).strip()
    t = _QUOTE_WRAP_RE.sub("", t).strip()
    return t


def is_refusal(text: str, patterns: Optional[Sequence[str]] = None) -> bool:
    low = (text or "").lower()
    for p in patterns or DEFAULT_REFUSALS:
        if p.lower() in low:
            return True
    return False


def char_jaccard(a: str, b: str, n: int = 3) -> float:
    """문자 n-gram Jaccard 유사도."""

    def grams(s: str) -> set:
        s = re.sub(r"\s+", "", s)
        if len(s) < n:
            return {s} if s else set()
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    ga, gb = grams(a), grams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def validate_generated_row(
    text: str,
    origin: str,
    parent_text: str = "",
    min_hangul_ratio: float = 0.3,
    min_len: int = 5,
    max_len: int = 500,
    target_min: Optional[int] = None,
    target_max: Optional[int] = None,
    paraphrase_sim_low: float = 0.05,
    paraphrase_sim_high: float = 0.85,
) -> Tuple[bool, str, str]:
    """생성물 검증. Returns (ok, cleaned_text, reason)."""
    cleaned = strip_meta_wrappers(text)
    if not cleaned or len(cleaned.strip()) < min_len:
        return False, cleaned, "empty_or_too_short"
    if is_refusal(cleaned):
        return False, cleaned, "refusal"
    if hangul_ratio(cleaned) < min_hangul_ratio:
        return False, cleaned, "low_hangul_ratio"
    if len(cleaned) > max_len:
        return False, cleaned, "too_long"
    if target_min is not None and len(cleaned) < target_min:
        return False, cleaned, "below_target_length"
    if target_max is not None and len(cleaned) > target_max:
        return False, cleaned, "above_target_length"

    if origin in {"ai_paraphrased", "ai_hybrid"} and parent_text:
        if cleaned.strip() == parent_text.strip():
            return False, cleaned, "identical_to_parent"
        sim = char_jaccard(cleaned, parent_text)
        if sim >= paraphrase_sim_high:
            return False, cleaned, "too_similar_to_parent"
        if origin == "ai_paraphrased" and sim <= paraphrase_sim_low:
            return False, cleaned, "too_different_from_parent"

    # 메타 서두가 남아 있으면
    for p in DEFAULT_META_PREFIXES:
        if cleaned.startswith(p):
            return False, cleaned, "meta_prefix"

    return True, cleaned, "ok"


def remove_duplicates_and_conflicts(
    df: pd.DataFrame,
    conflicts_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, int], pd.DataFrame]:
    """동일 text 중복 제거. 서로 다른 label이면 둘 다 제거."""
    stats = {"exact_dup_removed": 0, "label_conflict_removed": 0}
    if len(df) == 0:
        return df, stats, pd.DataFrame()

    work = df.copy()
    work["_text_key"] = work["text"].astype(str)

    conflict_rows: List[pd.DataFrame] = []
    keep_mask = pd.Series(True, index=work.index)

    for text, part in work.groupby("_text_key"):
        labels = set(part["label"].astype(int).tolist())
        if len(labels) > 1:
            keep_mask.loc[part.index] = False
            conflict_rows.append(part.drop(columns=["_text_key"], errors="ignore"))
            stats["label_conflict_removed"] += len(part)
        elif len(part) > 1:
            # 첫 행만 유지
            drop_idx = part.index[1:]
            keep_mask.loc[drop_idx] = False
            stats["exact_dup_removed"] += len(drop_idx)

    conflicts = (
        pd.concat(conflict_rows, ignore_index=True) if conflict_rows else pd.DataFrame()
    )
    if conflicts_path is not None and len(conflicts) > 0:
        conflicts_path.parent.mkdir(parents=True, exist_ok=True)
        conflicts.to_csv(conflicts_path, index=False, encoding="utf-8-sig")
        logger.warning(
            "라벨 충돌 %d행 제거 → %s", len(conflicts), conflicts_path
        )

    out = work.loc[keep_mask].drop(columns=["_text_key"]).reset_index(drop=True)
    return out, stats, conflicts


def mask_pii(text: str) -> str:
    """전화번호·이메일·주민번호 형태 마스킹. URL/멘션도 동일 규칙으로."""
    if not text:
        return text
    t = text
    t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", t)
    t = re.sub(r"\b01[016789]-?\d{3,4}-?\d{4}\b", "[PHONE]", t)
    t = re.sub(r"\b\d{6}-?[1-4]\d{6}\b", "[RRN]", t)
    t = re.sub(r"https?://\S+|www\.\S+", "[URL]", t)
    t = re.sub(r"@[\w가-힣.]+", "[MENTION]", t)
    return t


def length_bucket(n: int) -> str:
    if n < 20:
        return "<20"
    if n <= 100:
        return "20~100"
    return ">100"


def shortcut_baseline_auc(
    df: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, Any]:
    """메타 특징 / 문자 n-gram 로지스틱 회귀 AUC (group 분할).

    AUC가 높으면 라벨-형식 교락 경고.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    result: Dict[str, Any] = {
        "meta_auc": None,
        "ngram_auc": None,
        "warning": None,
    }
    if len(df) < 30 or df["label"].nunique() < 2:
        result["warning"] = "표본 부족으로 지름길 점검 생략"
        return result

    work = df.copy()
    work["length"] = work["text"].astype(str).str.len()
    work["n_emoji"] = work["text"].astype(str).apply(
        lambda t: len(re.findall(r"[\U0001F300-\U0001F9FF]", t))
    )
    work["n_hashtag"] = work["text"].astype(str).str.count(r"#")
    work["n_newline"] = work["text"].astype(str).str.count("\n")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    try:
        tr, te = next(
            gss.split(work, work["label"], work["group_id"].astype(str))
        )
    except Exception:
        result["warning"] = "group 분할 실패로 지름길 점검 생략"
        return result

    train, test = work.iloc[tr], work.iloc[te]
    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        result["warning"] = "분할 후 단일 클래스 — 지름길 점검 생략"
        return result

    # meta
    meta_cols = ["length", "n_emoji", "n_hashtag", "n_newline"]
    Xtr = train[meta_cols].values
    Xte = test[meta_cols].values
    clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(Xtr, train["label"].astype(int))
    proba = clf.predict_proba(Xte)[:, 1]
    meta_auc = float(roc_auc_score(test["label"].astype(int), proba))
    result["meta_auc"] = meta_auc

    # char n-gram
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=5000)
    Xtr2 = vec.fit_transform(train["text"].astype(str))
    Xte2 = vec.transform(test["text"].astype(str))
    clf2 = LogisticRegression(max_iter=500, random_state=seed)
    clf2.fit(Xtr2, train["label"].astype(int))
    proba2 = clf2.predict_proba(Xte2)[:, 1]
    ngram_auc = float(roc_auc_score(test["label"].astype(int), proba2))
    result["ngram_auc"] = ngram_auc

    if meta_auc >= 0.8:
        result["warning"] = (
            f"메타 특징만으로 AUC={meta_auc:.3f} ≥ 0.8 — "
            "라벨이 길이·형식과 교락되어 있을 수 있습니다. 분포를 재조정하세요."
        )
        logger.warning("%s", result["warning"])
    return result


def quota_sample(
    df: pd.DataFrame,
    per_cell: Dict[Tuple[str, str], int],
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """source_type × length_bucket 칸별 목표 건수 샘플링."""
    rng = np.random.default_rng(seed)
    work = df.copy()
    if "length_bucket" not in work.columns:
        work["length_bucket"] = work["text"].astype(str).str.len().map(length_bucket)

    parts = []
    shortfalls: Dict[str, int] = {}
    for (stype, bucket), n_want in per_cell.items():
        mask = (work["source_type"].astype(str) == stype) & (
            work["length_bucket"].astype(str) == bucket
        )
        pool = work.loc[mask]
        if len(pool) == 0:
            shortfalls[f"{stype}|{bucket}"] = n_want
            continue
        if len(pool) <= n_want:
            parts.append(pool)
            if len(pool) < n_want:
                shortfalls[f"{stype}|{bucket}"] = n_want - len(pool)
        else:
            idx = rng.choice(pool.index.to_numpy(), size=n_want, replace=False)
            parts.append(work.loc[idx])

    out = pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0].copy()
    report = {"shortfalls": shortfalls, "n_sampled": len(out)}
    return out, report
