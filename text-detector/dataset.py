# 이 파일의 역할:
# 데이터 로딩·검증·전처리·분할(누수 검사 포함)·토크나이징·길이 버킷 부여를 담당한다.
# data_report / split_report 저장, StratifiedGroupKFold 우선 분할, 토큰화 캐시를 지원한다.

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from preprocess import clean_text
from utils import save_json

logger = logging.getLogger("sns_ai_detector")

REQUIRED_COLUMNS = ("text", "label")
OPTIONAL_COLUMNS = ("source_type", "group_id")

# 대형 그룹 경고 임계값 (전체 대비)
LARGE_GROUP_RATIO = 0.30
# 클래스 비율 편차 경고 (percentage points)
CLASS_RATIO_GAP_WARN = 0.10


def assign_length_bucket(
    length: int, bins: Sequence[int] = (20, 100)
) -> str:
    """글자 수 기준 길이 버킷 라벨을 반환한다.

    Args:
        length: 텍스트 글자 수.
        bins: 경계값. 기본 [20, 100] → '<20', '20~100', '>100'.

    Returns:
        버킷 이름 문자열.
    """
    bins = list(bins)
    if length < bins[0]:
        return f"<{bins[0]}"
    for i in range(len(bins) - 1):
        if bins[i] <= length <= bins[i + 1]:
            return f"{bins[i]}~{bins[i + 1]}"
    return f">{bins[-1]}"


def coerce_label(value: Any, row_idx: int) -> int:
    """라벨을 안전하게 정수 0/1로 변환한다.

    허용: 0, 1, "0", "1", 0.0, 1.0, True/False.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        iv = int(value)
    elif isinstance(value, (float, np.floating)):
        if float(value) != float(value):  # NaN
            raise ValueError(f"행 {row_idx}: label이 NaN입니다. 0 또는 1로 수정하세요.")
        if float(value) not in (0.0, 1.0):
            raise ValueError(
                f"행 {row_idx}: label={value!r} 는 {{0,1}}이 아닙니다. "
                f"사람=0, AI=1 만 허용됩니다."
            )
        iv = int(value)
    elif isinstance(value, str):
        s = value.strip()
        if s in {"0", "1"}:
            iv = int(s)
        else:
            try:
                fv = float(s)
            except ValueError as exc:
                raise ValueError(
                    f"행 {row_idx}: label={value!r} 를 정수로 변환할 수 없습니다."
                ) from exc
            if fv not in (0.0, 1.0):
                raise ValueError(
                    f"행 {row_idx}: label={value!r} 는 {{0,1}}이 아닙니다."
                )
            iv = int(fv)
    else:
        raise ValueError(
            f"행 {row_idx}: label 타입 {type(value).__name__}={value!r} 를 "
            f"변환할 수 없습니다."
        )
    if iv not in (0, 1):
        raise ValueError(
            f"행 {row_idx}: label={iv} 는 {{0,1}} 외 값입니다."
        )
    return iv


def file_content_hash(path: Union[str, Path], max_bytes: int = 8_000_000) -> str:
    """데이터 파일 해시(캐시 키용). 대용량은 앞부분+크기+mtime으로 근사."""
    path = Path(path)
    h = hashlib.sha256()
    h.update(str(path.resolve()).encode("utf-8"))
    stat = path.stat()
    h.update(str(stat.st_size).encode())
    h.update(str(int(stat.st_mtime)).encode())
    with path.open("rb") as f:
        chunk = f.read(max_bytes)
        h.update(chunk)
    return h.hexdigest()[:16]


def load_raw_dataframe(
    data_path: Union[str, Path],
    max_rows: Optional[int] = None,
    skip_bad_lines: bool = False,
) -> pd.DataFrame:
    """CSV 또는 JSONL을 확장자로 판별해 DataFrame으로 로드한다.

    - 인코딩: utf-8-sig 우선.
    - JSONL 깨진 줄: 줄 번호 보고. skip_bad_lines면 건너뜀.
    - max_rows: 디버깅용 상한.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {path}\n"
            f"경로를 확인하거나 --data_path 를 올바른 파일로 지정하세요."
        )

    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", nrows=max_rows)
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"CSV 인코딩 오류({path}). utf-8-sig로 저장한 뒤 다시 시도하세요.\n"
                f"원인: {exc}"
            ) from exc
    elif suffix in {".jsonl", ".json"}:
        rows: List[Dict[str, Any]] = []
        bad: List[str] = []
        with path.open("r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    msg = f"JSONL {i}행 파싱 실패: {exc.msg}"
                    if skip_bad_lines:
                        bad.append(msg)
                        logger.warning("%s — 건너뜁니다 (--skip_bad_lines).", msg)
                        continue
                    raise ValueError(
                        f"{msg}\n해당 줄을 고치거나 --skip_bad_lines 를 사용하세요."
                    ) from exc
                if max_rows is not None and len(rows) >= max_rows:
                    break
        if bad:
            logger.warning("JSONL 깨진 줄 %d건 건너뜀", len(bad))
        df = pd.DataFrame(rows)
    else:
        raise ValueError(
            f"지원하지 않는 확장자입니다: '{suffix}'. "
            f"CSV(.csv) 또는 JSONL(.jsonl)만 사용할 수 있습니다."
        )

    if max_rows is not None:
        logger.info("max_rows=%d 적용 → %d행 로드", max_rows, len(df))
    return df


def find_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """같은 text에 서로 다른 label이 붙은 충돌 샘플을 반환한다."""
    if len(df) == 0:
        return pd.DataFrame(columns=list(df.columns) + ["n_labels"])
    grouped = df.groupby("text")["label"].nunique()
    conflict_texts = set(grouped[grouped > 1].index.tolist())
    if not conflict_texts:
        return pd.DataFrame(columns=list(df.columns))
    out = df[df["text"].isin(conflict_texts)].copy()
    out["n_labels"] = out["text"].map(grouped)
    return out.reset_index(drop=True)


def build_data_report(
    df: pd.DataFrame,
    removed: Dict[str, int],
    conflicts: pd.DataFrame,
) -> Dict[str, Any]:
    """data_report.json 용 요약 dict."""
    n = len(df)
    lengths = df["length"] if n else pd.Series(dtype=float)
    label_counts = df["label"].value_counts().to_dict() if n else {}
    dup_count = int(df["text"].duplicated().sum()) if n else 0
    report: Dict[str, Any] = {
        "n_rows": n,
        "removed": removed,
        "class_distribution": {
            "human_0": int(label_counts.get(0, 0)),
            "ai_1": int(label_counts.get(1, 0)),
        },
        "length": {
            "mean": float(lengths.mean()) if n else None,
            "median": float(lengths.median()) if n else None,
            "p5": float(lengths.quantile(0.05)) if n else None,
            "p95": float(lengths.quantile(0.95)) if n else None,
            "min": int(lengths.min()) if n else None,
            "max": int(lengths.max()) if n else None,
        },
        "exact_duplicate_texts": dup_count,
        "label_conflicts": int(len(conflicts)),
        "source_type_missing_rate": None,
        "group_id_missing_rate": None,
        "by_source_type": {},
    }
    if "source_type" in df.columns:
        miss = float(df["source_type"].isna().mean()) if n else 1.0
        report["source_type_missing_rate"] = miss
        by_src: Dict[str, Any] = {}
        for src, part in df.groupby(df["source_type"].astype(str), dropna=False):
            lc = part["label"].value_counts().to_dict()
            by_src[str(src)] = {
                "n": int(len(part)),
                "human_0": int(lc.get(0, 0)),
                "ai_1": int(lc.get(1, 0)),
            }
        report["by_source_type"] = by_src
    if "group_id" in df.columns:
        report["group_id_missing_rate"] = (
            float(df["group_id"].isna().mean()) if n else 1.0
        )
    return report


def data_report_to_markdown(report: Dict[str, Any]) -> str:
    """data_report Markdown."""
    lines = ["# Data Report", ""]
    lines.append(f"- n_rows: {report.get('n_rows')}")
    lines.append(f"- removed: `{report.get('removed')}`")
    cd = report.get("class_distribution", {})
    lines.append(
        f"- class: human={cd.get('human_0')} / AI={cd.get('ai_1')}"
    )
    ln = report.get("length", {})
    lines.append(
        f"- length mean/median/p5/p95: "
        f"{ln.get('mean')}/{ln.get('median')}/{ln.get('p5')}/{ln.get('p95')}"
    )
    lines.append(f"- exact_duplicate_texts: {report.get('exact_duplicate_texts')}")
    lines.append(f"- label_conflicts: {report.get('label_conflicts')}")
    lines.append(
        f"- source_type missing rate: {report.get('source_type_missing_rate')}"
    )
    lines.append(
        f"- group_id missing rate: {report.get('group_id_missing_rate')}"
    )
    lines.append("")
    return "\n".join(lines)


def validate_and_clean_dataframe(
    df: pd.DataFrame,
    collapse_repeats: bool = False,
    max_repeat: Optional[int] = None,
    length_bins: Sequence[int] = (20, 100),
    newline_token: bool = False,
    report_dir: Optional[Union[str, Path]] = None,
    require_both_classes: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """필수 컬럼/라벨 검증 후 텍스트를 정제하고 빈 행을 제거한다.

    Returns:
        (정제 DataFrame, data_report dict)
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"필수 컬럼이 없습니다: {missing}\n"
            f"CSV/JSONL에 'text', 'label' 컬럼을 추가하세요. "
            f"현재 컬럼: {list(df.columns)}"
        )

    work = df.copy()
    removed: Dict[str, int] = {
        "empty_or_nan_text": 0,
        "empty_after_clean": 0,
        "non_str_text_coerced": 0,
    }

    # text 타입 강제
    coerced = 0
    new_texts: List[Any] = []
    for v in work["text"].tolist():
        if pd.isna(v):
            new_texts.append(v)
        elif not isinstance(v, str):
            coerced += 1
            new_texts.append(str(v))
        else:
            new_texts.append(v)
    if coerced:
        logger.warning("text가 str이 아닌 값 %d건을 str()로 변환했습니다.", coerced)
        removed["non_str_text_coerced"] = coerced
    work["text"] = new_texts

    # 라벨 변환 (행 번호 포함)
    labels: List[int] = []
    for i, v in enumerate(work["label"].tolist()):
        labels.append(coerce_label(v, i))
    work["label"] = labels

    n_before = len(work)
    # NaN / 공백만
    pre_empty = work["text"].isna() | (
        work["text"].astype(str).str.strip().str.len() == 0
    )
    removed["empty_or_nan_text"] = int(pre_empty.sum())

    work["text"] = [
        clean_text(
            "" if pd.isna(t) else str(t),
            collapse_repeats=collapse_repeats,
            max_repeat=max_repeat,
            newline_token=newline_token,
        )
        for t in work["text"].tolist()
    ]
    post_empty = work["text"].str.len() == 0
    # clean 후 빈 문자열 (원래 비어있지 않았는데 빈 경우 포함)
    removed["empty_after_clean"] = int((~pre_empty & post_empty).sum())
    drop_mask = post_empty
    work = work.loc[~drop_mask].reset_index(drop=True)
    logger.info(
        "전처리: 전체 %d행 중 빈 텍스트 %d행 제거 → %d행 남음",
        n_before,
        int(drop_mask.sum()),
        len(work),
    )

    if len(work) == 0:
        raise ValueError(
            "전처리 후 남은 데이터가 없습니다. "
            "원본 text가 비어있거나 과도하게 정제되지 않았는지 확인하세요."
        )

    if require_both_classes:
        uniq = set(work["label"].unique().tolist())
        if uniq != {0, 1} and len(uniq) < 2:
            raise ValueError(
                f"라벨이 단일 클래스입니다: {uniq}\n"
                f"이진 분류 학습을 위해 사람(0)과 AI(1) 샘플이 모두 필요합니다."
            )

    work["length"] = work["text"].str.len().astype(int)
    work["length_bucket"] = work["length"].apply(
        lambda L: assign_length_bucket(int(L), length_bins)
    )

    if "source_type" not in work.columns:
        work["source_type"] = None
    else:
        miss = float(work["source_type"].isna().mean())
        logger.info("source_type 결측률: %.1f%%", 100.0 * miss)
    if "group_id" not in work.columns:
        work["group_id"] = None
    else:
        miss = float(work["group_id"].isna().mean())
        logger.info("group_id 결측률: %.1f%%", 100.0 * miss)

    conflicts = find_label_conflicts(work)
    if len(conflicts) > 0:
        logger.warning(
            "라벨 충돌 %d행 (같은 text에 다른 label). "
            "--dedup 시 충돌 샘플은 모두 제거됩니다.",
            len(conflicts),
        )

    _log_data_summary(work)
    report = build_data_report(work, removed, conflicts)

    if report_dir is not None:
        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_json(report, out / "data_report.json")
        (out / "data_report.md").write_text(
            data_report_to_markdown(report), encoding="utf-8"
        )
        if len(conflicts) > 0:
            conflicts.to_csv(
                out / "data_conflicts.csv", index=False, encoding="utf-8-sig"
            )
        logger.info("data_report 저장: %s", out)

    return work, report


def _log_data_summary(df: pd.DataFrame) -> None:
    """클래스/길이/중복 요약을 로그로 출력한다."""
    n = len(df)
    label_counts = df["label"].value_counts().to_dict()
    n_human = int(label_counts.get(0, 0))
    n_ai = int(label_counts.get(1, 0))
    dup_count = int(df["text"].duplicated().sum())
    lengths = df["length"]
    logger.info(
        "데이터 요약 | n=%d | human(0)=%d (%.1f%%) | AI(1)=%d (%.1f%%)",
        n,
        n_human,
        100.0 * n_human / n,
        n_ai,
        100.0 * n_ai / n,
    )
    logger.info(
        "길이 분포 | min=%d median=%d mean=%.1f max=%d p5=%.0f p95=%.0f",
        int(lengths.min()),
        int(lengths.median()),
        float(lengths.mean()),
        int(lengths.max()),
        float(lengths.quantile(0.05)),
        float(lengths.quantile(0.95)),
    )
    logger.info("길이 버킷 분포: %s", df["length_bucket"].value_counts().to_dict())
    logger.info("중복 text 건수(첫 등장 제외): %d", dup_count)


def deduplicate_texts(df: pd.DataFrame) -> pd.DataFrame:
    """동일 text 중복을 제거한다.

    라벨 충돌(같은 text, 다른 label)이 있으면 해당 text의 모든 행을 제거한다.
    충돌이 없으면 첫 행만 남긴다.
    """
    n_before = len(df)
    conflicts = find_label_conflicts(df)
    if len(conflicts) > 0:
        conflict_set = set(conflicts["text"].tolist())
        df = df.loc[~df["text"].isin(conflict_set)].copy()
        logger.warning(
            "dedup: 라벨 충돌 text %d개 관련 행을 모두 제거했습니다.",
            len(conflict_set),
        )
    out = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    logger.info(
        "중복 제거(--dedup): %d → %d (제거 %d)",
        n_before,
        len(out),
        n_before - len(out),
    )
    return out


def check_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fail_on_leak: bool = False,
) -> Dict[str, int]:
    """split 간 text 중복 및 group_id 교집합을 검사하고 경고한다."""
    stats: Dict[str, int] = {}

    def _text_overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
        return int(len(set(a["text"]) & set(b["text"])))

    pairs = [
        ("train_val", train_df, val_df),
        ("train_test", train_df, test_df),
        ("val_test", val_df, test_df),
    ]
    leak_found = False
    for name, a, b in pairs:
        n_text = _text_overlap(a, b)
        stats[f"text_overlap_{name}"] = n_text
        if n_text > 0:
            leak_found = True
            logger.warning(
                "누수 경고: %s 간 동일 text %d건이 겹칩니다. "
                "--dedup 또는 group 분할을 검토하세요.",
                name,
                n_text,
            )

        if a["group_id"].notna().any() and b["group_id"].notna().any():
            ga = set(a.loc[a["group_id"].notna(), "group_id"].astype(str))
            gb = set(b.loc[b["group_id"].notna(), "group_id"].astype(str))
            n_group = len(ga & gb)
        else:
            n_group = 0
        stats[f"group_overlap_{name}"] = n_group
        if n_group > 0:
            leak_found = True
            logger.warning(
                "누수 경고: %s 간 동일 group_id %d개가 겹칩니다.",
                name,
                n_group,
            )

    if not leak_found:
        logger.info("누수 검사: text/group_id 교집합 없음 (양호)")
    elif fail_on_leak:
        raise ValueError(
            "누수가 감지되었습니다 (--fail_on_leak). "
            f"상세: {stats}\n"
            "group 분할·--dedup을 검토하거나 --fail_on_leak 를 끄세요."
        )
    return stats


def _log_split_summary(name: str, df: pd.DataFrame) -> None:
    """개별 split의 크기·라벨·버킷 분포를 출력한다."""
    n = len(df)
    if n == 0:
        logger.warning("[%s] 비어 있습니다.", name)
        return
    label_ratio = df["label"].value_counts(normalize=True).to_dict()
    buckets = df["length_bucket"].value_counts().to_dict()
    logger.info(
        "[%s] n=%d | label비율=%s | 버킷=%s",
        name,
        n,
        {int(k): round(float(v), 3) for k, v in label_ratio.items()},
        buckets,
    )


def _pos_ratio(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return float("nan")
    return float(df["label"].mean())


def _warn_class_ratio_gap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_ratio: float,
) -> None:
    """분할 후 클래스 비율 편차가 크면 경고한다.

    GroupShuffleSplit은 stratify를 보장하지 않으므로 반드시 확인한다.
    StratifiedGroupKFold도 완전 보장은 아니므로 동일하게 점검한다.
    """
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        r = _pos_ratio(part)
        if r != r:
            continue
        gap = abs(r - full_ratio)
        logger.info(
            "클래스 비율 점검 [%s]: pos=%.3f (전체 %.3f, 편차 %.1f%%p)",
            name,
            r,
            full_ratio,
            100.0 * gap,
        )
        if gap > CLASS_RATIO_GAP_WARN:
            logger.warning(
                "[%s] 클래스 비율 편차가 %.1f%%p > %.0f%%p 입니다. "
                "GroupShuffleSplit은 stratify를 보장하지 않습니다. "
                "seed를 바꾸거나 그룹 균형을 점검하세요.",
                name,
                100.0 * gap,
                100.0 * CLASS_RATIO_GAP_WARN,
            )


def _stratified_two_way(
    df: pd.DataFrame,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """라벨 stratified 로 train/holdout 이분할."""
    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError("비율은 0과 1 사이여야 합니다.")
    min_class = int(df["label"].value_counts().min())
    if min_class < 2:
        logger.warning("클래스당 표본이 부족하여 stratify 없이 분할합니다.")
        return train_test_split(df, test_size=test_ratio, random_state=seed)
    return train_test_split(
        df, test_size=test_ratio, random_state=seed, stratify=df["label"]
    )


def split_random(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """라벨 기준 stratified random 분할.

    클래스 비율은 거의 같지만 같은 작성자/원문의 글이 train과 test에 섞일 수 있다.
    """
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio({val_ratio})+test_ratio({test_ratio}) 합이 1 이상입니다. "
            f"합이 1 미만이 되도록 줄이세요."
        )
    train_df, temp_df = _stratified_two_way(df, val_ratio + test_ratio, seed)
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = _stratified_two_way(temp_df, relative_test, seed + 1)
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _try_stratified_group_split(
    df: pd.DataFrame,
    test_ratio: float,
    seed: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """StratifiedGroupKFold로 holdout 인덱스를 얻는다. 실패 시 None.

    StratifiedGroupKFold: 그룹이 한 fold에만 들어가면서 클래스 비율을
    최대한 보존한다.
    GroupShuffleSplit: 그룹 단위 랜덤 분할만 하며 stratify를 보장하지 않는다.
    """
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError:
        return None

    n_splits = max(2, int(round(1.0 / test_ratio)))
    n_splits = min(n_splits, len(df["group_id"].astype(str).unique()))
    if n_splits < 2:
        return None
    try:
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        # 첫 번째 fold의 test를 holdout으로 사용
        for train_idx, hold_idx in sgkf.split(
            df, df["label"].astype(int), df["group_id"].astype(str)
        ):
            actual = len(hold_idx) / len(df)
            # 목표 비율과 너무 다르면 다음 fold
            if abs(actual - test_ratio) > 0.15:
                continue
            return train_idx, hold_idx
        # 가장 가까운 fold
        best = None
        best_gap = 1e9
        for train_idx, hold_idx in sgkf.split(
            df, df["label"].astype(int), df["group_id"].astype(str)
        ):
            gap = abs(len(hold_idx) / len(df) - test_ratio)
            if gap < best_gap:
                best_gap = gap
                best = (train_idx, hold_idx)
        return best
    except Exception as exc:
        logger.warning("StratifiedGroupKFold 실패 → GroupShuffleSplit 폴백: %s", exc)
        return None


def split_group(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """group_id 기준 분할. 같은 group_id는 한 split에만 존재한다.

    Returns:
        train, val, test, method_name ('stratified_group_kfold'|'group_shuffle_split')
    """
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio({val_ratio})+test_ratio({test_ratio}) 합이 1 이상입니다."
        )

    # 대형 그룹 경고
    vc = df["group_id"].astype(str).value_counts(normalize=True)
    large = vc[vc > LARGE_GROUP_RATIO]
    if len(large) > 0:
        logger.warning(
            "일부 group_id가 전체의 %.0f%%를 초과합니다: %s. "
            "분할 비율이 왜곡될 수 있습니다.",
            100.0 * LARGE_GROUP_RATIO,
            {k: round(float(v), 3) for k, v in large.head(5).items()},
        )

    holdout_ratio = val_ratio + test_ratio
    method = "stratified_group_kfold"
    result = _try_stratified_group_split(df, holdout_ratio, seed)
    if result is None:
        method = "group_shuffle_split"
        logger.info(
            "GroupShuffleSplit 사용: stratify 미보장. "
            "분할 후 클래스 비율 편차를 반드시 확인하세요."
        )
        groups = df["group_id"].astype(str)
        labels = df["label"].astype(int)
        best_train_idx, best_hold_idx = None, None
        best_score = -1.0
        for attempt in range(20):
            gss = GroupShuffleSplit(
                n_splits=1, test_size=holdout_ratio, random_state=seed + attempt
            )
            train_idx, hold_idx = next(gss.split(df, labels, groups))
            score = 1.0 - abs(labels.iloc[hold_idx].mean() - labels.mean())
            if score > best_score:
                best_score = score
                best_train_idx, best_hold_idx = train_idx, hold_idx
        assert best_train_idx is not None and best_hold_idx is not None
        train_idx, hold_idx = best_train_idx, best_hold_idx
    else:
        train_idx, hold_idx = result
        logger.info("StratifiedGroupKFold 기반 분할 사용 (클래스 비율 보존 우선).")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    hold_df = df.iloc[hold_idx].reset_index(drop=True)

    relative_test = test_ratio / (val_ratio + test_ratio)
    # holdout → val/test 도 group 단위
    sub = _try_stratified_group_split(hold_df, relative_test, seed + 100)
    if sub is None:
        hold_groups = hold_df["group_id"].astype(str)
        hold_labels = hold_df["label"].astype(int)
        best_val_idx, best_test_idx = None, None
        best_score = -1.0
        for attempt in range(20):
            gss2 = GroupShuffleSplit(
                n_splits=1, test_size=relative_test, random_state=seed + 100 + attempt
            )
            val_idx, test_idx = next(gss2.split(hold_df, hold_labels, hold_groups))
            score = 1.0 - abs(
                hold_labels.iloc[test_idx].mean() - df["label"].mean()
            )
            if score > best_score:
                best_score = score
                best_val_idx, best_test_idx = val_idx, test_idx
        assert best_val_idx is not None and best_test_idx is not None
        val_df = hold_df.iloc[best_val_idx].reset_index(drop=True)
        test_df = hold_df.iloc[best_test_idx].reset_index(drop=True)
        if method == "stratified_group_kfold":
            method = "stratified_group_kfold+group_shuffle_split"
    else:
        val_idx, test_idx = sub
        val_df = hold_df.iloc[val_idx].reset_index(drop=True)
        test_df = hold_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df, method


def split_source(
    df: pd.DataFrame,
    holdout_source: str,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """특정 source_type을 test에만 두는 OOD 분할."""
    if "source_type" not in df.columns or df["source_type"].isna().all():
        raise ValueError(
            "split_strategy=source 를 쓰려면 source_type 컬럼이 필요합니다.\n"
            "데이터에 source_type을 추가하거나 다른 전략을 사용하세요."
        )
    if holdout_source is None or holdout_source == "":
        raise ValueError(
            "split_strategy=source 일 때 --holdout_source 를 지정해야 합니다. "
            "예: --holdout_source comment"
        )

    available = sorted(df["source_type"].dropna().astype(str).unique())
    mask = df["source_type"].astype(str) == str(holdout_source)
    test_df = df.loc[mask].reset_index(drop=True)
    rest = df.loc[~mask].reset_index(drop=True)
    if len(test_df) == 0:
        raise ValueError(
            f"holdout_source='{holdout_source}' 에 해당하는 행이 없습니다.\n"
            f"존재하는 source_type: {available}"
        )
    if len(rest) == 0:
        raise ValueError(
            f"holdout_source='{holdout_source}' 외 데이터가 없어 train/val을 만들 수 없습니다."
        )

    train_df, val_df = _stratified_two_way(rest, val_ratio, seed)
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _prepare_groups_for_split(
    df: pd.DataFrame,
    strict_split: bool,
) -> Tuple[pd.DataFrame, bool]:
    """group 전략용 group_id 정리. 폴백 필요 시 fallback=True.

    - 컬럼 없음/전부 결측: strict면 에러, 아니면 fallback.
    - 일부 결측: 고유 group으로 채우고 경고.
    """
    work = df.copy()
    if "group_id" not in work.columns or work["group_id"].isna().all():
        if strict_split:
            raise ValueError(
                "split_strategy=group 인데 group_id가 없거나 전부 결측입니다.\n"
                "group_id를 추가하거나 --strict_split 없이 실행하면 "
                "random stratified로 폴백합니다."
            )
        logger.warning(
            "group_id 없음/전부 결측 → random stratified로 폴백합니다. "
            "(--strict_split 로 에러화 가능)"
        )
        return work, True

    na_mask = work["group_id"].isna()
    if na_mask.any():
        n_na = int(na_mask.sum())
        logger.warning(
            "group_id 결측 %d행 → 각 행을 고유 group으로 취급합니다.", n_na
        )
        # 고유 임시 ID
        fillers = [f"__missing_group_{i}" for i in range(n_na)]
        work.loc[na_mask, "group_id"] = fillers
    return work, False


def _split_stats(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    if n == 0:
        return {"n": 0}
    lc = df["label"].value_counts(normalize=True).to_dict()
    out: Dict[str, Any] = {
        "n": n,
        "label_ratio": {str(int(k)): float(v) for k, v in lc.items()},
        "length_bucket": {
            str(k): int(v) for k, v in df["length_bucket"].value_counts().items()
        },
    }
    if "source_type" in df.columns and df["source_type"].notna().any():
        out["source_type"] = {
            str(k): int(v)
            for k, v in df["source_type"].astype(str).value_counts().items()
        }
    return out


def build_split_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy: str,
    seed: int,
    leakage: Dict[str, int],
    fallback: bool = False,
    method: Optional[str] = None,
) -> Dict[str, Any]:
    """split_report.json 내용."""
    return {
        "strategy": strategy,
        "method": method,
        "seed": seed,
        "fallback": fallback,
        "splits": {
            "train": _split_stats(train_df),
            "val": _split_stats(val_df),
            "test": _split_stats(test_df),
        },
        "leakage": leakage,
    }


def split_report_to_markdown(report: Dict[str, Any]) -> str:
    lines = ["# Split Report", ""]
    lines.append(f"- strategy: `{report.get('strategy')}`")
    lines.append(f"- method: `{report.get('method')}`")
    lines.append(f"- seed: {report.get('seed')}")
    lines.append(f"- fallback: {report.get('fallback')}")
    lines.append("")
    for name, st in report.get("splits", {}).items():
        lines.append(f"## {name}")
        lines.append(f"- n: {st.get('n')}")
        lines.append(f"- label_ratio: `{st.get('label_ratio')}`")
        lines.append(f"- length_bucket: `{st.get('length_bucket')}`")
        lines.append("")
    lines.append("## Leakage")
    lines.append(f"```\n{report.get('leakage')}\n```")
    lines.append("")
    return "\n".join(lines)


def split_dataframe(
    df: pd.DataFrame,
    strategy: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    holdout_source: Optional[str] = None,
    dedup: bool = False,
    splits_dir: Optional[Union[str, Path]] = None,
    report_dir: Optional[Union[str, Path]] = None,
    strict_split: bool = False,
    fail_on_leak: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """전략에 따라 train/val/test를 분할하고 누수 검사·리포트를 저장한다."""
    work = deduplicate_texts(df) if dedup else df.copy()
    fallback = False
    method: Optional[str] = None
    used_strategy = strategy

    if strategy == "random":
        train_df, val_df, test_df = split_random(work, val_ratio, test_ratio, seed)
        method = "stratified_shuffle"
    elif strategy == "group":
        work, need_fallback = _prepare_groups_for_split(work, strict_split)
        if need_fallback:
            fallback = True
            used_strategy = "random"
            train_df, val_df, test_df = split_random(work, val_ratio, test_ratio, seed)
            method = "stratified_shuffle_fallback"
        else:
            train_df, val_df, test_df, method = split_group(
                work, val_ratio, test_ratio, seed
            )
    elif strategy == "source":
        train_df, val_df, test_df = split_source(
            work, holdout_source or "", val_ratio, seed
        )
        method = "holdout_source"
    else:
        raise ValueError(
            f"알 수 없는 split_strategy: '{strategy}'. "
            f"random | group | source 중 하나를 선택하세요."
        )

    leakage = check_leakage(train_df, val_df, test_df, fail_on_leak=fail_on_leak)
    _log_split_summary("train", train_df)
    _log_split_summary("val", val_df)
    _log_split_summary("test", test_df)
    _warn_class_ratio_gap(train_df, val_df, test_df, float(work["label"].mean()))

    split_report = build_split_report(
        train_df,
        val_df,
        test_df,
        strategy=used_strategy,
        seed=seed,
        leakage=leakage,
        fallback=fallback,
        method=method,
    )

    if splits_dir is not None:
        out = Path(splits_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
            cols = [
                c
                for c in [
                    "text",
                    "label",
                    "source_type",
                    "group_id",
                    "length",
                    "length_bucket",
                ]
                if c in part.columns
            ]
            part[cols].to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
        logger.info("분할 인덱스 저장: %s", out)

    if report_dir is not None:
        rd = Path(report_dir)
        rd.mkdir(parents=True, exist_ok=True)
        save_json(split_report, rd / "split_report.json")
        (rd / "split_report.md").write_text(
            split_report_to_markdown(split_report), encoding="utf-8"
        )
        logger.info("split_report 저장: %s", rd)

    return train_df, val_df, test_df, split_report


class SNSTextDataset(Dataset):
    """토크나이즈된 SNS 텍스트 이진 분류 Dataset."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[int]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: int = 128,
        encodings: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.max_length = max_length

        if encodings is not None:
            self.encodings = encodings
        else:
            if tokenizer is None:
                raise ValueError("tokenizer 또는 encodings 중 하나가 필요합니다.")
            self.encodings = tokenizer(
                self.texts,
                truncation=True,
                max_length=max_length,
                padding=False,
            )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            key: self.encodings[key][idx] for key in self.encodings
        }
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def _cache_key(
    data_path: Union[str, Path],
    model_name: str,
    max_length: int,
    collapse_repeats: bool,
    max_repeat: Optional[int],
    newline_token: bool,
    split_name: str,
    seed: int,
    strategy: str,
) -> str:
    raw = "|".join(
        [
            file_content_hash(data_path) if Path(data_path).exists() else str(data_path),
            model_name,
            str(max_length),
            str(collapse_repeats),
            str(max_repeat),
            str(newline_token),
            split_name,
            str(seed),
            strategy,
        ]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def tokenize_dataframe(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    num_proc: int = 1,
    cache_dir: Optional[Union[str, Path]] = None,
    cache_key: Optional[str] = None,
    no_cache: bool = False,
) -> SNSTextDataset:
    """DataFrame을 토크나이즈해 SNSTextDataset을 만든다.

    datasets.Dataset.map(batched=True) 사용.
    Windows/MPS에서는 num_proc>1 이 불안정할 수 있으므로 기본 1.
    (fork 미지원·MPS 멀티프로세스 이슈)
    """
    from datasets import Dataset as HFDataset

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    cache_path = None
    if cache_dir is not None and cache_key is not None and not no_cache:
        cache_path = Path(cache_dir) / f"{cache_key}.json"
        if cache_path.exists():
            logger.info("토큰화 캐시 재사용: %s", cache_path)
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return SNSTextDataset(
                texts=payload["texts"],
                labels=payload["labels"],
                encodings=payload["encodings"],
                max_length=max_length,
            )

    # 메모리: text/label만 유지
    hf_ds = HFDataset.from_dict({"text": texts, "label": labels})

    def _tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    # Windows에서 num_proc>1 은 spawn 오버헤드·데드락 위험이 있어 기본 1 권장
    map_kwargs: Dict[str, Any] = {
        "batched": True,
        "remove_columns": ["text"],
    }
    if num_proc and num_proc > 1 and sys.platform != "win32":
        map_kwargs["num_proc"] = num_proc
    elif num_proc > 1 and sys.platform == "win32":
        logger.warning(
            "Windows에서는 tokenize_num_proc>1 을 1로 강제합니다 "
            "(멀티프로세싱 이슈)."
        )

    tokenized = hf_ds.map(_tokenize_batch, **map_kwargs)

    # truncation 비율 / 평균 토큰 길이
    lengths = [len(x) for x in tokenized["input_ids"]]
    trunc_ratio = float(np.mean([L >= max_length for L in lengths])) if lengths else 0.0
    logger.info(
        "토큰 길이: mean=%.1f median=%.1f | truncation 비율(>=max_length)=%.1f%%",
        float(np.mean(lengths)) if lengths else 0.0,
        float(np.median(lengths)) if lengths else 0.0,
        100.0 * trunc_ratio,
    )

    encodings = {
        k: list(tokenized[k])
        for k in tokenized.column_names
        if k not in {"label", "labels"}
    }
    ds = SNSTextDataset(
        texts=texts,
        labels=labels,
        encodings=encodings,
        max_length=max_length,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"texts": texts, "labels": labels, "encodings": encodings},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info("토큰화 캐시 저장: %s", cache_path)

    return ds


def prepare_datasets(
    data_path: Union[str, Path],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    split_strategy: str = "random",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    holdout_source: Optional[str] = None,
    dedup: bool = False,
    collapse_repeats: bool = False,
    max_repeat: Optional[int] = None,
    length_bins: Sequence[int] = (20, 100),
    splits_dir: Optional[Union[str, Path]] = None,
    report_dir: Optional[Union[str, Path]] = None,
    max_rows: Optional[int] = None,
    skip_bad_lines: bool = False,
    strict_split: bool = False,
    fail_on_leak: bool = False,
    newline_token: bool = False,
    tokenize_num_proc: int = 1,
    cache_dir: Optional[Union[str, Path]] = None,
    no_cache: bool = False,
    model_name: str = "",
) -> Tuple[
    SNSTextDataset,
    SNSTextDataset,
    SNSTextDataset,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
    Dict[str, Any],
]:
    """로딩부터 토크나이징까지 한 번에 수행한다.

    Returns:
        train_ds, val_ds, test_ds, train_df, val_df, test_df, data_report, split_report
    """
    raw = load_raw_dataframe(
        data_path, max_rows=max_rows, skip_bad_lines=skip_bad_lines
    )
    cleaned, data_report = validate_and_clean_dataframe(
        raw,
        collapse_repeats=collapse_repeats,
        max_repeat=max_repeat,
        length_bins=length_bins,
        newline_token=newline_token,
        report_dir=report_dir,
        require_both_classes=True,
    )
    train_df, val_df, test_df, split_report = split_dataframe(
        cleaned,
        strategy=split_strategy,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        holdout_source=holdout_source,
        dedup=dedup,
        splits_dir=splits_dir,
        report_dir=report_dir,
        strict_split=strict_split,
        fail_on_leak=fail_on_leak,
    )

    cdir = cache_dir
    train_ds = tokenize_dataframe(
        train_df,
        tokenizer,
        max_length=max_length,
        num_proc=tokenize_num_proc,
        cache_dir=cdir,
        cache_key=_cache_key(
            data_path,
            model_name or "model",
            max_length,
            collapse_repeats,
            max_repeat,
            newline_token,
            "train",
            seed,
            split_strategy,
        ),
        no_cache=no_cache,
    )
    val_ds = tokenize_dataframe(
        val_df,
        tokenizer,
        max_length=max_length,
        num_proc=tokenize_num_proc,
        cache_dir=cdir,
        cache_key=_cache_key(
            data_path,
            model_name or "model",
            max_length,
            collapse_repeats,
            max_repeat,
            newline_token,
            "val",
            seed,
            split_strategy,
        ),
        no_cache=no_cache,
    )
    test_ds = tokenize_dataframe(
        test_df,
        tokenizer,
        max_length=max_length,
        num_proc=tokenize_num_proc,
        cache_dir=cdir,
        cache_key=_cache_key(
            data_path,
            model_name or "model",
            max_length,
            collapse_repeats,
            max_repeat,
            newline_token,
            "test",
            seed,
            split_strategy,
        ),
        no_cache=no_cache,
    )
    return (
        train_ds,
        val_ds,
        test_ds,
        train_df,
        val_df,
        test_df,
        data_report,
        split_report,
    )
