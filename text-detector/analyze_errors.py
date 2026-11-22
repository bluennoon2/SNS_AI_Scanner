# 이 파일의 역할:
# 예측 결과에서 오탐(FP)/미탐(FN) 상위 샘플을 추출하고
# 규칙 기반 휴리스틱 태그로 오류 집중도를 요약한다.
# 태그는 가설 생성용이며 원인 증명이 아니다.

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

import logging

from utils import load_json, save_json, setup_logging

logger = logging.getLogger("sns_ai_detector")

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U00002700-\U000027BF"
    "\U0001F600-\U0001F64F"
    "]+",
    flags=re.UNICODE,
)
_HASHTAG_RE = re.compile(r"#[\w가-힣]+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@[\w가-힣.]+")
_JAMO_RE = re.compile(r"[ㅋㅎㅠㅜ]{2,}")
_FORMAL_RE = re.compile(r"(습니다|입니다|하세요|하십시오|됩니다)\s*[.!?]?$")
_EXCLAIM_RE = re.compile(r"^[와우헐대박오ㄷㄱㅋㅎ!\s.?]+$", re.IGNORECASE)


def mask_pii(text: str) -> str:
    """URL/멘션을 마스킹한다."""
    text = _URL_RE.sub("[URL]", text)
    text = _MENTION_RE.sub("[MENTION]", text)
    return text


def heuristic_tags(text: str) -> List[str]:
    """규칙 기반 휴리스틱 태그 목록을 반환한다."""
    tags: List[str] = []
    if text is None:
        return ["empty"]
    t = str(text)
    n = len(t)
    if n < 20:
        tags.append("very_short")
    emoji_chars = _EMOJI_RE.findall(t)
    emoji_len = sum(len(e) for e in emoji_chars)
    if n > 0 and emoji_len / n >= 0.15:
        tags.append("emoji_heavy")
    if len(_HASHTAG_RE.findall(t)) >= 2:
        tags.append("hashtag_heavy")
    if _URL_RE.search(t):
        tags.append("has_url")
    if _MENTION_RE.search(t):
        tags.append("has_mention")
    if _JAMO_RE.search(t):
        tags.append("jamo_repeat")
    if "\n" in t:
        tags.append("multiline")
    if _FORMAL_RE.search(t.strip()):
        tags.append("formal_ending")
    # 감탄/자모 위주 짧은 반응
    compact = re.sub(r"\s+", "", t)
    if n <= 15 and (_EXCLAIM_RE.match(t) or re.fullmatch(r"[ㅋㅎㅠㅜ!?~.]+", compact)):
        tags.append("low_info_exclaim")
    return tags


def _ensure_pred_cols(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    work = df.copy()
    if "prob_ai" not in work.columns:
        raise ValueError("predictions에 prob_ai 컬럼이 필요합니다.")
    if "label" not in work.columns:
        raise ValueError("predictions에 label 컬럼이 필요합니다.")
    if "pred" not in work.columns and "pred_label" not in work.columns:
        work["pred"] = (work["prob_ai"] >= threshold).astype(int)
    elif "pred" not in work.columns:
        work["pred"] = work["pred_label"]
    return work


def analyze_errors(
    predictions: Union[str, Path, pd.DataFrame],
    threshold: Optional[float] = None,
    threshold_json: Optional[Union[str, Path]] = None,
    top_k: int = 50,
    output_dir: Union[str, Path] = "outputs/error_analysis",
    mask_pii_flag: bool = False,
) -> Dict[str, Any]:
    """FP/FN 상위 샘플과 태그별 집중도를 분석한다."""
    if threshold is None and threshold_json is not None:
        payload = load_json(threshold_json)
        threshold = float(payload.get("threshold", 0.5))
    if threshold is None:
        threshold = 0.5

    if isinstance(predictions, (str, Path)):
        path = Path(predictions)
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        df = predictions

    work = _ensure_pred_cols(df, float(threshold))
    # pred 재계산 (threshold 명시 시)
    work["pred"] = (work["prob_ai"] >= float(threshold)).astype(int)
    work["tags"] = work["text"].astype(str).apply(heuristic_tags)

    fp = work[(work["label"] == 0) & (work["pred"] == 1)].sort_values(
        "prob_ai", ascending=False
    )
    fn = work[(work["label"] == 1) & (work["pred"] == 0)].sort_values(
        "prob_ai", ascending=True
    )

    fp_top = fp.head(top_k).copy()
    fn_top = fn.head(top_k).copy()
    if mask_pii_flag:
        fp_top["text"] = fp_top["text"].astype(str).apply(mask_pii)
        fn_top["text"] = fn_top["text"].astype(str).apply(mask_pii)

    # 태그별 FP/FN율 vs 전체
    def _tag_rates(subset: pd.DataFrame, name: str) -> Dict[str, Any]:
        rates: Dict[str, Any] = {}
        all_n = len(work)
        sub_n = len(subset)
        overall_rate = sub_n / all_n if all_n else 0.0
        # explode tags
        tag_rows: List[str] = []
        for tags in work["tags"].tolist():
            tag_rows.extend(tags)
        from collections import Counter

        tag_counts = Counter(tag_rows)
        sub_tag_rows: List[str] = []
        for tags in subset["tags"].tolist():
            sub_tag_rows.extend(tags)
        sub_counts = Counter(sub_tag_rows)

        for tag, cnt_all in tag_counts.items():
            # 해당 태그를 가진 전체 샘플 수
            n_tag = int((work["tags"].apply(lambda ts: tag in ts)).sum())
            n_tag_err = int((subset["tags"].apply(lambda ts: tag in ts)).sum())
            rate = n_tag_err / n_tag if n_tag else 0.0
            rates[tag] = {
                "n_with_tag": n_tag,
                f"n_{name}": n_tag_err,
                f"{name}_rate": rate,
                "overall_rate": overall_rate,
                "lift": (rate / overall_rate) if overall_rate > 0 else float("nan"),
            }
        return rates

    fp_tag = _tag_rates(fp, "fp")
    fn_tag = _tag_rates(fn, "fn")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    err_dir = out / "error_analysis"
    err_dir.mkdir(parents=True, exist_ok=True)

    fp_top.to_csv(err_dir / "fp_top.csv", index=False, encoding="utf-8-sig")
    fn_top.to_csv(err_dir / "fn_top.csv", index=False, encoding="utf-8-sig")

    summary = {
        "threshold": float(threshold),
        "n_total": int(len(work)),
        "n_fp": int(len(fp)),
        "n_fn": int(len(fn)),
        "top_k": top_k,
        "fp_tag_rates": fp_tag,
        "fn_tag_rates": fn_tag,
        "disclaimer": (
            "휴리스틱 태그는 규칙 기반 가설 생성용이며 오류 원인 증명이 아닙니다. "
            "데이터 보강·임계값·모델 개선 방향을 탐색하는 보조 지표로만 사용하세요."
        ),
    }
    save_json(summary, err_dir / "error_summary.json")

    md_lines = [
        "# Error Analysis",
        "",
        f"> {summary['disclaimer']}",
        "",
        f"- threshold: {threshold}",
        f"- FP: {len(fp)} / FN: {len(fn)} / total: {len(work)}",
        "",
        "## FP 태그별 발생률 (vs 전체)",
        "",
        "| Tag | n_with_tag | FP rate | Overall FP rate | Lift |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag, info in sorted(
        fp_tag.items(), key=lambda x: -x[1].get("fp_rate", 0)
    ):
        md_lines.append(
            f"| {tag} | {info['n_with_tag']} | {info['fp_rate']:.3f} | "
            f"{info['overall_rate']:.3f} | {info['lift']:.2f} |"
        )
    md_lines += [
        "",
        "## FN 태그별 발생률 (vs 전체)",
        "",
        "| Tag | n_with_tag | FN rate | Overall FN rate | Lift |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag, info in sorted(
        fn_tag.items(), key=lambda x: -x[1].get("fn_rate", 0)
    ):
        md_lines.append(
            f"| {tag} | {info['n_with_tag']} | {info['fn_rate']:.3f} | "
            f"{info['overall_rate']:.3f} | {info['lift']:.2f} |"
        )
    md_lines += ["", f"## FP Top {top_k}", ""]
    for _, row in fp_top.iterrows():
        md_lines.append(
            f"- prob={row['prob_ai']:.3f} tags={row['tags']}: {row['text'][:120]}"
        )
    md_lines += ["", f"## FN Top {top_k}", ""]
    for _, row in fn_top.iterrows():
        md_lines.append(
            f"- prob={row['prob_ai']:.3f} tags={row['tags']}: {row['text'][:120]}"
        )
    (err_dir / "error_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("오류 분석 저장: %s", err_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="예측 오류 분석 (FP/FN)")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--threshold_json",
        type=str,
        default=None,
        help="threshold.json 경로 (미지정 시 predictions 상위 탐색)",
    )
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mask_pii", action="store_true", default=False)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    setup_logging()
    args = build_parser().parse_args(argv)
    pred_path = Path(args.predictions)
    thr_json = args.threshold_json
    if thr_json is None:
        cand = pred_path.parent / "threshold.json"
        if cand.exists():
            thr_json = str(cand)
    out_dir = args.output_dir or str(pred_path.parent)
    analyze_errors(
        predictions=pred_path,
        threshold=args.threshold,
        threshold_json=thr_json,
        top_k=args.top_k,
        output_dir=out_dir,
        mask_pii_flag=args.mask_pii,
    )


if __name__ == "__main__":
    main()
