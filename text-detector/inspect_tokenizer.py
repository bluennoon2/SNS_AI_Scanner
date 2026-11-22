# 이 파일의 역할:
# SNS 도메인 텍스트에 대한 토크나이저 점검.
# [UNK]/줄바꿈/이모지/자모 반복 손실을 보고하고 권장 max_length를 제안한다.
# vocab을 임의로 수정하지 않는다. --newline_token 은 별도 전처리 옵션이다.

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from transformers import AutoTokenizer

from dataset import load_raw_dataframe, validate_and_clean_dataframe
from preprocess import NEWLINE_PLACEHOLDER, clean_text
import logging

from utils import ensure_dir, save_json, setup_logging

logger = logging.getLogger("sns_ai_detector")

PATTERN_SAMPLES = [
    ("emoji", "오늘 날씨 미쳤다 😂🔥 #일상"),
    ("hashtag", "주말 카페 #카페스타그램 #일상공유"),
    ("newline", "첫 줄입니다.\n둘째 줄입니다.\n셋째 줄"),
    ("jamo_k", "ㅋㅋㅋㅋㅋ 진짜 웃기다"),
    ("jamo_h", "ㅎㅎㅎㅎ 그래그래"),
    ("jamo_y", "ㅠㅠㅠ 너무 슬퍼"),
    ("url", "이거 봐봐 https://example.com/path?x=1"),
    ("mention", "야 @friend 이거 봤어?"),
    ("exclaim", "대박!!!!!! 진짜...?"),
    ("ellipsis", "그래서...... 결국 이렇게 됨"),
]


def tokenize_detail(tokenizer, text: str) -> Dict[str, Any]:
    ids = tokenizer.encode(text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    unk_id = tokenizer.unk_token_id
    n_unk = sum(1 for i in ids if i == unk_id) if unk_id is not None else 0
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    return {
        "text": text,
        "tokens": tokens,
        "n_tokens": len(ids),
        "n_unk": n_unk,
        "has_unk": n_unk > 0,
        "decoded": decoded,
        "decode_differs": decoded.replace(" ", "") != text.replace(" ", ""),
    }


def analyze_newline_preservation(tokenizer, text: str = "A\nB\nC") -> Dict[str, Any]:
    detail = tokenize_detail(tokenizer, text)
    joined = " ".join(detail["tokens"])
    preserved = ("\n" in joined) or ("Ċ" in joined) or ("▁" in text)  # 일부 BPE
    # KcELECTRA는 WordPiece — 줄바꿈이 사라지는 경우가 많음
    newline_in_tokens = any("\n" in t or t in {"Ċ", "<0x0A>"} for t in detail["tokens"])
    return {
        "input": text,
        "tokens": detail["tokens"],
        "newline_token_present": newline_in_tokens,
        "note": (
            "줄바꿈이 토큰으로 보존되지 않으면 문단 구조 신호가 손실될 수 있습니다. "
            "옵션 --newline_token 으로 [NL] 치환을 검토하세요 (vocab 수정 아님)."
        ),
        "suggestion": (
            f"전처리에서 \\n → {NEWLINE_PLACEHOLDER} "
            "(train/evaluate/predict 동일 적용, inference_meta에 기록)"
        ),
    }


def recommend_max_length(token_lengths: Sequence[int]) -> Dict[str, Any]:
    import numpy as np

    arr = np.asarray(token_lengths, dtype=float)
    if len(arr) == 0:
        return {"recommended": 128, "reason": "샘플 없음"}
    stats = {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }
    trunc = {}
    for ml in (64, 128, 256):
        trunc[str(ml)] = float((arr > ml).mean())
    # p95 기준으로 여유 있게
    p95 = stats["p95"]
    if p95 <= 64:
        rec = 64
    elif p95 <= 128:
        rec = 128
    else:
        rec = 256
    return {
        "stats": stats,
        "truncation_ratio_by_max_length": trunc,
        "recommended": rec,
        "reason": f"토큰 길이 p95≈{p95:.0f} → max_length={rec} 권장",
    }


def run_inspect(
    data_path: str,
    model_name: str = "beomi/KcELECTRA-base",
    max_length: int = 128,
    n_samples: int = 200,
    output_dir: str = "outputs/tokenizer_report",
    seed: int = 42,
    newline_token: bool = False,
) -> Dict[str, Any]:
    logger.info("토크나이저 로드: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pattern_results = []
    for name, sample in PATTERN_SAMPLES:
        cleaned = clean_text(sample, newline_token=newline_token)
        d = tokenize_detail(tokenizer, cleaned)
        d["pattern"] = name
        pattern_results.append(d)
        logger.info(
            "[%s] unk=%s n_tokens=%d tokens=%s",
            name,
            d["has_unk"],
            d["n_tokens"],
            d["tokens"][:20],
        )

    raw = load_raw_dataframe(data_path)
    cleaned, _ = validate_and_clean_dataframe(
        raw, newline_token=newline_token, require_both_classes=False
    )
    texts = cleaned["text"].tolist()
    rng = random.Random(seed)
    sample_texts = texts if len(texts) <= n_samples else rng.sample(texts, n_samples)

    sample_details = [tokenize_detail(tokenizer, t) for t in sample_texts]
    total_tokens = sum(d["n_tokens"] for d in sample_details) or 1
    total_unk = sum(d["n_unk"] for d in sample_details)
    unk_ratio = total_unk / total_tokens

    # 이모지/특수문자 한정 UNK (대략)
    emoji_re = re.compile("[\U0001F300-\U0001F9FF]")
    emoji_unk = 0
    emoji_tok = 0
    for d in sample_details:
        if emoji_re.search(d["text"]):
            emoji_tok += d["n_tokens"]
            emoji_unk += d["n_unk"]
    emoji_unk_ratio = (emoji_unk / emoji_tok) if emoji_tok else 0.0

    nl_info = analyze_newline_preservation(tokenizer)
    lengths = [d["n_tokens"] for d in sample_details]
    rec = recommend_max_length(lengths)

    # UNK/과분절 상위
    worst = sorted(sample_details, key=lambda d: (-d["n_unk"], -d["n_tokens"]))[:20]
    decode_diffs = [d for d in sample_details if d["decode_differs"]][:20]

    report = {
        "model_name": model_name,
        "n_samples": len(sample_details),
        "unk_ratio_all_tokens": unk_ratio,
        "unk_ratio_emoji_subset": emoji_unk_ratio,
        "newline": nl_info,
        "max_length_recommendation": rec,
        "pattern_samples": pattern_results,
        "worst_unk_samples": [
            {"text": d["text"][:200], "n_unk": d["n_unk"], "tokens": d["tokens"][:40]}
            for d in worst
            if d["n_unk"] > 0
        ],
        "decode_diff_samples": [
            {"text": d["text"][:200], "decoded": d["decoded"][:200]}
            for d in decode_diffs
        ],
        "disclaimer": (
            "정보 손실이 커 보여도 토크나이저 vocab을 임의로 수정하지 마세요. "
            "줄바꿈 보존이 필요하면 --newline_token 전처리 옵션을 검토하세요."
        ),
        "queried_max_length": max_length,
    }

    out = ensure_dir(output_dir)
    # 단일 파일명으로도 저장 + 디렉터리 표준
    save_json(report, out / "tokenizer_report.json")
    # 요청 경로 호환: outputs/tokenizer_report.json 도 가능하도록 parent에도?
    md = [
        "# Tokenizer Report",
        "",
        f"> {report['disclaimer']}",
        "",
        f"- model: `{model_name}`",
        f"- samples: {len(sample_details)}",
        f"- UNK ratio: {unk_ratio:.4%}",
        f"- emoji-subset UNK ratio: {emoji_unk_ratio:.4%}",
        f"- newline preserved: {nl_info['newline_token_present']}",
        f"- recommended max_length: **{rec['recommended']}** ({rec['reason']})",
        "",
        "## Truncation by max_length",
        f"```\n{rec.get('truncation_ratio_by_max_length')}\n```",
        "",
        "## Pattern samples",
    ]
    for d in pattern_results:
        md.append(
            f"- **{d['pattern']}**: unk={d['has_unk']} tokens=`{d['tokens'][:15]}`"
        )
    md += ["", "## Newline note", "", nl_info["note"], "", nl_info["suggestion"], ""]
    (out / "tokenizer_report.md").write_text("\n".join(md), encoding="utf-8")
    # 편의: outputs/tokenizer_report.json 심볼릭 대신 복사성 저장
    save_json(report, Path(output_dir).parent / "tokenizer_report.json")
    (Path(output_dir).parent / "tokenizer_report.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    logger.info("토크나이저 리포트 저장: %s", out)
    logger.info("권장 max_length=%s", rec["recommended"])
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KcELECTRA 토크나이저 SNS 도메인 점검")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--model_name", type=str, default="beomi/KcELECTRA-base")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--output_dir", type=str, default="outputs/tokenizer_inspect")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--newline_token",
        action="store_true",
        default=False,
        help="점검 시 줄바꿈을 [NL]로 치환한 결과도 확인 (기본 OFF)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    setup_logging()
    args = build_parser().parse_args(argv)
    run_inspect(
        data_path=args.data_path,
        model_name=args.model_name,
        max_length=args.max_length,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        newline_token=args.newline_token,
    )


if __name__ == "__main__":
    main()
