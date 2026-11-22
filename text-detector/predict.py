# 이 파일의 역할:
# 학습된 체크포인트로 단일/배치/인터랙티브 추론을 수행한다.
# 로드 우선순위: CLI > inference_meta.json > threshold.json/run_config.json > 기본값.

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import clean_text
from utils import detect_device, load_json, setup_logging

logger = logging.getLogger("sns_ai_detector")

DEFAULT_MIN_TEXT_WARN_LEN = 5
DEFAULT_MIN_CHARS = 20


class Detector:
    """AI 생성 텍스트 탐지 추론기."""

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        threshold: float = 0.5,
        max_length: int = 128,
        min_text_warn_len: int = DEFAULT_MIN_TEXT_WARN_LEN,
        min_chars: int = DEFAULT_MIN_CHARS,
        collapse_repeats: bool = False,
        max_repeat: Optional[int] = None,
        newline_token: bool = False,
        meta_sources: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = float(threshold)
        self.max_length = int(max_length)
        self.min_text_warn_len = int(min_text_warn_len)
        self.min_chars = int(min_chars)
        self.collapse_repeats = collapse_repeats
        self.max_repeat = max_repeat
        self.newline_token = newline_token
        self.meta_sources = meta_sources or {}
        self.model.eval()

    def _clean_one(self, text: Any) -> str:
        if text is None or (isinstance(text, float) and math.isnan(text)):
            return ""
        return clean_text(
            text if isinstance(text, str) else str(text),
            collapse_repeats=self.collapse_repeats,
            max_repeat=self.max_repeat,
            newline_token=self.newline_token,
        )

    def _normalize_texts(self, texts: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(texts, str):
            raw_list: List[str] = [texts]
        else:
            raw_list = list(texts)
        return [self._clean_one(t) for t in raw_list]

    @torch.inference_mode()
    def predict_proba(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        sort_by_length: bool = True,
    ) -> List[Optional[float]]:
        """AI 확률 리스트. 빈 입력은 None(NaN)으로 반환하며 배치를 중단하지 않는다."""
        if isinstance(texts, str):
            original = [texts]
        else:
            original = list(texts)
        cleaned = self._normalize_texts(original)
        n = len(cleaned)
        out: List[Optional[float]] = [None] * n

        # 비어 있지 않은 인덱스만 추론
        valid_idx = [i for i, t in enumerate(cleaned) if t]
        empty_idx = [i for i, t in enumerate(cleaned) if not t]
        for i in empty_idx:
            logger.warning("빈/정제후 빈 텍스트 (index=%d) → prob_ai=NaN", i)

        if not valid_idx:
            return out

        # 길이순 정렬로 패딩 효율↑, 결과는 원래 순서로 복원
        order = list(valid_idx)
        if sort_by_length:
            order = sorted(valid_idx, key=lambda i: len(cleaned[i]))

        iterator = range(0, len(order), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="predict", leave=False)

        for start in iterator:
            batch_order = order[start : start + batch_size]
            feed = [cleaned[i] for i in batch_order]
            encoded = self.tokenizer(
                feed,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            batch_probs = F.softmax(outputs.logits, dim=-1)[:, 1]
            for j, idx in enumerate(batch_order):
                out[idx] = float(batch_probs[j].detach().cpu())

        return out

    def predict(
        self,
        texts: Union[str, Sequence[str]],
        threshold: Optional[float] = None,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[Dict[str, Any]]:
        """{text, prob_ai, label, pred_label, threshold, warning?/low_confidence?}"""
        th = float(self.threshold if threshold is None else threshold)
        if isinstance(texts, str):
            original: List[str] = [texts]
        else:
            original = list(texts)

        cleaned = self._normalize_texts(original)
        probs = self.predict_proba(
            cleaned, batch_size=batch_size, show_progress=show_progress
        )

        results: List[Dict[str, Any]] = []
        for raw, cleaned_t, prob in zip(original, cleaned, probs):
            item: Dict[str, Any] = {
                "text": raw,
                "prob_ai": prob,
                "threshold": th,
            }
            if prob is None:
                item["label"] = None
                item["pred_label"] = None
                item["warning"] = "빈 입력이라 신뢰도가 없습니다 (prob_ai=NaN)."
                item["low_confidence"] = True
            else:
                lab = int(prob >= th)
                item["label"] = lab
                item["pred_label"] = lab
                if len(cleaned_t) < self.min_chars:
                    item["low_confidence"] = True
                    item["warning"] = (
                        f"짧은 입력(정제 후 {len(cleaned_t)}자 < {self.min_chars}). "
                        "신뢰도가 낮을 수 있습니다."
                    )
            results.append(item)
        return results


def _load_meta_bundle(ckpt: Path) -> Dict[str, Any]:
    """inference_meta / threshold / run_config 를 모아 반환."""
    bundle: Dict[str, Any] = {
        "threshold": None,
        "max_length": None,
        "preprocess_options": {},
        "sources": {},
    }
    meta_path = ckpt / "inference_meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        bundle["threshold"] = meta.get("threshold")
        bundle["max_length"] = meta.get("max_length")
        bundle["preprocess_options"] = meta.get("preprocess_options") or {}
        bundle["sources"]["inference_meta"] = str(meta_path)
        return bundle

    # 구버전 호환: threshold.json
    for cand in [ckpt / "threshold.json", ckpt.parent / "threshold.json"]:
        if cand.exists():
            payload = load_json(cand)
            bundle["threshold"] = payload.get("threshold", 0.5)
            bundle["sources"]["threshold_json"] = str(cand)
            break

    for cand in [ckpt.parent / "run_config.json"]:
        if cand.exists():
            rc = load_json(cand)
            if bundle["max_length"] is None:
                bundle["max_length"] = rc.get("max_length")
            po = {
                "collapse_repeats": rc.get("collapse_repeats", False),
                "max_repeat": rc.get("max_repeat"),
                "newline_token": rc.get("newline_token", False),
            }
            bundle["preprocess_options"] = po
            bundle["sources"]["run_config"] = str(cand)
            break

    return bundle


def load_detector(
    checkpoint_dir: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    max_length: Optional[int] = None,
    threshold: Optional[float] = None,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> Detector:
    """체크포인트에서 Detector 로드.

    우선순위: 인자(threshold/max_length) > inference_meta.json > threshold.json/run_config > 기본값.
    """
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {ckpt}\n예: outputs/smoke/model"
        )

    torch_device = (
        detect_device(str(device) if device is not None else None)
        if not isinstance(device, torch.device)
        else device
    )
    logger.info("Detector 로드: %s (device=%s)", ckpt, torch_device)

    bundle = _load_meta_bundle(ckpt)
    sources: Dict[str, str] = {}

    # max_length
    resolved_max = 128
    if max_length is not None:
        resolved_max = int(max_length)
        sources["max_length"] = "CLI"
    elif bundle.get("max_length") is not None:
        resolved_max = int(bundle["max_length"])
        sources["max_length"] = bundle["sources"].get(
            "inference_meta", bundle["sources"].get("run_config", "meta")
        )
    else:
        sources["max_length"] = "default(128)"
        logger.warning(
            "메타데이터에 max_length가 없습니다(구버전 체크포인트). 기본값 128 사용."
        )

    # threshold
    resolved_th = 0.5
    if threshold is not None:
        resolved_th = float(threshold)
        sources["threshold"] = "CLI"
    elif bundle.get("threshold") is not None:
        resolved_th = float(bundle["threshold"])
        sources["threshold"] = bundle["sources"].get(
            "inference_meta",
            bundle["sources"].get("threshold_json", "meta"),
        )
    else:
        sources["threshold"] = "default(0.5)"
        logger.warning(
            "메타데이터/threshold.json이 없습니다(구버전). 기본 임계값 0.5 사용."
        )

    po = bundle.get("preprocess_options") or {}
    logger.info(
        "설정 소스 | max_length=%s←%s | threshold=%.4f←%s",
        resolved_max,
        sources["max_length"],
        resolved_th,
        sources["threshold"],
    )

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    # map_location: GPU 학습 → CPU 추론 가능
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    model.to(torch_device)
    model.eval()

    return Detector(
        model=model,
        tokenizer=tokenizer,
        device=torch_device,
        threshold=resolved_th,
        max_length=resolved_max,
        min_chars=min_chars,
        collapse_repeats=bool(po.get("collapse_repeats", False)),
        max_repeat=po.get("max_repeat"),
        newline_token=bool(po.get("newline_token", False)),
        meta_sources=sources,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SNS AI 텍스트 탐지 추론")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--interactive", action="store_true", default=False)
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "jsonl"],
        default=None,
        help="출력 포맷. 미지정 시 입력 확장자 따름.",
    )
    parser.add_argument("--min_chars", type=int, default=DEFAULT_MIN_CHARS)
    return parser


def _save_frame(df: pd.DataFrame, out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    else:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")


def run_interactive(detector: Detector) -> None:
    print("인터랙티브 모드. 빈 줄=건너뜀, exit 또는 Ctrl-D/Ctrl-Z 종료.")
    while True:
        try:
            line = input("> ").rstrip("\n")
        except EOFError:
            print()
            break
        if line.strip().lower() in {"exit", "quit", "q"}:
            break
        if not line.strip():
            continue
        r = detector.predict(line)[0]
        prob = r["prob_ai"]
        if prob is None:
            print("prob_ai=NaN | low_confidence")
        else:
            print(
                f"prob_ai={prob:.6f} | pred={r['pred_label']} "
                f"({'AI' if r['pred_label'] == 1 else '사람'}) | th={r['threshold']}"
            )
            if r.get("warning"):
                print(f"warning: {r['warning']}")


def main(argv: Optional[List[str]] = None) -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.interactive and args.text is None and args.input_file is None:
        parser.error("--text, --input_file, 또는 --interactive 중 하나를 지정하세요.")

    detector = load_detector(
        args.checkpoint,
        device=args.device,
        max_length=args.max_length,
        threshold=args.threshold,
        min_chars=args.min_chars,
    )

    if args.interactive:
        run_interactive(detector)
        return

    if args.text is not None:
        results = detector.predict(args.text, threshold=args.threshold)
        r = results[0]
        print(f"text     : {r['text']}")
        if r["prob_ai"] is None:
            print("prob_ai  : NaN")
        else:
            print(f"prob_ai  : {r['prob_ai']:.6f}")
        print(
            f"label    : {r['pred_label']} "
            f"({'AI' if r['pred_label'] == 1 else '사람' if r['pred_label'] == 0 else 'N/A'})"
        )
        print(f"threshold: {r['threshold']}")
        if r.get("warning"):
            print(f"warning  : {r['warning']}")
        return

    in_path = Path(args.input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {in_path}")
    suffix = in_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(in_path, encoding="utf-8-sig")
        default_fmt = "csv"
    elif suffix in {".jsonl", ".json"}:
        df = pd.read_json(in_path, lines=True, encoding="utf-8")
        default_fmt = "jsonl"
    else:
        raise ValueError("입력 파일은 CSV 또는 JSONL이어야 합니다.")

    col = args.text_column
    if col not in df.columns:
        raise ValueError(
            f"입력 파일에 '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}"
        )

    texts = df[col].tolist()
    results = detector.predict(
        texts, threshold=args.threshold, batch_size=args.batch_size, show_progress=True
    )
    # 행 수·순서 보존
    assert len(results) == len(df)
    df["prob_ai"] = [
        (float("nan") if r["prob_ai"] is None else r["prob_ai"]) for r in results
    ]
    df["pred_label"] = [r["pred_label"] for r in results]
    df["pred"] = df["pred_label"]  # deprecated alias
    df["threshold"] = [r["threshold"] for r in results]
    df["low_confidence"] = [bool(r.get("low_confidence")) for r in results]
    df["warning"] = [r.get("warning", "") for r in results]

    fmt = args.format or default_fmt
    if args.output_file:
        out_path = Path(args.output_file)
    else:
        ext = ".jsonl" if fmt == "jsonl" else ".csv"
        out_path = in_path.with_name(in_path.stem + "_preds" + ext)
    _save_frame(df, out_path, fmt)
    print(f"저장 완료: {out_path} ({len(df)}행)")


if __name__ == "__main__":
    main()
