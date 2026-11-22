# 이 파일의 역할:
# 저장된 체크포인트로 test(또는 외부 데이터)를 재평가하는 CLI.
# --eval_all 로 분할 없이 전체 평가, 임계값 비교·오류 분석을 지원한다.

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from analyze_errors import analyze_errors
from config import TrainConfig, _parse_length_bins, normalize_target_fpr
from dataset import (
    SNSTextDataset,
    assign_length_bucket,
    load_raw_dataframe,
    split_dataframe,
    validate_and_clean_dataframe,
)
from metrics import (
    build_prediction_frame,
    compare_threshold_candidates,
    evaluate_predictions,
    select_threshold,
    threshold_comparison_to_markdown,
)
from predict import _load_meta_bundle
from utils import detect_device, load_json, pin_memory_for_device, save_json, set_seed, setup_logging

logger = logging.getLogger("sns_ai_detector")


def _softmax_ai_probs(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs[:, 1]


@torch.inference_mode()
def collect_probs(model, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_probs: List[float] = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outputs = model(**batch)
        probs = _softmax_ai_probs(outputs.logits.detach().cpu().numpy())
        all_probs.extend(probs.tolist())
    return np.asarray(all_probs, dtype=float)


def run_evaluate(
    checkpoint: str,
    data_path: str,
    output_dir: str,
    max_length: Optional[int] = None,
    batch_size: int = 64,
    split_strategy: str = "random",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    holdout_source: Optional[str] = None,
    seed: int = 42,
    dedup: bool = False,
    length_bins: Optional[List[int]] = None,
    threshold: Optional[float] = None,
    threshold_mode: Optional[str] = None,
    target_fpr: Optional[List[float]] = None,
    compare_thresholds: bool = False,
    evaluate_split: str = "test",
    eval_all: bool = False,
    num_workers: int = 0,
    error_top_k: int = 50,
    bootstrap: int = 0,
    device: Optional[str] = None,
    strict_split: bool = False,
) -> dict:
    """체크포인트로 평가를 실행한다."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    global logger
    logger = setup_logging(out / "evaluate.log")

    set_seed(seed)
    torch_device = detect_device(device)
    logger.info("선택된 device: %s", torch_device)

    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {ckpt}\n"
            f"--checkpoint 에 model 디렉터리(예: outputs/run1/model)를 지정하세요."
        )

    bundle = _load_meta_bundle(ckpt)
    # max_length 우선순위: CLI > meta > 128
    if max_length is None:
        max_length = int(bundle.get("max_length") or 128)
        logger.info("max_length=%d (메타/기본)", max_length)
    else:
        logger.info("max_length=%d (CLI)", max_length)

    resolved_threshold = threshold
    if resolved_threshold is None and threshold_mode is None:
        if bundle.get("threshold") is not None:
            resolved_threshold = float(bundle["threshold"])
            logger.info(
                "threshold=%.4f ← %s",
                resolved_threshold,
                bundle.get("sources"),
            )
        else:
            resolved_threshold = 0.5
            logger.warning("threshold 메타 없음 → 0.5 사용")

    bins = length_bins or [20, 100]
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    model.to(torch_device)
    model.eval()

    po = bundle.get("preprocess_options") or {}
    raw = load_raw_dataframe(data_path)
    cleaned, _ = validate_and_clean_dataframe(
        raw,
        length_bins=bins,
        newline_token=bool(po.get("newline_token", False)),
        collapse_repeats=bool(po.get("collapse_repeats", False)),
        max_repeat=po.get("max_repeat"),
        report_dir=out,
        require_both_classes=False,
    )

    if eval_all:
        eval_df = cleaned.reset_index(drop=True)
        # 임계값 비교를 위해 임시 val=test 동일 경고
        val_df = eval_df
        test_df = eval_df
        logger.warning(
            "--eval_all: 분할 없이 전체 데이터를 평가합니다. "
            "이 데이터로 임계값을 새로 튜닝하면 같은 데이터 평가는 낙관적입니다."
        )
    else:
        train_df, val_df, test_df, _ = split_dataframe(
            cleaned,
            strategy=split_strategy,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            holdout_source=holdout_source,
            dedup=dedup,
            splits_dir=out / "splits",
            report_dir=out,
            strict_split=strict_split,
        )
        split_map = {"train": train_df, "val": val_df, "test": test_df}
        if evaluate_split not in split_map:
            raise ValueError("evaluate_split은 train|val|test 중 하나여야 합니다.")
        eval_df = split_map[evaluate_split]
        if len(eval_df) == 0:
            raise ValueError(f"{evaluate_split} split이 비어 있습니다.")

    def _probs_for(df):
        ds = SNSTextDataset(
            df["text"].tolist(),
            df["label"].tolist(),
            tokenizer=tokenizer,
            max_length=int(max_length),
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(
                tokenizer=tokenizer, pad_to_multiple_of=8
            ),
            num_workers=num_workers,
            pin_memory=pin_memory_for_device(torch_device),
        )
        return collect_probs(model, loader, torch_device)

    eval_probs = _probs_for(eval_df)

    # 임계값 결정
    tfp = normalize_target_fpr(target_fpr)
    if threshold_mode is not None:
        if eval_all:
            logger.warning(
                "외부/전체 데이터로 임계값 튜닝 중 → 같은 데이터 평가는 낙관적일 수 있습니다."
            )
        # val에서 튜닝 (eval_all이면 eval_df=val)
        tune_df = val_df if not eval_all else eval_df
        tune_probs = _probs_for(tune_df) if not eval_all else eval_probs
        resolved_threshold, _ = select_threshold(
            tune_df["label"].tolist(),
            tune_probs.tolist(),
            mode=threshold_mode,
            fixed_threshold=0.5,
            target_fpr=tfp,
        )
        logger.info("threshold_mode=%s → threshold=%.4f", threshold_mode, resolved_threshold)

    assert resolved_threshold is not None

    if compare_thresholds:
        if eval_all:
            logger.warning(
                "compare_thresholds + eval_all: val≈test 동일 → 낙관적 비교일 수 있습니다."
            )
            val_for_cmp, val_p = eval_df, eval_probs
            test_for_cmp, test_p = eval_df, eval_probs
        else:
            val_for_cmp = val_df
            val_p = _probs_for(val_df)
            test_for_cmp = test_df if evaluate_split == "test" else eval_df
            test_p = eval_probs if evaluate_split == "test" else _probs_for(test_for_cmp)
        rows = compare_threshold_candidates(
            val_for_cmp["label"].tolist(),
            val_p.tolist(),
            test_for_cmp["label"].tolist(),
            test_p.tolist(),
            target_fprs=tfp,
        )
        save_json(rows, out / "threshold_comparison.json")
        (out / "threshold_comparison.md").write_text(
            threshold_comparison_to_markdown(rows), encoding="utf-8"
        )

    # length_bucket 보장
    if "length_bucket" not in eval_df.columns:
        eval_df = eval_df.copy()
        eval_df["length"] = eval_df["text"].str.len()
        eval_df["length_bucket"] = eval_df["length"].apply(
            lambda L: assign_length_bucket(int(L), bins)
        )

    pred_df = build_prediction_frame(
        texts=eval_df["text"].tolist(),
        labels=eval_df["label"].tolist(),
        probs=eval_probs.tolist(),
        threshold=float(resolved_threshold),
        lengths=eval_df["length"].tolist(),
        length_buckets=eval_df["length_bucket"].tolist(),
        source_types=eval_df["source_type"].tolist()
        if "source_type" in eval_df.columns
        else None,
    )
    report = evaluate_predictions(
        pred_df,
        threshold=float(resolved_threshold),
        split_name="all" if eval_all else evaluate_split,
        output_dir=out,
        bootstrap=bootstrap,
        seed=seed,
    )
    analyze_errors(
        predictions=pred_df,
        threshold=float(resolved_threshold),
        top_k=error_top_k,
        output_dir=out,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="저장된 모델로 재평가")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=defaults.data_path)
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=defaults.eval_batch_size)
    parser.add_argument(
        "--split_strategy",
        type=str,
        choices=["random", "group", "source"],
        default=defaults.split_strategy,
    )
    parser.add_argument("--val_ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--test_ratio", type=float, default=defaults.test_ratio)
    parser.add_argument("--holdout_source", type=str, default=None)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--dedup", action="store_true", default=False)
    parser.add_argument(
        "--length_bins", type=_parse_length_bins, default=defaults.length_bins
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--threshold_mode",
        type=str,
        choices=["fixed", "max_f1", "target_fpr", "youden"],
        default=None,
    )
    parser.add_argument("--target_fpr", type=float, nargs="+", default=[0.05])
    parser.add_argument(
        "--compare_thresholds",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--evaluate_split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument(
        "--eval_all",
        action="store_true",
        default=False,
        help="분할 없이 전체 데이터를 평가 세트로 사용",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--error_top_k", type=int, default=50)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--strict_split", action="store_true", default=False)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_evaluate(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        split_strategy=args.split_strategy,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        holdout_source=args.holdout_source,
        seed=args.seed,
        dedup=args.dedup,
        length_bins=args.length_bins,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        target_fpr=args.target_fpr,
        compare_thresholds=args.compare_thresholds,
        evaluate_split=args.evaluate_split,
        eval_all=args.eval_all,
        num_workers=args.num_workers,
        error_top_k=args.error_top_k,
        bootstrap=args.bootstrap,
        device=args.device,
        strict_split=args.strict_split,
    )


if __name__ == "__main__":
    main()
