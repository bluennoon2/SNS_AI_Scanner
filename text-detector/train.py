# 이 파일의 역할:
# KcELECTRA(또는 교체 모델) 파인튜닝 학습 진입점(CLI).
# 데이터 분할 → Trainer 학습 → validation 임계값 선택 → test 평가 → 산출물 저장.

from __future__ import annotations

import inspect
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)

from analyze_errors import analyze_errors
from config import TrainConfig, build_arg_parser, config_from_args, normalize_target_fpr
from dataset import file_content_hash, prepare_datasets
from metrics import (
    build_prediction_frame,
    compare_threshold_candidates,
    compute_binary_metrics,
    evaluate_predictions,
    select_threshold,
    threshold_comparison_to_markdown,
)
from utils import (
    detect_device,
    ensure_dir,
    get_env_info,
    get_training_args_eval_key,
    pin_memory_for_device,
    resolve_precision,
    save_json,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("sns_ai_detector")


class FocalLoss(nn.Module):
    """멀티클래스 focal loss (이진 분류 num_labels=2)."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # label smoothing은 CE 계열 권장. focal에서는 soft target 근사.
        ce = F.cross_entropy(
            logits,
            labels,
            reduction="none",
            label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0,
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            at = alpha.gather(0, labels.view(-1))
            loss = at * loss
        return loss.mean()


class CustomLossTrainer(Trainer):
    """weighted CE / focal / label smoothing을 지원하는 Trainer.

    주의: 가중 손실·focal을 쓰면 확률이 보정되지 않아 0.5 임계값의 의미가
    달라진다. validation에서 임계값 재탐색(--threshold_mode)을 권장한다.
    """

    def __init__(
        self,
        loss_name: str = "ce",
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        label_smoothing: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.loss_name == "focal":
            alpha_t = None
            if self.focal_alpha is not None:
                # alpha: positive class weight 근사 → [1-a, a]
                a = float(self.focal_alpha)
                alpha_t = torch.tensor([1.0 - a, a], dtype=torch.float)
            elif self.class_weights is not None:
                alpha_t = self.class_weights
            loss_fct = FocalLoss(
                gamma=self.focal_gamma,
                alpha=alpha_t,
                label_smoothing=self.label_smoothing,
            )
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            weight = None
            if self.loss_name == "weighted_ce" and self.class_weights is not None:
                weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=self.label_smoothing
                if self.label_smoothing > 0
                else 0.0,
            )
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), labels.view(-1)
            )
        return (loss, outputs) if return_outputs else loss


# 하위 호환 별칭
WeightedTrainer = CustomLossTrainer


def _softmax_ai_probs(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs[:, 1]


def make_compute_metrics(threshold: float = 0.5):
    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        probs = _softmax_ai_probs(logits)
        m = compute_binary_metrics(labels, probs, threshold=threshold)
        return {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": 0.0 if m["roc_auc"] != m["roc_auc"] else m["roc_auc"],
            "fpr": 0.0 if m["fpr"] != m["fpr"] else m["fpr"],
        }

    return compute_metrics


def predict_probs_with_trainer(trainer: Trainer, dataset) -> np.ndarray:
    pred_out = trainer.predict(dataset)
    return _softmax_ai_probs(pred_out.predictions)


def build_training_arguments(
    cfg: TrainConfig,
    use_fp16: bool,
    use_bf16: bool,
    train_size: int,
    device: torch.device,
) -> TrainingArguments:
    """버전 호환 인자명을 반영한 TrainingArguments."""
    eval_key = get_training_args_eval_key()
    metric_name = "f1" if cfg.metric_for_best == "f1" else "loss"
    greater_is_better = cfg.metric_for_best == "f1"
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    pin_mem = pin_memory_for_device(device)

    kwargs: Dict[str, Any] = {
        "output_dir": str(Path(cfg.output_dir) / "checkpoints"),
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.eval_batch_size,
        "learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "lr_scheduler_type": cfg.scheduler,
        "gradient_accumulation_steps": cfg.grad_accum,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "seed": cfg.seed,
        "logging_steps": cfg.logging_steps,
        "save_total_limit": cfg.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": metric_name,
        "greater_is_better": greater_is_better,
        "report_to": [],
        "dataloader_num_workers": cfg.num_workers,
        "remove_unused_columns": False,
        "save_strategy": "epoch",
        "group_by_length": cfg.group_by_length,
        "max_grad_norm": cfg.max_grad_norm,
        "dataloader_pin_memory": pin_mem,
    }
    if "data_seed" in ta_params:
        kwargs["data_seed"] = cfg.seed
    if "dataloader_persistent_workers" in ta_params:
        kwargs["dataloader_persistent_workers"] = cfg.num_workers > 0

    kwargs[eval_key] = "epoch"

    steps_per_epoch = max(1, (train_size + cfg.batch_size - 1) // cfg.batch_size)
    updates_per_epoch = max(1, (steps_per_epoch + cfg.grad_accum - 1) // cfg.grad_accum)
    total_steps = max(1, updates_per_epoch * max(cfg.epochs, 1))
    warmup_steps = int(round(total_steps * cfg.warmup_ratio))
    if cfg.warmup_ratio > 0 and warmup_steps == 0:
        warmup_steps = 1
    warmup_steps = max(0, min(warmup_steps, max(0, total_steps - 1)))

    if "warmup_ratio" in ta_params:
        kwargs["warmup_ratio"] = cfg.warmup_ratio
    else:
        kwargs["warmup_steps"] = warmup_steps
        logger.info(
            "warmup_steps=%d 로 환산 (total_steps≈%d, ratio=%.3f)",
            warmup_steps,
            total_steps,
            cfg.warmup_ratio,
        )

    kwargs = {k: v for k, v in kwargs.items() if k in ta_params or k == eval_key}
    return TrainingArguments(**kwargs)


def _resolve_class_weights(cfg: TrainConfig, labels: np.ndarray) -> Optional[torch.Tensor]:
    use_weighted = cfg.loss == "weighted_ce" or cfg.class_weight == "balanced"
    if not use_weighted and cfg.pos_weight is None:
        return None
    if cfg.pos_weight is not None:
        # [neg_weight=1, pos_weight]
        w = np.array([1.0, float(cfg.pos_weight)], dtype=float)
    else:
        w = compute_class_weight(
            class_weight="balanced", classes=np.array([0, 1]), y=labels
        )
    logger.info("class weights: %s", w.tolist())
    return torch.tensor(w, dtype=torch.float)


def save_inference_meta(
    model_dir: Path,
    cfg: TrainConfig,
    threshold: float,
    env_info: Dict[str, Any],
    data_hash: str,
) -> None:
    """model/inference_meta.json — model/만 복사해도 추론 가능하도록."""
    meta = {
        "base_model": cfg.model_name,
        "max_length": cfg.max_length,
        "threshold": threshold,
        "threshold_mode": cfg.threshold_mode,
        "label_map": {"0": "human", "1": "ai"},
        "preprocess_options": {
            "collapse_repeats": cfg.collapse_repeats,
            "max_repeat": cfg.max_repeat,
            "newline_token": cfg.newline_token,
        },
        "loss": cfg.loss,
        "seed": cfg.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "library_versions": {
            "torch": env_info.get("torch"),
            "transformers": env_info.get("transformers"),
            "datasets": env_info.get("datasets"),
        },
        "train_data_hash": data_hash,
    }
    save_json(meta, model_dir / "inference_meta.json")


def run_training(cfg: TrainConfig) -> Dict[str, Any]:
    """전체 학습 파이프라인."""
    output_dir = ensure_dir(cfg.output_dir)
    logger_local = setup_logging(output_dir / "train.log")
    global logger
    logger = logger_local

    device = detect_device(cfg.device)
    set_seed(cfg.seed, deterministic=cfg.deterministic, device=device)
    hf_set_seed(cfg.seed)

    logger.info("선택된 device: %s", device)
    use_fp16, use_bf16 = resolve_precision(device, cfg.fp16, cfg.bf16, logger)
    logger.info(
        "정밀도: fp16=%s bf16=%s | 유효 배치=%d (batch=%d × accum=%d)",
        use_fp16,
        use_bf16,
        cfg.batch_size * cfg.grad_accum,
        cfg.batch_size,
        cfg.grad_accum,
    )
    if device.type == "mps":
        logger.info(
            "MPS 안내: PYTORCH_ENABLE_MPS_FALLBACK=1 권장, num_workers=0, pin_memory=False"
        )

    env_info = get_env_info(device)
    run_config = {
        **cfg.to_dict(),
        "env": env_info,
        "resolved_fp16": use_fp16,
        "resolved_bf16": use_bf16,
        "effective_batch_size": cfg.batch_size * cfg.grad_accum,
    }
    save_json(run_config, output_dir / "run_config.json")

    if cfg.loss in {"weighted_ce", "focal"} or cfg.class_weight == "balanced":
        logger.warning(
            "가중/focal 손실 사용 중 → 확률이 보정되지 않을 수 있습니다. "
            "validation 임계값 재탐색(--threshold_mode max_f1|youden|target_fpr)을 권장합니다."
        )

    logger.info("토크나이저/모델 로드: %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2
    )

    cache_dir = cfg.cache_dir or str(output_dir / "cache")
    data_hash = (
        file_content_hash(cfg.data_path) if Path(cfg.data_path).exists() else "unknown"
    )

    (
        train_ds,
        val_ds,
        test_ds,
        train_df,
        val_df,
        test_df,
        _data_report,
        _split_report,
    ) = prepare_datasets(
        data_path=cfg.data_path,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        split_strategy=cfg.split_strategy,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
        holdout_source=cfg.holdout_source,
        dedup=cfg.dedup,
        collapse_repeats=cfg.collapse_repeats,
        max_repeat=cfg.max_repeat,
        length_bins=cfg.length_bins,
        splits_dir=output_dir / "splits",
        report_dir=output_dir,
        max_rows=cfg.max_rows,
        skip_bad_lines=cfg.skip_bad_lines,
        strict_split=cfg.strict_split,
        fail_on_leak=cfg.fail_on_leak,
        newline_token=cfg.newline_token,
        tokenize_num_proc=cfg.tokenize_num_proc,
        cache_dir=cache_dir,
        no_cache=cfg.no_cache,
        model_name=cfg.model_name,
    )

    if cfg.epochs <= 0:
        logger.info("epochs<=0 → 데이터 리포트만 생성하고 학습을 건너뜁니다.")
        return {"output_dir": str(output_dir), "skipped_train": True}

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    labels_np = np.asarray(train_df["label"].tolist(), dtype=int)
    class_weights = _resolve_class_weights(cfg, labels_np)

    training_args = build_training_arguments(
        cfg, use_fp16, use_bf16, train_size=len(train_ds), device=device
    )
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
    ]

    trainer_sig = inspect.signature(Trainer.__init__)
    processing_key = (
        "processing_class" if "processing_class" in trainer_sig.parameters else "tokenizer"
    )
    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(threshold=cfg.fixed_threshold),
        callbacks=callbacks,
        loss_name=cfg.loss if cfg.loss != "ce" or class_weights is not None else "ce",
        class_weights=class_weights,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=cfg.focal_alpha,
        label_smoothing=cfg.label_smoothing,
    )
    # ce + no weights → 일반 Trainer도 가능하지만 CustomLossTrainer로 통일
    if cfg.loss == "ce" and class_weights is None and cfg.label_smoothing <= 0:
        # 기존 동작과 동일한 기본 Trainer 경로
        plain = {
            k: v
            for k, v in trainer_kwargs.items()
            if k
            not in {
                "loss_name",
                "class_weights",
                "focal_gamma",
                "focal_alpha",
                "label_smoothing",
            }
        }
        plain[processing_key] = tokenizer
        trainer: Trainer = Trainer(**plain)
    else:
        trainer_kwargs[processing_key] = tokenizer
        effective_loss = cfg.loss
        if class_weights is not None and effective_loss == "ce":
            effective_loss = "weighted_ce"
        trainer_kwargs["loss_name"] = effective_loss
        trainer = CustomLossTrainer(**trainer_kwargs)

    logger.info(
        "학습 시작 (epochs=%d, batch=%d, lr=%s, loss=%s)",
        cfg.epochs,
        cfg.batch_size,
        cfg.lr,
        getattr(trainer, "loss_name", "ce"),
    )
    train_result = trainer.train()
    logger.info("학습 완료: %s", train_result.metrics)

    logger.info(
        "임계값 선택: validation에서만 수행. test로 튜닝하면 데이터 누수입니다."
    )
    val_probs = predict_probs_with_trainer(trainer, val_ds)
    test_probs = predict_probs_with_trainer(trainer, test_ds)

    target_fprs = normalize_target_fpr(cfg.target_fpr)
    threshold, val_metrics_at_th = select_threshold(
        val_df["label"].tolist(),
        val_probs.tolist(),
        mode=cfg.threshold_mode,
        fixed_threshold=cfg.fixed_threshold,
        target_fpr=target_fprs,
    )

    if cfg.compare_thresholds:
        rows = compare_threshold_candidates(
            val_df["label"].tolist(),
            val_probs.tolist(),
            test_df["label"].tolist(),
            test_probs.tolist(),
            target_fprs=target_fprs,
        )
        save_json(rows, output_dir / "threshold_comparison.json")
        (output_dir / "threshold_comparison.md").write_text(
            threshold_comparison_to_markdown(rows), encoding="utf-8"
        )
        logger.info("threshold_comparison 저장 (%d 후보)", len(rows))

    threshold_payload = {
        "threshold": threshold,
        "mode": cfg.threshold_mode,
        "target_fpr": target_fprs,
        "selected_on": "validation",
        "note": "임계값은 validation에서만 결정하고 test에 그대로 적용한다.",
        "val_metrics": val_metrics_at_th,
    }
    save_json(threshold_payload, output_dir / "threshold.json")
    logger.info("선택된 threshold=%.4f (mode=%s)", threshold, cfg.threshold_mode)

    model_dir = ensure_dir(output_dir / "model")
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    save_inference_meta(model_dir, cfg, threshold, env_info, data_hash)
    # threshold.json도 model/에 복사해 model만으로 추론 가능하게
    save_json(threshold_payload, model_dir / "threshold.json")
    logger.info("모델 저장: %s", model_dir)

    pred_df = build_prediction_frame(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        probs=test_probs.tolist(),
        threshold=threshold,
        lengths=test_df["length"].tolist(),
        length_buckets=test_df["length_bucket"].tolist(),
        source_types=test_df["source_type"].tolist()
        if "source_type" in test_df.columns
        else None,
    )
    report = evaluate_predictions(
        pred_df,
        threshold=threshold,
        split_name="test",
        output_dir=output_dir,
        bootstrap=cfg.bootstrap,
        seed=cfg.seed,
    )
    try:
        analyze_errors(
            predictions=pred_df,
            threshold=threshold,
            top_k=50,
            output_dir=output_dir,
        )
    except Exception as exc:
        logger.warning("오류 분석 실패(무시): %s", exc)

    return {
        "threshold": threshold,
        "train_metrics": train_result.metrics,
        "test_report": report,
        "output_dir": str(output_dir),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser(description="KcELECTRA 기반 SNS AI 텍스트 탐지기 학습")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    run_training(cfg)


if __name__ == "__main__":
    main()
