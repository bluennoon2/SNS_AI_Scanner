# 이 파일의 역할:
# 전체/버킷별/소스별 분류 지표, FPR, 혼동행렬, 임계값 탐색·비교,
# 리포트(JSON/MD) 및 예측 CSV 저장을 담당한다.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("sns_ai_detector")


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """한 클래스만 있으면 NaN을 반환하는 안전한 ROC-AUC."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        logger.warning(
            "ROC-AUC 계산 불가: 해당 부분집합에 클래스가 하나뿐입니다. NaN 처리합니다."
        )
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_binary_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """이진 분류 지표를 계산한다.

    FPR = FP / (FP + TN) = 사람(0)을 AI(1)로 오판한 비율.
    FNR = FN / (FN + TP)
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    # 한 클래스만 있으면 FPR/FNR이 정의되지 않을 수 있음 → NaN (0으로 위장 금지)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else float("nan")

    metrics: Dict[str, Any] = {
        "n": int(len(y_true_arr)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true_arr, y_prob_arr),
        "fpr": fpr,
        "fnr": fnr,
        "mean_prob_ai": float(np.nanmean(y_prob_arr)) if len(y_prob_arr) else float("nan"),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "label_distribution": {
            "human_0": int((y_true_arr == 0).sum()),
            "ai_1": int((y_true_arr == 1).sum()),
        },
    }
    return metrics


def _default_grid() -> np.ndarray:
    return np.linspace(0.01, 0.99, 99)


def search_threshold_max_f1(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, Any]]:
    """validation에서 F1이 최대인 임계값을 탐색한다.

    주의: 임계값은 validation에서만 결정하고 test에 그대로 적용해야 한다.
    test로 임계값을 튜닝하면 데이터 누수이다.

    동점 확률: 같은 확률값이 많으면 그리드 임계값 사이에서 FPR/F1이
    계단식으로 변한다. 동점 구간에서는 더 낮은 임계값(재현율 유리)을 선호하지
    않고, 그리드 상 최초 최대 F1을 유지한다.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    best_t = 0.5
    best_f1 = -1.0
    best_metrics: Dict[str, Any] = {}
    for t in grid:
        m = compute_binary_metrics(y_true, y_prob, threshold=float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
            best_metrics = m
    logger.info("임계값 탐색(max_f1): threshold=%.3f, f1=%.4f", best_t, best_f1)
    return best_t, best_metrics


def search_threshold_youden(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Youden's J = TPR − FPR 를 최대화하는 임계값."""
    if grid is None:
        grid = _default_grid()
    best_t = 0.5
    best_j = -1e9
    best_metrics: Dict[str, Any] = {}
    for t in grid:
        m = compute_binary_metrics(y_true, y_prob, threshold=float(t))
        tpr = m["recall"]
        fpr = m["fpr"]
        if fpr != fpr:
            continue
        j = float(tpr) - float(fpr)
        if j > best_j:
            best_j = j
            best_t = float(t)
            best_metrics = m
    if not best_metrics:
        best_metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
        best_t = 0.5
    logger.info(
        "임계값 탐색(youden): threshold=%.3f, J=%.4f, f1=%.4f",
        best_t,
        best_j,
        best_metrics.get("f1", float("nan")),
    )
    return best_t, best_metrics


def search_threshold_target_fpr(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    target_fpr: float = 0.05,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, Any]]:
    """FPR ≤ target_fpr 을 만족하는 가장 낮은 임계값을 선택한다.

    주의: 임계값은 validation에서만 결정하고 test에 그대로 적용해야 한다.

    신뢰도: validation의 사람(0) 샘플 수가 1/target_fpr 보다 작으면
    FPR 추정이 불안정하다 (예: FPR 1%에 negative < 100).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    n_neg = int((y_true_arr == 0).sum())
    min_needed = int(np.ceil(1.0 / max(target_fpr, 1e-6)))
    if n_neg < min_needed:
        logger.warning(
            "표본 부족 경고: validation 사람(0) 샘플=%d < 1/target_fpr≈%d. "
            "FPR≤%.3f 추정이 불안정합니다.",
            n_neg,
            min_needed,
            target_fpr,
        )

    if grid is None:
        grid = _default_grid()
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for t in grid:
        m = compute_binary_metrics(y_true, y_prob, threshold=float(t))
        fpr = m["fpr"]
        if fpr == fpr and fpr <= target_fpr:
            candidates.append((float(t), m))

    if not candidates:
        t = float(grid[-1])
        m = compute_binary_metrics(y_true, y_prob, threshold=t)
        logger.warning(
            "target_fpr=%.3f 을 만족하는 임계값이 없습니다. "
            "가장 높은 임계값 %.3f (FPR=%.4f)를 사용합니다.",
            target_fpr,
            t,
            m["fpr"],
        )
        return t, m

    best_t, best_m = min(candidates, key=lambda x: x[0])
    logger.info(
        "임계값 탐색(target_fpr<=%.3f): threshold=%.3f, fpr=%.4f, f1=%.4f",
        target_fpr,
        best_t,
        best_m["fpr"],
        best_m["f1"],
    )
    return best_t, best_m


def select_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    mode: str = "fixed",
    fixed_threshold: float = 0.5,
    target_fpr: Union[float, Sequence[float]] = 0.05,
) -> Tuple[float, Dict[str, Any]]:
    """threshold_mode에 따라 임계값을 선택한다.

    중요: 이 함수는 validation 예측에만 사용한다.
    test 세트로 임계값을 튜닝하는 것은 데이터 누수이므로 금지한다.

    val에 한 클래스뿐이면 0.5로 폴백하고 경고한다.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true_arr)) < 2:
        logger.warning(
            "validation에 한 클래스만 있어 임계값 탐색이 불가합니다. "
            "threshold=0.5 로 폴백합니다."
        )
        m = compute_binary_metrics(y_true, y_prob, threshold=fixed_threshold)
        return float(fixed_threshold), m

    if isinstance(target_fpr, (list, tuple)):
        tfp = float(target_fpr[0]) if target_fpr else 0.05
    else:
        tfp = float(target_fpr)

    if mode == "fixed":
        m = compute_binary_metrics(y_true, y_prob, threshold=fixed_threshold)
        logger.info("임계값 fixed=%.3f", fixed_threshold)
        return float(fixed_threshold), m
    if mode == "max_f1":
        return search_threshold_max_f1(y_true, y_prob)
    if mode == "youden":
        return search_threshold_youden(y_true, y_prob)
    if mode == "target_fpr":
        return search_threshold_target_fpr(y_true, y_prob, target_fpr=tfp)
    raise ValueError(
        f"알 수 없는 threshold_mode: '{mode}'. "
        f"fixed | max_f1 | target_fpr | youden 중 하나를 선택하세요."
    )


def compare_threshold_candidates(
    y_true_val: Sequence[int],
    y_prob_val: Sequence[float],
    y_true_test: Sequence[int],
    y_prob_test: Sequence[float],
    target_fprs: Sequence[float] = (0.05,),
) -> List[Dict[str, Any]]:
    """여러 임계값 후보를 val에서 정하고 test 지표와 함께 비교한다.

    주의: 임계값 자체는 validation에서만 결정한다.
    test 지표는 '선택된 임계값을 test에 적용한 결과'이며,
    test로 임계값을 다시 튜닝하지 않는다.
    val에서 FPR 1%를 맞췄어도 test 달성 FPR은 달라질 수 있다.
    """
    rows: List[Dict[str, Any]] = []

    def _row(name: str, threshold: float) -> Dict[str, Any]:
        val_m = compute_binary_metrics(y_true_val, y_prob_val, threshold=threshold)
        test_m = compute_binary_metrics(y_true_test, y_prob_test, threshold=threshold)
        return {
            "name": name,
            "threshold": threshold,
            "val": {
                "accuracy": val_m["accuracy"],
                "precision": val_m["precision"],
                "recall": val_m["recall"],
                "f1": val_m["f1"],
                "fpr": val_m["fpr"],
                "fnr": val_m["fnr"],
            },
            "test": {
                "accuracy": test_m["accuracy"],
                "precision": test_m["precision"],
                "recall": test_m["recall"],
                "f1": test_m["f1"],
                "fpr": test_m["fpr"],
                "fnr": test_m["fnr"],
            },
            "test_achieved_fpr": test_m["fpr"],
            "val_vs_test_fpr_gap": (
                float(test_m["fpr"] - val_m["fpr"])
                if val_m["fpr"] == val_m["fpr"] and test_m["fpr"] == test_m["fpr"]
                else float("nan")
            ),
        }

    # 0.5
    rows.append(_row("fixed_0.5", 0.5))

    t_f1, _ = search_threshold_max_f1(y_true_val, y_prob_val)
    rows.append(_row("max_f1", t_f1))

    t_youden, _ = search_threshold_youden(y_true_val, y_prob_val)
    rows.append(_row("youden", t_youden))

    for tfp in target_fprs:
        t_tf, _ = search_threshold_target_fpr(
            y_true_val, y_prob_val, target_fpr=float(tfp)
        )
        rows.append(_row(f"target_fpr_{tfp}", t_tf))

    return rows


def threshold_comparison_to_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Threshold Comparison",
        "",
        "> 임계값은 validation에서만 결정. test 지표는 적용 결과이며 test로 튜닝하지 않음.",
        "",
        "| Name | Threshold | Val F1 | Val FPR | Test F1 | Test FPR | FPR gap (test-val) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {_fmt_float(r['threshold'], 3)} | "
            f"{_fmt_float(r['val']['f1'])} | {_fmt_float(r['val']['fpr'])} | "
            f"{_fmt_float(r['test']['f1'])} | {_fmt_float(r['test']['fpr'])} | "
            f"{_fmt_float(r.get('val_vs_test_fpr_gap'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def bootstrap_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float,
    n_boot: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """test 지표의 95% bootstrap 신뢰구간."""
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    n = len(y_true_arr)
    keys = ["accuracy", "precision", "recall", "f1", "fpr", "fnr"]
    store = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_binary_metrics(y_true_arr[idx], y_prob_arr[idx], threshold=threshold)
        for k in keys:
            v = m[k]
            if v == v:
                store[k].append(float(v))
    out: Dict[str, Any] = {}
    for k, vals in store.items():
        if not vals:
            out[k] = {"mean": float("nan"), "ci95": [float("nan"), float("nan")]}
            continue
        arr = np.asarray(vals)
        out[k] = {
            "mean": float(arr.mean()),
            "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        }
    return out


def _subset_metrics(
    part: pd.DataFrame,
    threshold: float,
    min_samples_warn: int,
    key_name: str,
    key_value: str,
) -> Dict[str, Any]:
    m = compute_binary_metrics(
        part["label"].tolist(),
        part["prob_ai"].tolist(),
        threshold=threshold,
    )
    m[key_name] = key_value
    n_classes = part["label"].nunique()
    if n_classes < 2:
        m["single_class"] = True
        m["note"] = "한 클래스만 있어 FPR 또는 FNR/ROC-AUC가 정의되지 않을 수 있음"
    if m["n"] < min_samples_warn:
        m["warning"] = f"표본 부족 (n={m['n']} < {min_samples_warn})"
        logger.warning("%s '%s': %s", key_name, key_value, m["warning"])
    return m


def metrics_by_bucket(
    df: pd.DataFrame,
    prob_col: str = "prob_ai",
    label_col: str = "label",
    bucket_col: str = "length_bucket",
    threshold: float = 0.5,
    min_samples_warn: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """길이 버킷별 지표."""
    results: Dict[str, Dict[str, Any]] = {}
    tmp = df.copy()
    if prob_col != "prob_ai":
        tmp["prob_ai"] = tmp[prob_col]
    if label_col != "label":
        tmp["label"] = tmp[label_col]
    for bucket, part in tmp.groupby(bucket_col, sort=False):
        results[str(bucket)] = _subset_metrics(
            part, threshold, min_samples_warn, "bucket", str(bucket)
        )
    return results


def metrics_by_source(
    df: pd.DataFrame,
    prob_col: str = "prob_ai",
    label_col: str = "label",
    source_col: str = "source_type",
    threshold: float = 0.5,
    min_samples_warn: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """source_type별 지표."""
    if source_col not in df.columns or df[source_col].isna().all():
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    tmp = df.copy()
    if prob_col != "prob_ai":
        tmp["prob_ai"] = tmp[prob_col]
    for source, part in tmp.groupby(source_col, sort=False):
        if pd.isna(source):
            continue
        results[str(source)] = _subset_metrics(
            part, threshold, min_samples_warn, "source_type", str(source)
        )
    return results


def metrics_by_bucket_source(
    df: pd.DataFrame,
    threshold: float = 0.5,
    min_samples_warn: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """버킷 × source_type 교차 지표."""
    if "length_bucket" not in df.columns:
        return {}
    if "source_type" not in df.columns or df["source_type"].isna().all():
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    for (bucket, source), part in df.groupby(
        ["length_bucket", "source_type"], sort=False
    ):
        if pd.isna(source):
            continue
        key = f"{bucket}|{source}"
        m = _subset_metrics(
            part, threshold, min_samples_warn, "cross", key
        )
        m["bucket"] = str(bucket)
        m["source_type"] = str(source)
        results[key] = m
    return results


def build_prediction_frame(
    texts: Sequence[str],
    labels: Sequence[int],
    probs: Sequence[float],
    threshold: float,
    lengths: Optional[Sequence[int]] = None,
    length_buckets: Optional[Sequence[str]] = None,
    source_types: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """평가용 예측 DataFrame을 만든다."""
    probs_arr = np.asarray(probs, dtype=float)
    preds = (probs_arr >= threshold).astype(int)
    data: Dict[str, Any] = {
        "text": list(texts),
        "label": list(labels),
        "prob_ai": probs_arr.tolist(),
        "pred": preds.tolist(),
        "pred_label": preds.tolist(),
        "threshold": [float(threshold)] * len(texts),
    }
    if lengths is not None:
        data["length"] = list(lengths)
    else:
        data["length"] = [len(t) for t in texts]
    if length_buckets is not None:
        data["length_bucket"] = list(length_buckets)
    if source_types is not None:
        data["source_type"] = list(source_types)
    return pd.DataFrame(data)


def _fmt_float(v: Any, digits: int = 4) -> str:
    if v is None:
        return "N/A"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if fv != fv:
        return "NaN"
    return f"{fv:.{digits}f}"


def metrics_to_markdown(report: Dict[str, Any]) -> str:
    """지표 리포트를 Markdown 표로 변환한다."""
    lines: List[str] = ["# 평가 리포트", ""]
    lines.append(f"- threshold: `{report.get('threshold')}`")
    lines.append(f"- split: `{report.get('split', 'test')}`")
    lines.append("")

    overall = report.get("overall", {})
    lines.append("## 전체 지표")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for key in ["n", "accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "fnr", "mean_prob_ai"]:
        val = overall.get(key)
        lines.append(
            f"| {key} | {val if key == 'n' else _fmt_float(val)} |"
        )
    cm = overall.get("confusion_matrix", {})
    lines.append(
        f"| confusion (TN/FP/FN/TP) | {cm.get('tn')}/{cm.get('fp')}/{cm.get('fn')}/{cm.get('tp')} |"
    )
    lines.append("")

    def _table(title: str, mapping: Dict[str, Dict[str, Any]], id_key: str) -> None:
        if not mapping:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            f"| {id_key} | n | Prec | Rec | F1 | FPR | FNR | ROC-AUC | mean_prob | Note |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for key, m in mapping.items():
            note = m.get("warning") or m.get("note") or ""
            lines.append(
                f"| {key} | {m.get('n')} | {_fmt_float(m.get('precision'))} | "
                f"{_fmt_float(m.get('recall'))} | {_fmt_float(m.get('f1'))} | "
                f"{_fmt_float(m.get('fpr'))} | {_fmt_float(m.get('fnr'))} | "
                f"{_fmt_float(m.get('roc_auc'))} | {_fmt_float(m.get('mean_prob_ai'))} | {note} |"
            )
        lines.append("")

    _table("길이 버킷별 지표", report.get("by_length_bucket", {}), "Bucket")
    _table("source_type별 지표", report.get("by_source_type", {}), "Source")
    _table("버킷×source 교차", report.get("by_bucket_source", {}), "Cross")

    if report.get("bootstrap"):
        lines.append("## Bootstrap 95% CI")
        lines.append("")
        lines.append(f"```\n{report['bootstrap']}\n```")
        lines.append("")

    if report.get("summary_warning"):
        lines.append("## 경고 요약")
        lines.append("")
        lines.append(f"> {report['summary_warning']}")
        lines.append("")

    return "\n".join(lines)


def fpr_summary_warning(
    by_bucket: Dict[str, Dict[str, Any]],
    high_fpr: float = 0.15,
) -> Optional[str]:
    """FPR이 높은 버킷이 있으면 한 줄 요약 경고를 만든다."""
    high: List[str] = []
    for bucket, m in by_bucket.items():
        fpr = m.get("fpr")
        if fpr is None or fpr != fpr:
            continue
        if float(fpr) >= high_fpr:
            high.append(f"{bucket}(FPR={float(fpr):.0%})")
    if not high:
        return None
    short_hint = any(h.startswith("<") for h in high)
    msg = " / ".join(high)
    if short_hint:
        return f"{msg} — 짧은 글 오탐 위험 높음"
    return f"{msg} — 해당 구간 예측을 보조 지표로만 사용하세요"


def try_save_confusion_matrix_png(
    cm: Dict[str, int],
    path: Union[str, Path],
) -> bool:
    """matplotlib이 있으면 혼동행렬 PNG를 저장한다."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib이 없어 confusion_matrix.png 저장을 건너뜁니다.")
        return False

    matrix = np.array(
        [[cm.get("tn", 0), cm.get("fp", 0)], [cm.get("fn", 0), cm.get("tp", 0)]]
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0 (Human)", "Pred 1 (AI)"])
    ax.set_yticklabels(["True 0 (Human)", "True 1 (AI)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("혼동행렬 저장: %s", path)
    return True


def evaluate_predictions(
    pred_df: pd.DataFrame,
    threshold: float,
    split_name: str = "test",
    output_dir: Optional[Union[str, Path]] = None,
    bootstrap: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """예측 DataFrame으로 전체/버킷/소스/교차 지표를 계산하고 저장한다."""
    overall = compute_binary_metrics(
        pred_df["label"].tolist(),
        pred_df["prob_ai"].tolist(),
        threshold=threshold,
    )
    by_bucket = {}
    if "length_bucket" in pred_df.columns:
        by_bucket = metrics_by_bucket(pred_df, threshold=threshold)
    by_source = metrics_by_source(pred_df, threshold=threshold)
    by_cross = metrics_by_bucket_source(pred_df, threshold=threshold)
    warning = fpr_summary_warning(by_bucket)
    boot = bootstrap_metrics(
        pred_df["label"].tolist(),
        pred_df["prob_ai"].tolist(),
        threshold=threshold,
        n_boot=bootstrap,
        seed=seed,
    )

    report: Dict[str, Any] = {
        "split": split_name,
        "threshold": threshold,
        "overall": overall,
        "by_length_bucket": by_bucket,
        "by_source_type": by_source,
        "by_bucket_source": by_cross,
        "bootstrap": boot or None,
        "summary_warning": warning,
    }

    logger.info("=== %s 전체 지표 ===", split_name)
    logger.info(
        "Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f AUC=%s FPR=%s",
        overall["accuracy"],
        overall["precision"],
        overall["recall"],
        overall["f1"],
        _fmt_float(overall["roc_auc"]),
        _fmt_float(overall["fpr"]),
    )
    cm = overall["confusion_matrix"]
    logger.info(
        "Confusion TN=%d FP=%d FN=%d TP=%d", cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    )
    if by_bucket:
        logger.info("--- 길이 버킷별 ---")
        for bucket, m in by_bucket.items():
            logger.info(
                "  [%s] n=%d F1=%.4f FPR=%s FNR=%s%s",
                bucket,
                m["n"],
                m["f1"],
                _fmt_float(m["fpr"]),
                _fmt_float(m["fnr"]),
                f" | {m.get('warning', '')}",
            )
    if by_source:
        logger.info("--- source_type별 ---")
        for source, m in by_source.items():
            logger.info(
                "  [%s] n=%d F1=%.4f FPR=%s",
                source,
                m["n"],
                m["f1"],
                _fmt_float(m["fpr"]),
            )
    if warning:
        logger.warning("%s", warning)

    if output_dir is not None:
        from utils import save_json

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_json(report, out / "metrics_report.json")
        (out / "metrics_report.md").write_text(
            metrics_to_markdown(report), encoding="utf-8"
        )
        pred_df.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
        try_save_confusion_matrix_png(cm, out / "confusion_matrix.png")
        logger.info("평가 산출물 저장: %s", out)

    return report
