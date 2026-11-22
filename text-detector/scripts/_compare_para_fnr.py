"""llm_v1 vs llm_v2 held-out paraphrase FNR 비교 (임시 스크립트)."""
from __future__ import annotations

import pandas as pd


def analyze(name: str, pred_path: str, data_path: str, thr: float) -> None:
    pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    data = pd.read_csv(data_path, encoding="utf-8-sig")
    m = pred.merge(
        data[["text", "origin", "generator", "prompt_version"]],
        on="text",
        how="left",
    )
    m = m.copy()
    m["pred2"] = (m["prob_ai"] >= thr).astype(int)
    ai = m[m["label"] == 1]
    tn = int(((m.label == 0) & (m.pred2 == 0)).sum())
    fp = int(((m.label == 0) & (m.pred2 == 1)).sum())
    fn = int(((m.label == 1) & (m.pred2 == 0)).sum())
    tp = int(((m.label == 1) & (m.pred2 == 1)).sum())
    n = len(m)
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    print(f"=== {name} thr={thr} n={n} AI={len(ai)} ===")
    print(
        f"Acc={(tn + tp) / n:.3f} F1={f1:.3f} "
        f"FPR={fp / (tn + fp) if tn + fp else 0:.3f} "
        f"FNR={fn / (fn + tp) if fn + tp else 0:.3f} "
        f"CM TN{tn} FP{fp} FN{fn} TP{tp}"
    )
    print("origin miss:")
    for o, g in ai.groupby(ai["origin"].fillna("NA")):
        fn_o = int((g["pred2"] == 0).sum())
        n_o = len(g)
        print(f"  {o}: n={n_o} FN={fn_o} miss={fn_o / n_o if n_o else 0:.3f}")
    fns = m[(m.label == 1) & (m.pred2 == 0)][
        ["text", "prob_ai", "origin", "length", "length_bucket"]
    ]
    if len(fns):
        print("FN samples:")
        for _, r in fns.iterrows():
            t = str(r.text)[:70].replace("\n", " ")
            print(f"  [{r.origin}] p={r.prob_ai:.3f} len={r.length} {t}")


if __name__ == "__main__":
    analyze(
        "llm_v1",
        "outputs/llm_v1/test_predictions.csv",
        "data/dataset_llm_v1.csv",
        0.42,
    )
    analyze(
        "llm_v2@0.5",
        "outputs/llm_v2/test_predictions.csv",
        "data/dataset_llm_v2.csv",
        0.5,
    )
    analyze(
        "llm_v2@0.60_tFPR",
        "outputs/llm_v2/test_predictions.csv",
        "data/dataset_llm_v2.csv",
        0.6,
    )
