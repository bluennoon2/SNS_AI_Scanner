"""한국어 HF 데이터셋 후보 load_dataset 실측 프로브 (trust_remote_code 미사용)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import get_dataset_config_names, load_dataset

ROOT = Path(__file__).resolve().parent.parent

# (hf_id, role_hint, text_field_hints)
CANDIDATES: List[Tuple[str, str, List[str]]] = [
    # --- human informal / review / comment ---
    ("Blpeng/nsmc", "human", ["document", "text"]),
    ("e9t/nsmc", "human", ["document", "text"]),
    ("LLM-SocialMedia/Korean-YouTube-Comment-Sentiment-Dataset", "human", ["text", "comment"]),
    ("jeanlee/kmhas_korean_hate_speech", "human", ["text", "comment", "comments"]),
    ("moon1ite/kor_hate", "human", ["comments", "comment", "text"]),
    ("nayohan/korean-hate-speech", "human", ["comments", "comment", "text", "news_title"]),
    ("smilegate-ai/kor_unsmile", "human", ["문장", "text", "sentence"]),
    ("jason9693/APEACH", "human", ["text", "comment"]),
    ("nayohan/APEACH", "human", ["text", "comment"]),
    ("nayohan/KOLD", "human", ["comment", "text", "OFFENSE_CONTENT"]),
    ("humane-lab/K-HATERS", "human", ["comment", "text", "content"]),
    ("AI-it/korean-hate-speech", "human", ["comments", "text"]),
    ("91veMe4Plus-Project/korean_conversational_dialogue", "human", ["text", "utterance", "dialogue", "content"]),
    ("nayohan/141_korean_multi_session_dialogue", "human", ["text", "utterance", "content"]),
    ("jwengr/DACON-Korean-Review-Obfuscation", "human", ["sentence", "text"]),
    ("HAERAE-HUB/KOREAN-WEBTEXT", "human", ["text", "content"]),
    ("OLAResearch/KOREAN-WEBTEXT", "human", ["text", "content"]),
    # --- AI / synthetic / LLM text ---
    ("keepsteady/XDAC_LLM_comments_kor", "ai", ["comment", "text", "generated", "llm_comment"]),
    ("keepsteady/LGC_test", "ai", ["comment", "text", "generated"]),
    ("HAERAE-HUB/KOREAN-SyntheticText-1.5B", "ai", ["text", "content", "output"]),
    ("OLAResearch/KOREAN-SyntheticText", "ai", ["text", "content", "output"]),
    ("NeurIPS-2026-LEDE/LEDE-dataset", "ai", ["ai_article", "text", "summary", "title"]),
    ("BitTranslate/chatgpt-prompts-Korean", "ai", ["prompt", "text", "instruction"]),
    ("kimhyeongjun/Korean_synthetic_financial_dataset_2024", "ai", ["text", "content", "response"]),
    ("podongchip/korean-synthetic-industry-news", "ai", ["text", "content", "article"]),
    # already known in project
    ("ewhk9887/korean_code_reviews_from_github", "ood", ["review", "text"]),
    ("ghfla/korean-medical-dialogue-summary-dataset", "ood", ["text", "dialogue", "summary"]),
    ("nwirandx/Korean-OSCE-history-taking-dialogues", "ood", ["text", "dialogue"]),
]


_HANGUL = re.compile(r"[가-힣]")


def hangul_ratio(s: str) -> float:
    if not s:
        return 0.0
    return len(_HANGUL.findall(s)) / max(len(s), 1)


def pick_text_field(cols: List[str], hints: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for h in hints:
        if h in cols:
            return h
        if h.lower() in lower:
            return lower[h.lower()]
    # fallback: first string-like name
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ("text", "comment", "document", "sentence", "content", "review", "utter")):
            return c
    return cols[0] if cols else None


def sample_texts(ds, field: str, n: int = 30) -> List[str]:
    out: List[str] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        v = row.get(field)
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def probe(hf_id: str, role: str, hints: List[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "hf_id": hf_id,
        "role_hint": role,
        "ok": False,
        "error": None,
        "configs": None,
        "config_used": None,
        "split": None,
        "columns": None,
        "text_field": None,
        "n_sample": 0,
        "mean_len": None,
        "mean_hangul": None,
        "sample_preview": None,
        "gated_or_auth": False,
        "note": "",
    }
    try:
        configs = get_dataset_config_names(hf_id)
    except Exception as exc:
        configs = []
        cfg_err = f"{type(exc).__name__}: {exc}"
    else:
        cfg_err = None
    row["configs"] = configs[:20] if configs else ([] if cfg_err is None else [f"<config_err:{cfg_err[:80]}>"])

    config = configs[0] if configs else None
    last_err = None
    for conf_try in ([config] if config else [None]) + ([None] if config else []):
        try:
            kwargs: Dict[str, Any] = {"split": "train[:40]", "trust_remote_code": False}
            if conf_try:
                ds = load_dataset(hf_id, conf_try, **kwargs)
                row["config_used"] = conf_try
            else:
                ds = load_dataset(hf_id, **kwargs)
                row["config_used"] = None
            row["split"] = "train[:40]"
            break
        except Exception as exc:
            last_err = exc
            # try test/validation small
            for sp in ("test[:40]", "validation[:40]", "dev[:40]"):
                try:
                    kwargs = {"split": sp, "trust_remote_code": False}
                    if conf_try:
                        ds = load_dataset(hf_id, conf_try, **kwargs)
                        row["config_used"] = conf_try
                    else:
                        ds = load_dataset(hf_id, **kwargs)
                        row["config_used"] = None
                    row["split"] = sp
                    last_err = None
                    break
                except Exception as exc2:
                    last_err = exc2
            if last_err is None:
                break
    else:
        ds = None  # type: ignore

    if last_err is not None and "ds" not in dir():
        msg = str(last_err)
        row["error"] = f"{type(last_err).__name__}: {msg[:220]}"
        low = msg.lower()
        if any(k in low for k in ("gated", "401", "403", "auth", "permission", "access")):
            row["gated_or_auth"] = True
        return row

    try:
        cols = list(ds.column_names) if hasattr(ds, "column_names") else list(ds.features.keys())
        row["columns"] = cols
        field = pick_text_field(cols, hints)
        row["text_field"] = field
        if not field:
            row["error"] = "no_text_field"
            return row
        texts = sample_texts(ds, field, 30)
        row["n_sample"] = len(texts)
        if not texts:
            row["error"] = "empty_texts"
            return row
        row["mean_len"] = round(sum(len(t) for t in texts) / len(texts), 1)
        row["mean_hangul"] = round(sum(hangul_ratio(t) for t in texts) / len(texts), 3)
        row["sample_preview"] = texts[0][:80].replace("\n", " ")
        row["ok"] = row["mean_hangul"] >= 0.25
        if row["mean_hangul"] < 0.25:
            row["note"] = "hangul_ratio_low — 한국어 본문 아님/혼합 가능"
            row["ok"] = False
        else:
            row["ok"] = True
        return row
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:220]}"
        return row


def main() -> None:
    results = []
    for hf_id, role, hints in CANDIDATES:
        print(f"PROBE {hf_id} ...", flush=True)
        r = probe(hf_id, role, hints)
        status = "OK" if r["ok"] else "FAIL"
        print(
            f"  {status} field={r.get('text_field')} hangul={r.get('mean_hangul')} "
            f"len={r.get('mean_len')} err={r.get('error')}",
            flush=True,
        )
        results.append(r)

    out = ROOT / "data" / "hf_ko_candidates_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
