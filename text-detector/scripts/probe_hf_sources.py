#!/usr/bin/env python
"""HF 사람 데이터 후보 탐색·검증. trust_remote_code 미사용. 기억으로 이름 단정 금지."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_build.quality import hangul_ratio, length_bucket  # noqa: E402

SEARCH_QUERIES = [
    "korean review",
    "korean comment",
    "nsmc",
    "korean hate speech",
    "korean community",
    "korean sns",
]


def length_profile(texts: List[str]) -> Dict[str, Any]:
    if not texts:
        return {"n": 0}
    lengths = [len(t) for t in texts]
    buckets = {"<20": 0, "20~100": 0, ">100": 0}
    for n in lengths:
        buckets[length_bucket(n)] += 1
    hanguls = [hangul_ratio(t) for t in texts]
    return {
        "n": len(texts),
        "mean_len": round(sum(lengths) / len(lengths), 1),
        "buckets": buckets,
        "mean_hangul_ratio": round(sum(hanguls) / len(hanguls), 3),
    }


def repo_file_kinds(api, hf_id: str) -> Dict[str, Any]:
    """스크립트 전용 여부 판별용 파일 종류."""
    out = {"py_scripts": [], "data_files": [], "script_only": False, "error": None}
    try:
        files = list(api.list_repo_files(hf_id, repo_type="dataset"))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    data_ext = (".parquet", ".csv", ".json", ".jsonl", ".txt", ".tsv", ".arrow")
    for f in files:
        low = f.lower()
        if low.endswith(".py") and "/" not in f.rstrip("/").split("/")[-1].replace(
            f.split("/")[-1], ""
        ):
            # top-level or any .py loader
            if low.endswith(".py"):
                out["py_scripts"].append(f)
        if any(low.endswith(ext) for ext in data_ext):
            out["data_files"].append(f)
    # 스크립트만 있고 데이터 파일이 없으면 script-based
    if out["py_scripts"] and not out["data_files"]:
        out["script_only"] = True
    return out


def probe_one(api, hf_id: str, text_field_hint: Optional[str], n_sample: int) -> Dict[str, Any]:
    from datasets import load_dataset

    row: Dict[str, Any] = {
        "hf_id": hf_id,
        "exists": False,
        "gated": None,
        "private": None,
        "license": None,
        "script_only": None,
        "load_ok": False,
        "text_field": None,
        "columns": [],
        "length_profile": {},
        "toxic_hint": False,
        "ai_mix_warning": False,
        "status": "unknown",
        "reason": "",
        "error": None,
    }
    try:
        info = api.dataset_info(hf_id)
        row["exists"] = True
        row["gated"] = bool(getattr(info, "gated", False))
        row["private"] = bool(getattr(info, "private", False))
        card = getattr(info, "cardData", None) or getattr(info, "card_data", None) or {}
        if isinstance(card, dict):
            row["license"] = card.get("license")
        elif hasattr(card, "get"):
            row["license"] = card.get("license")
    except Exception as exc:
        row["status"] = "info_failed"
        row["reason"] = f"dataset_info 실패: {type(exc).__name__}"
        row["error"] = str(exc)[:200]
        return row

    if row["gated"] or row["private"]:
        row["status"] = "gated_or_private"
        row["reason"] = "gated/private — 접근 토큰·승인 필요"
        return row

    kinds = repo_file_kinds(api, hf_id)
    row["script_only"] = kinds.get("script_only")
    if kinds.get("error"):
        row["error"] = kinds["error"]
    if kinds.get("script_only"):
        row["status"] = "script_only_excluded"
        row["reason"] = "로더 스크립트만 있고 parquet/csv/jsonl 없음 — 제외"
        return row

    # streaming 소량 로드
    try:
        ds = load_dataset(hf_id, split="train", streaming=True)
    except Exception as exc:
        # split 이름이 train이 아닐 수 있음 — 재시도 없이 기록
        row["status"] = "load_failed"
        row["reason"] = f"load_dataset 실패: {type(exc).__name__}"
        row["error"] = str(exc)[:240]
        return row

    samples = []
    try:
        for i, ex in enumerate(ds):
            if i >= n_sample:
                break
            samples.append(ex)
    except Exception as exc:
        row["status"] = "stream_failed"
        row["reason"] = f"streaming 읽기 실패: {type(exc).__name__}"
        row["error"] = str(exc)[:240]
        return row

    if not samples:
        row["status"] = "empty"
        row["reason"] = "샘플 0건"
        return row

    cols = list(samples[0].keys())
    row["columns"] = cols
    # text_field 추론
    candidates = []
    if text_field_hint:
        candidates.append(text_field_hint)
    candidates.extend(
        [
            "text",
            "document",
            "content",
            "sentence",
            "review",
            "comment",
            "title",
            "utterance",
        ]
    )
    text_field = None
    for c in candidates:
        if c in cols:
            text_field = c
            break
    if text_field is None:
        # 문자열 컬럼 중 가장 긴 평균
        best, best_len = None, -1.0
        for c in cols:
            vals = [str(s.get(c, "")) for s in samples if s.get(c) is not None]
            if not vals:
                continue
            mean_l = sum(len(v) for v in vals) / len(vals)
            if mean_l > best_len:
                best, best_len = c, mean_l
        text_field = best

    if not text_field:
        row["status"] = "no_text_field"
        row["reason"] = "텍스트 컬럼을 찾지 못함"
        return row

    row["text_field"] = text_field
    texts = [str(s.get(text_field, "") or "") for s in samples]
    texts = [t for t in texts if t.strip()]
    row["length_profile"] = length_profile(texts)
    row["load_ok"] = True
    row["status"] = "ok"
    row["reason"] = "소량 로드 성공"

    # 혐오 힌트
    low_id = hf_id.lower()
    if any(k in low_id for k in ("hate", "toxic", "abuse", "offensive")):
        row["toxic_hint"] = True

    # AI 혼입 경고: 카드에 연도가 있으면 대략 판단 (없으면 경고)
    row["ai_mix_warning"] = True  # 수집시기 불명확 시 기본 경고; YAML에서 낮출 수 있음
    return row


def search_candidates(api, limit_per_query: int = 8) -> List[str]:
    found: List[str] = []
    seen = set()
    for q in SEARCH_QUERIES:
        try:
            for ds in api.list_datasets(search=q, limit=limit_per_query):
                # language filter is best-effort; list_datasets may not filter ko strictly
                cid = getattr(ds, "id", None) or str(ds)
                if cid in seen:
                    continue
                seen.add(cid)
                found.append(cid)
        except Exception as exc:
            print(f"[warn] list_datasets({q!r}) 실패: {type(exc).__name__}: {exc}")
    return found


def load_yaml_sources(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("sources") or [])


def main() -> None:
    parser = argparse.ArgumentParser(description="HF 사람 소스 probe")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "human_sources.yaml"),
    )
    parser.add_argument("--n_sample", type=int, default=200)
    parser.add_argument("--search", action="store_true", default=True)
    parser.add_argument("--no-search", action="store_false", dest="search")
    parser.add_argument("--update_yaml", action="store_true", default=False)
    parser.add_argument("--extra_ids", nargs="*", default=None)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    cfg_path = Path(args.config)
    existing = load_yaml_sources(cfg_path)
    existing_ids = [s["hf_id"] for s in existing if s.get("hf_id")]

    candidates: List[str] = list(existing_ids)
    if args.extra_ids:
        candidates.extend(args.extra_ids)
    if args.search:
        print("=== Hub 검색 중 (실행 결과만 사용) ===")
        searched = search_candidates(api)
        print(f"검색으로 찾은 후보 수: {len(searched)}")
        for cid in searched:
            if cid not in candidates:
                candidates.append(cid)

    # 힌트: YAML의 text_field
    hints = {s["hf_id"]: s.get("text_field") for s in existing if s.get("hf_id")}

    results: List[Dict[str, Any]] = []
    print(f"\n=== probe 대상 {len(candidates)}개 (sample={args.n_sample}) ===\n")
    for hf_id in candidates:
        print(f"... {hf_id}")
        r = probe_one(api, hf_id, hints.get(hf_id), args.n_sample)
        results.append(r)

    # 표 출력
    print("\n| hf_id | status | license | text_field | n | mean_len | buckets | hangul | toxic | ai_warn | reason |")
    print("|---|---|---|---|---:|---:|---|---:|:---:|:---:|---|")
    for r in results:
        lp = r.get("length_profile") or {}
        buckets = lp.get("buckets") or {}
        bstr = (
            f"<20={buckets.get('<20',0)},20~100={buckets.get('20~100',0)},>100={buckets.get('>100',0)}"
            if buckets
            else "-"
        )
        print(
            f"| {r['hf_id']} | {r['status']} | {r.get('license')} | {r.get('text_field')} | "
            f"{lp.get('n', 0)} | {lp.get('mean_len', '-')} | {bstr} | "
            f"{lp.get('mean_hangul_ratio', '-')} | "
            f"{'Y' if r.get('toxic_hint') else ''} | "
            f"{'Y' if r.get('ai_mix_warning') else ''} | {r.get('reason', '')} |"
        )

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n통과(ok): {len(ok)} / {len(results)}")
    for r in ok:
        print(f"  - {r['hf_id']} field={r['text_field']} license={r.get('license')} profile={r.get('length_profile')}")

    if args.update_yaml:
        today = date.today().isoformat()
        by_id = {r["hf_id"]: r for r in results}
        updated: List[Dict[str, Any]] = []
        seen = set()

        for src in existing:
            hid = src.get("hf_id")
            seen.add(hid)
            r = by_id.get(hid)
            item = dict(src)
            if r is None:
                updated.append(item)
                continue
            if r["status"] == "ok":
                item["verified"] = True
                item["checked_at"] = today
                item["enabled"] = True if not item.get("allow_toxic") else item.get("enabled", False)
                if r.get("license"):
                    item["license"] = r["license"]
                if r.get("text_field"):
                    item["text_field"] = r["text_field"]
                item["length_profile"] = r.get("length_profile")
                item["reason"] = "probe 통과 (소량 streaming 로드 확인)"
                if r.get("toxic_hint"):
                    item["allow_toxic"] = True
                    item["enabled"] = False
                    item["reason"] = "혐오/독성 힌트 — 기본 비활성. allow_toxic 명시 후 enabled 가능"
                if r.get("ai_mix_warning"):
                    item["ai_mix_warning"] = True
            else:
                item["verified"] = False
                item["enabled"] = False
                item["checked_at"] = today
                item["reason"] = r.get("reason") or r.get("status")
            updated.append(item)

        # 새로 통과한 검색 결과 추가
        for r in ok:
            if r["hf_id"] in seen:
                continue
            # 뉴스/영문 등 너무 동떨어진 것은 후보로만 추가(비활성) — 한글 비율로 필터
            lp = r.get("length_profile") or {}
            if float(lp.get("mean_hangul_ratio") or 0) < 0.3:
                continue
            toxic = bool(r.get("toxic_hint"))
            updated.append(
                {
                    "hf_id": r["hf_id"],
                    "split": "train",
                    "text_field": r.get("text_field") or "text",
                    "source_type": "comment" if toxic else "review",
                    "license": r.get("license") or "unknown",
                    "enabled": False if toxic else True,
                    "allow_toxic": toxic,
                    "verified": True,
                    "checked_at": today,
                    "length_profile": lp,
                    "ai_mix_warning": bool(r.get("ai_mix_warning")),
                    "reason": "Hub 검색+probe 통과 (신규 추가). 문체 적합성은 사용자 확인 필요.",
                    "collected_before": "unknown",
                }
            )

        out = {
            "sources": updated,
            "_meta": {
                "probed_at": today,
                "note": (
                    "작성자/스레드 ID가 없는 코퍼스는 텍스트 정규화 해시로만 group이 묶이므로 "
                    "같은 작성자 글이 train/test에 섞이는 누수는 통제할 수 없다."
                ),
            },
        }
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
        print(f"\nYAML 업데이트: {cfg_path}")


if __name__ == "__main__":
    main()
