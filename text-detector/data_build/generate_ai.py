# AI 생성/조작 데이터 생성 (resume·즉시 append 지원).

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

from data_build.llm_clients import LLMClient, RateLimitedPool
from data_build.quality import char_jaccard, length_bucket, validate_generated_row
from data_build.schema import (
    empty_row,
    make_gen_group_id,
    make_row_id,
    make_task_id,
)

logger = logging.getLogger("sns_ai_detector")


def load_generation_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_ai_from_files(
    files: Sequence[str],
    human_df: pd.DataFrame,
    default_generator: str = "cursor",
    prompt_version: str = "v1",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Cursor/로컬이 만든 AI JSONL·CSV를 스키마 행으로 변환하고 group_id를 매핑.

    필수 컬럼: text, origin (ai_generated|ai_paraphrased|ai_hybrid)
    권장: parent_id 또는 parent_text (파생 origin), source_type, generator
    """
    by_row = {}
    by_text = {}
    if human_df is not None and len(human_df):
        for _, r in human_df.iterrows():
            rid = str(r["row_id"])
            by_row[rid] = r
            by_text[str(r["text"]).strip()] = r

    rows: List[Dict[str, Any]] = []
    stats = {"loaded": 0, "skipped": 0, "no_parent": 0, "bad_origin": 0}
    valid_origins = {"ai_generated", "ai_paraphrased", "ai_hybrid"}

    for fp in files:
        path = Path(fp)
        if not path.exists():
            logger.warning("AI 파일 없음 — 건너뜀: %s", path)
            continue
        if path.suffix.lower() == ".csv":
            raw = pd.read_csv(path, encoding="utf-8-sig")
        elif path.suffix.lower() == ".jsonl":
            raw = pd.read_json(path, lines=True)
        elif path.suffix.lower() == ".json":
            raw = pd.read_json(path)
        else:
            logger.warning("지원 확장자 아님 — 건너뜀: %s", path)
            continue

        for _, rec in raw.iterrows():
            text = str(rec.get("text", "") or "").strip()
            origin = str(rec.get("origin", "") or "").strip()
            if not text or origin not in valid_origins:
                stats["bad_origin"] += 1
                stats["skipped"] += 1
                continue

            source_type = str(rec.get("source_type") or "post")
            generator = str(rec.get("generator") or default_generator)
            if generator == "mock":
                raise ValueError(
                    f"{path}: generator=mock 이 포함되어 있습니다. "
                    "실데이터 경로에는 mock을 넣을 수 없습니다. generator=cursor|offline 등을 쓰세요."
                )
            pv = str(rec.get("prompt_version") or prompt_version)
            mod_ratio = rec.get("mod_ratio", "")
            parent_id = str(rec.get("parent_id") or "").strip()
            parent_text = str(rec.get("parent_text") or "").strip()

            parent = None
            if parent_id and parent_id in by_row:
                parent = by_row[parent_id]
            elif parent_text and parent_text in by_text:
                parent = by_text[parent_text]
                parent_id = str(parent["row_id"])

            if origin in {"ai_paraphrased", "ai_hybrid"}:
                if parent is None:
                    stats["no_parent"] += 1
                    stats["skipped"] += 1
                    continue
                gid = str(parent["group_id"])
                source_type = str(parent.get("source_type") or source_type)
            else:
                tid = make_task_id(
                    parent_id or "root",
                    origin,
                    pv,
                    generator,
                    str(rec.get("variant_seed") or text[:32]),
                )
                gid = str(rec.get("group_id") or make_gen_group_id(tid))
                parent_id = ""

            ok, cleaned, reason = validate_generated_row(
                text, origin, parent_text=str(parent["text"]) if parent is not None else ""
            )
            if not ok:
                stats["skipped"] += 1
                stats[f"filter_{reason}"] = stats.get(f"filter_{reason}", 0) + 1
                continue

            rows.append(
                empty_row(
                    text=cleaned,
                    label=1,
                    source_type=source_type,
                    group_id=gid,
                    row_id=make_row_id(cleaned, salt=f"{generator}-{origin}-{len(rows)}"),
                    origin=origin,
                    parent_id=parent_id,
                    generator=generator,
                    prompt_version=pv,
                    mod_ratio=mod_ratio if mod_ratio != "" else "",
                )
            )
            stats["loaded"] += 1

    logger.info("extra_ai_files 로드: %s", stats)
    return rows, stats


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        try:
            import os

            os.fsync(f.fileno())
        except Exception:
            pass


def load_completed_task_ids(path: Path) -> set:
    """성공(ok=True) task_id만 완료로 본다.

    예외/쿼터 실패는 resume 시 재시도. 깨진 마지막 줄은 버리고 계속.
    """
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("깨진 jsonl 줄 건너뜀: %s", line[:80])
                continue
            tid = obj.get("task_id")
            if tid and obj.get("ok") is True:
                done.add(tid)
    return done


def split_segments(text: str) -> List[str]:
    """문장/구간 분할."""
    parts = re.split(r"(?<=[.!?。\n])\s+", text.strip())
    parts = [p for p in parts if p.strip()]
    if not parts:
        return [text]
    return parts


def hybrid_synthesize(
    client: LLMClient,
    parent_text: str,
    prompt_version: str,
    generator: str,
    variant_seed: str,
    mod_ratio_range: Tuple[float, float] = (0.2, 0.6),
) -> Tuple[str, float, str]:
    """코드가 구간 선택 → LLM은 선택 구간만 재작성 → 이어붙임.

    Returns: (text, mod_ratio, note)
    """
    segs = split_segments(parent_text)
    if len(segs) <= 1:
        # 폴백: 짧은 원문은 전체 재작성 + 한 문장 확장 (단순 append만 하면 미탐이 많음)
        prompt = (
            f"다음 짧은 한국어 SNS 글을 의미는 유지하되 표현을 바꿔 다시 쓰고, "
            f"자연스러운 한 문장을 더해 25자 이상으로 만드세요. "
            f"원문 복붙·설명 금지. 본문만.\n원문: {parent_text}"
        )
        rewritten = client.generate(
            prompt,
            variant_seed=variant_seed,
            min_len=25,
            max_len=max(80, len(parent_text) * 4),
            temperature=1.05,
        ).strip()
        out = rewritten
        # 원문과 달라진 정도를 mod_ratio 근사로 기록
        ratio = 1.0 - float(char_jaccard(parent_text, out))
        return out, float(min(max(ratio, 0.35), 0.95)), "fallback_full_rewrite"

    rng = random.Random(variant_seed)
    lo, hi = mod_ratio_range
    target = lo + (hi - lo) * rng.random()
    n_edit = max(1, int(round(len(segs) * target)))
    n_edit = min(n_edit, len(segs))
    edit_idx = set(rng.sample(range(len(segs)), n_edit))

    out_segs: List[str] = []
    edited_chars = 0
    total_chars = 0
    for i, seg in enumerate(segs):
        total_chars += len(seg)
        if i in edit_idx:
            prompt = (
                f"다음 한국어 SNS 구간만 의미는 유지하고 표현·어휘·어순을 바꿔 다시 쓰세요. "
                f"원문 구절 복붙 금지. 설명 없이 재작성문만.\n원문: {seg}"
            )
            rewritten = client.generate(
                prompt,
                variant_seed=f"{variant_seed}-{i}",
                min_len=max(3, len(seg) // 2),
                max_len=max(len(seg) * 3, 40),
                temperature=1.0,
            ).strip()
            out_segs.append(rewritten)
            edited_chars += len(seg)
        else:
            out_segs.append(seg)  # 원문 바이트 동일 유지

    out = " ".join(out_segs)
    mod_ratio = edited_chars / max(total_chars, 1)
    return out, float(mod_ratio), "segment_rewrite"


def build_generate_prompt(
    source_type: str,
    style: str,
    topic: str,
    min_len: int,
    max_len: int,
) -> str:
    return (
        f"한국어 SNS {source_type} 글을 한 편만 작성하세요.\n"
        f"- 주제: {topic}\n"
        f"- 말투/스타일: {style}\n"
        f"- 글자 수: {min_len}~{max_len}자\n"
        f"- 실존 인물 사칭·혐오·개인정보 금지\n"
        f"- 설명/서두 없이 본문만 출력\n"
    )


PARAPHRASE_MODE_HINTS = {
    "tone_shift": "의미는 유지하되 말투·어휘를 눈에 띄게 바꿔라.",
    "expand": "의미를 유지한 채 문장을 자연스럽게 늘리고 표현을 재구성하라.",
    "reorder": "정보 순서를 바꾸고 다른 문장 구조로 다시 써라.",
    "sns_messy": "ㅋㅋ/ㅠㅠ/줄임말 등 SNS 구어체로 거칠게 다시 써라. 원문 복붙 금지.",
    "plain_clean": "담백하고 정제된 문장으로 다시 쓰되 원문 구절을 그대로 쓰지 마라.",
}


def build_paraphrase_prompt(
    text: str,
    source_type: str,
    mode: str = "tone_shift",
    style: str = "",
) -> str:
    hint = PARAPHRASE_MODE_HINTS.get(mode, PARAPHRASE_MODE_HINTS["tone_shift"])
    short_rule = ""
    if len(text.strip()) < 25:
        short_rule = (
            "- 원문이 짧다. 의미는 같게 유지하되 **최소 25자 이상**으로 자연스럽게 늘려 재작성하라.\n"
        )
    return (
        f"다음 한국어 SNS {source_type} 글을 재작성하세요.\n"
        f"- 핵심 의미·사실은 유지할 것 (주제가 바뀌면 안 됨)\n"
        f"- {hint}\n"
        f"- 스타일 힌트: {style or '자유'}\n"
        f"{short_rule}"
        f"- 원문 문장을 그대로 복사하지 말 것 (어휘·어순·말투를 바꿀 것)\n"
        f"- 서두·설명·따옴표 없이 본문만 출력\n"
        f"원문: {text}"
    )


def plan_ai_jobs(
    human_df: pd.DataFrame,
    n_generated: int,
    n_paraphrased: int,
    n_hybrid: int,
    gen_cfg: Dict[str, Any],
    generator: str,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """사람 분포에 맞춰 AI 작업 목록 생성."""
    rng = random.Random(seed)
    styles = [s["id"] if isinstance(s, dict) else s for s in gen_cfg.get("styles", ["casual"])]
    topics = list(gen_cfg.get("topics") or ["일상"])
    prompt_version = str(gen_cfg.get("prompt_version", "v1"))
    para_modes_cfg = gen_cfg.get("paraphrase_modes") or []
    para_modes = [
        (m.get("id") if isinstance(m, dict) else m)
        for m in para_modes_cfg
    ] or list(PARAPHRASE_MODE_HINTS.keys())
    length_bins = gen_cfg.get("length_bins") or {
        "short": {"min": 10, "max": 19},
        "medium": {"min": 20, "max": 100},
        "long": {"min": 101, "max": 220},
    }

    # 사람 source×bucket 분포
    hdf = human_df.copy()
    if "length_bucket" not in hdf.columns:
        hdf["length_bucket"] = hdf["text"].astype(str).str.len().map(length_bucket)
    dist = (
        hdf.groupby(["source_type", "length_bucket"]).size().to_dict()
        if len(hdf)
        else {}
    )
    keys = list(dist.keys()) or [("post", "20~100")]
    weights = [dist.get(k, 1) for k in keys]
    total_w = sum(weights) or 1

    def _pick_cell() -> Tuple[str, str]:
        r = rng.random() * total_w
        acc = 0.0
        for k, w in zip(keys, weights):
            acc += w
            if r <= acc:
                return k
        return keys[-1]

    def _len_range(bucket: str) -> Tuple[int, int]:
        mapping = {
            "<20": length_bins.get("short", {"min": 10, "max": 19}),
            "20~100": length_bins.get("medium", {"min": 20, "max": 100}),
            ">100": length_bins.get("long", {"min": 101, "max": 220}),
        }
        conf = mapping.get(bucket, {"min": 20, "max": 100})
        return int(conf["min"]), int(conf["max"])

    jobs: List[Dict[str, Any]] = []

    # generated
    for i in range(n_generated):
        stype, bucket = _pick_cell()
        mn, mx = _len_range(bucket)
        style = rng.choice(styles)
        topic = rng.choice(topics)
        variant = f"gen-{i}-{seed}"
        parent_id = ""
        tid = make_task_id(parent_id or "root", "ai_generated", prompt_version, generator, variant)
        jobs.append(
            {
                "task_id": tid,
                "origin": "ai_generated",
                "source_type": stype,
                "style": style,
                "topic": topic,
                "min_len": mn,
                "max_len": mx,
                "parent_id": "",
                "parent_text": "",
                "parent_group_id": "",
                "variant_seed": variant,
                "prompt_version": prompt_version,
                "generator": generator,
            }
        )

    # paraphrased / hybrid — source_type 균등(라운드로빈)으로 부모 선택 (라벨-소스 교락 완화)
    by_stype: Dict[str, List[Dict[str, Any]]] = {}
    for rec in hdf.to_dict(orient="records"):
        by_stype.setdefault(str(rec.get("source_type", "post")), []).append(rec)
    stype_order = sorted(by_stype.keys()) or ["post"]

    def _parent_jobs(n: int, origin: str) -> None:
        if not any(by_stype.values()):
            return
        counters = {st: 0 for st in stype_order}
        for i in range(n):
            st = stype_order[i % len(stype_order)]
            pool = by_stype.get(st) or next(iter(by_stype.values()))
            p = pool[counters[st] % len(pool)]
            counters[st] = counters.get(st, 0) + 1
            variant = f"{origin}-{i}-{seed}"
            tid = make_task_id(
                str(p["row_id"]), origin, prompt_version, generator, variant
            )
            style = rng.choice(styles)
            mode = rng.choice(para_modes)
            # 짧은 원문은 expand 모드 강제
            if len(str(p.get("text") or "")) < 25:
                mode = "expand"
            jobs.append(
                {
                    "task_id": tid,
                    "origin": origin,
                    "source_type": str(p.get("source_type", "post")),
                    "style": style,
                    "topic": rng.choice(topics),
                    "paraphrase_mode": mode,
                    "min_len": 5,
                    "max_len": 500,
                    "parent_id": str(p["row_id"]),
                    "parent_text": str(p["text"]),
                    "parent_group_id": str(p["group_id"]),
                    "variant_seed": variant,
                    "prompt_version": prompt_version,
                    "generator": generator,
                }
            )

    _parent_jobs(n_paraphrased, "ai_paraphrased")
    _parent_jobs(n_hybrid, "ai_hybrid")
    return jobs


def run_generation_jobs(
    client: LLMClient,
    jobs: List[Dict[str, Any]],
    cache_path: Path,
    fail_path: Path,
    resume: bool = True,
    concurrency: int = 4,
    rps: float = 2.0,
    max_requests: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """작업 실행. 결과는 즉시 jsonl append."""
    done = load_completed_task_ids(cache_path) if resume else set()
    pending = [j for j in jobs if j["task_id"] not in done]
    logger.info("생성 작업: 전체 %d / 완료 %d / 남음 %d", len(jobs), len(done), len(pending))

    if max_requests is not None:
        pending = pending[: max(0, max_requests)]
        logger.info("max_requests=%d → 이번 실행 %d건", max_requests, len(pending))

    filter_counts: Dict[str, int] = {}
    results: List[Dict[str, Any]] = []

    # 이미 캐시에 있는 성공분 로드
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("ok"):
                    results.append(obj["row"])

    def _one(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            origin = job["origin"]
            if origin == "ai_generated":
                prompt = build_generate_prompt(
                    job["source_type"],
                    job["style"],
                    job["topic"],
                    job["min_len"],
                    job["max_len"],
                )
                raw = client.generate(
                    prompt,
                    style=job["style"],
                    topic=job["topic"],
                    min_len=job["min_len"],
                    max_len=job["max_len"],
                    variant_seed=job["variant_seed"],
                    temperature=0.9,
                )
                # 길이 허용 폭을 조금 넓히고, 살짝 넘치면 잘라 통과 (필터 과다 폐기 완화)
                soft_max = int(job["max_len"] * 1.5) + 10
                ok, cleaned, reason = validate_generated_row(
                    raw,
                    origin,
                    target_min=max(5, job["min_len"] // 2),
                    target_max=soft_max,
                )
                if ok and len(cleaned) > job["max_len"] + 30:
                    cleaned = cleaned[: job["max_len"] + 29] + "…"
                mod_ratio = ""
                note = ""
                gid = make_gen_group_id(job["task_id"])
            elif origin == "ai_paraphrased":
                mode = str(job.get("paraphrase_mode") or "tone_shift")
                prompt = build_paraphrase_prompt(
                    job["parent_text"],
                    job["source_type"],
                    mode=mode,
                    style=str(job.get("style") or ""),
                )
                # 온도를 모드별로 높여 표현 다양성 확보
                temp = 1.05 if mode in {"sns_messy", "expand", "reorder"} else 0.95
                raw = client.generate(
                    prompt,
                    variant_seed=job["variant_seed"],
                    min_len=5,
                    max_len=500,
                    temperature=temp,
                )
                # 상한↓: 원문 근접 윤문(미탐) 제거 / 하한↓: 강한 재작성 허용
                ok, cleaned, reason = validate_generated_row(
                    raw,
                    origin,
                    parent_text=job["parent_text"],
                    paraphrase_sim_high=0.82,
                    paraphrase_sim_low=0.05,
                    min_len=8 if len(str(job["parent_text"])) < 25 else 5,
                )
                mod_ratio = ""
                note = mode
                gid = job["parent_group_id"]
            else:  # hybrid
                cleaned_h, mod_ratio_f, note = hybrid_synthesize(
                    client,
                    job["parent_text"],
                    job["prompt_version"],
                    job["generator"],
                    job["variant_seed"],
                )
                # hybrid는 원문 구간 잔존이 있어 paraphrase보다 상한을 조금 완화
                ok, cleaned, reason = validate_generated_row(
                    cleaned_h,
                    origin,
                    parent_text=job["parent_text"],
                    paraphrase_sim_high=0.88,
                    paraphrase_sim_low=0.08,
                    min_len=8 if len(str(job["parent_text"])) < 25 else 5,
                )
                mod_ratio = round(float(mod_ratio_f), 4)
                gid = job["parent_group_id"]

            if not ok:
                filter_counts[reason] = filter_counts.get(reason, 0) + 1
                _append_jsonl(
                    fail_path,
                    {"task_id": job["task_id"], "reason": reason, "origin": origin},
                )
                _append_jsonl(
                    cache_path,
                    {"task_id": job["task_id"], "ok": False, "reason": reason},
                )
                return None

            row = empty_row(
                text=cleaned,
                label=1,
                source_type=job["source_type"],
                group_id=gid,
                row_id=make_row_id(cleaned, salt=job["task_id"]),
                origin=origin,
                parent_id=job.get("parent_id") or "",
                generator=job["generator"],
                prompt_version=job["prompt_version"],
                mod_ratio=mod_ratio if mod_ratio != "" else "",
            )
            if note:
                row["mod_ratio"] = mod_ratio
            payload = {"task_id": job["task_id"], "ok": True, "row": row}
            _append_jsonl(cache_path, payload)
            return row
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            _append_jsonl(
                fail_path,
                {
                    "task_id": job["task_id"],
                    "reason": f"exception:{type(exc).__name__}",
                    "error": str(exc)[:200],
                },
            )
            _append_jsonl(
                cache_path,
                {"task_id": job["task_id"], "ok": False, "reason": "exception"},
            )
            filter_counts["exception"] = filter_counts.get("exception", 0) + 1
            return None

    try:
        pool = RateLimitedPool(concurrency=concurrency, rps=rps)
        outs = pool.map(_one, pending)
        for r in outs:
            if r is not None:
                results.append(r)
    except KeyboardInterrupt:
        logger.warning(
            "KeyboardInterrupt — 현재까지 캐시 보존: %s\n"
            "이어서: 같은 명령에 --resume 을 붙이세요.",
            cache_path,
        )
        raise

    return results, filter_counts
