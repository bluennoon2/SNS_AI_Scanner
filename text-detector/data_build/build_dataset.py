# 데이터 구축 오케스트레이터 CLI: 수집 → 생성 → 조립 → 리포트.

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from data_build.collect_human import collect_human
from data_build.generate_ai import (
    load_ai_from_files,
    load_generation_config,
    plan_ai_jobs,
    run_generation_jobs,
)
from data_build.llm_clients import build_client, mask_key_status
from data_build.quality import (
    length_bucket,
    remove_duplicates_and_conflicts,
    shortcut_baseline_auc,
)
from data_build.report import write_inspect_samples, write_reports
from data_build.schema import SCHEMA_COLUMNS, assert_group_consistency, ensure_dataframe
from utils import ensure_dir, save_json, setup_logging

logger = logging.getLogger("sns_ai_detector")

ROOT = Path(__file__).resolve().parent.parent


def parse_length_mix(s: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--length_mix 는 a,b,c 형식 (합≈1)")
    return parts[0], parts[1], parts[2]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SNS AI 탐지기 데이터셋 구축")
    p.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    p.add_argument("--target_total", type=int, default=None)
    p.add_argument("--class_ratio", type=float, default=0.5, help="사람 비율")
    p.add_argument(
        "--provider",
        choices=["auto", "gemini", "openai", "anthropic", "offline", "file", "mock"],
        default="auto",
    )
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--rps", type=float, default=2.0)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--dry_run", action="store_true", default=False)
    p.add_argument("--max_requests", type=int, default=None)
    p.add_argument(
        "--extra_ai_files",
        nargs="*",
        default=None,
        help="Cursor/로컬이 만든 AI JSONL·CSV (origin,text[,parent_id|parent_text])",
    )
    p.add_argument("--extra_human_files", nargs="*", default=None)
    p.add_argument("--extra_text_col", type=str, default="text")
    p.add_argument("--extra_source_type", type=str, default="post")
    p.add_argument("--extra_group_col", type=str, default=None)
    p.add_argument("--length_mix", type=parse_length_mix, default=(0.3, 0.4, 0.3))
    p.add_argument("--ratio_generated", type=float, default=0.25)
    p.add_argument("--ratio_paraphrased", type=float, default=0.5)
    p.add_argument("--ratio_hybrid", type=float, default=0.25)
    p.add_argument(
        "--ai_oversample",
        type=float,
        default=1.5,
        help="유사도 필터 폐기 대비 paraphrase/hybrid 작업 배수 (generated는 1.0)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--mask_pii", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--inspect_n", type=int, default=150)
    p.add_argument(
        "--human_sources",
        type=str,
        default=str(ROOT / "configs" / "human_sources.yaml"),
    )
    p.add_argument(
        "--generation_config",
        type=str,
        default=str(ROOT / "configs" / "generation.yaml"),
    )
    p.add_argument("--ood_generator", type=str, default=None)
    p.add_argument("--skip_shortcut", action="store_true", default=False)
    return p


def resolve_targets(mode: str, target_total: Optional[int], class_ratio: float):
    if target_total is None:
        target_total = 200 if mode == "pilot" else 20000
    n_human = int(round(target_total * class_ratio))
    n_ai = target_total - n_human
    return target_total, n_human, n_ai


def run_build(args: argparse.Namespace) -> Path:
    setup_logging()
    data_dir = ensure_dir(ROOT / "data")
    cache_dir = ensure_dir(data_dir / "cache")

    target_total, n_human, n_ai = resolve_targets(
        args.mode, args.target_total, args.class_ratio
    )
    n_gen = int(round(n_ai * args.ratio_generated))
    n_para = int(round(n_ai * args.ratio_paraphrased))
    n_hyb = max(0, n_ai - n_gen - n_para)

    # 출력 경로 — mock만 _mock 접미사. offline/file은 실데이터 경로로 취급하되 리포트에 명시.
    if args.output:
        out_path = Path(args.output)
    else:
        suffix = "_mock" if args.provider == "mock" else ""
        name = f"dataset_{args.mode}{suffix}.csv"
        out_path = data_dir / name
    if args.provider == "mock" and "_mock" not in out_path.stem:
        out_path = out_path.with_name(out_path.stem + "_mock" + out_path.suffix)
    if args.provider != "mock" and "_mock" in out_path.stem:
        raise ValueError(
            f"실데이터 출력 경로에 _mock 이 있습니다: {out_path}. "
            "이름을 바꾸거나 --provider mock 을 사용하세요."
        )

    run_id = f"{args.mode}_{args.seed}_{args.provider}"
    cache_path = cache_dir / f"generated_{run_id}.jsonl"
    fail_path = cache_dir / f"failures_{run_id}.jsonl"

    logger.info(
        "목표: total=%d human=%d AI=%d (gen=%d para=%d hybrid=%d) provider=%s",
        target_total,
        n_human,
        n_ai,
        n_gen,
        n_para,
        n_hyb,
        args.provider,
    )

    # --- dry_run ---
    if args.dry_run:
        from data_build.llm_clients import _try_load_dotenv

        _try_load_dotenv()
        est = n_gen + n_para + n_hyb
        print(
            f"[dry_run] 예상 생성 작업 ≈ {est}건 "
            f"(generated={n_gen}, paraphrased={n_para}, hybrid≈{n_hyb}+구간)\n"
            f"provider={args.provider}, concurrency={args.concurrency}, "
            f"max_requests={args.max_requests}\n"
            f"extra_ai_files={args.extra_ai_files}\n"
            f"키 상태: gemini={mask_key_status('gemini')} "
            f"openai={mask_key_status('openai')} "
            f"anthropic={mask_key_status('anthropic')}"
        )
        return out_path

    # --- 사람 수집 ---
    human_cache = cache_dir / f"human_{run_id}.csv"
    if human_cache.exists() and args.resume:
        human_df = pd.read_csv(human_cache, encoding="utf-8-sig")
        human_report = {"resumed": True, "n_human": len(human_df)}
        logger.info("사람 캐시 재사용: %s (%d)", human_cache, len(human_df))
    else:
        try:
            human_df, human_report = collect_human(
                config_path=Path(args.human_sources),
                extra_files=args.extra_human_files,
                extra_text_col=args.extra_text_col,
                extra_source_type=args.extra_source_type,
                extra_group_col=args.extra_group_col,
                target_n=n_human,
                length_mix=args.length_mix,
                apply_mask_pii=args.mask_pii,
                seed=args.seed,
            )
        except RuntimeError as exc:
            if args.provider == "mock" and not args.extra_human_files:
                logger.warning(
                    "사람 공개 소스 실패 — mock 모드용 합성 사람 시드를 생성합니다.\n%s",
                    exc,
                )
                human_df = _synthetic_human_seed(n_human, args.seed)
                human_report = {
                    "synthetic_seed": True,
                    "n_human": len(human_df),
                    "note": str(exc),
                }
            else:
                raise
        human_df.to_csv(human_cache, index=False, encoding="utf-8-sig")

    # --- AI: 파일 import 및/또는 생성 ---
    filter_counts: dict = {}
    ai_rows: list = []
    file_stats: dict = {}
    client_name = args.provider
    generator_name = args.provider

    if args.extra_ai_files:
        imported, file_stats = load_ai_from_files(
            args.extra_ai_files,
            human_df,
            default_generator="cursor",
        )
        ai_rows.extend(imported)
        logger.info("파일에서 AI %d행 import", len(imported))

    if args.provider == "file":
        if not args.extra_ai_files:
            raise RuntimeError(
                "--provider file 은 --extra_ai_files 가 필요합니다.\n"
                "예: origin,text,parent_id(또는 parent_text),source_type,generator 컬럼 JSONL/CSV"
            )
        client_name = "file"
        generator_name = "cursor"
    else:
        try:
            client = build_client(args.provider, model=args.model)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(2) from exc
        client_name = client.name
        generator_name = client.model or client.name

        if client.name == "mock":
            logger.warning(
                "⚠ mock 데이터는 파이프라인 동작 확인 전용이며 "
                "실제 성능 평가에 사용하면 안 됩니다."
            )
        if client.name == "offline":
            logger.warning(
                "⚠ offline 생성물은 외부 LLM이 아닙니다. "
                "상용 LLM·실제 SNS AI 성능으로 해석하지 마세요."
            )

        # 이미 파일로 채운 만큼 생성량 축소
        already = len(ai_rows)
        remain = max(0, n_ai - already)
        r_gen = int(round(remain * args.ratio_generated)) if remain else 0
        r_para = int(round(remain * args.ratio_paraphrased)) if remain else 0
        r_hyb = max(0, remain - r_gen - r_para) if remain else 0
        # 필터 폐기율↑에 대비해 para/hybrid 여유 생성 (최종 조립은 나온 행만)
        over = max(1.0, float(args.ai_oversample))
        plan_gen, plan_para, plan_hyb = (
            r_gen,
            int(round(r_para * over)),
            int(round(r_hyb * over)),
        )

        gen_cfg = load_generation_config(Path(args.generation_config))
        jobs = plan_ai_jobs(
            human_df,
            n_generated=plan_gen,
            n_paraphrased=plan_para,
            n_hybrid=plan_hyb,
            gen_cfg=gen_cfg,
            generator=generator_name,
            seed=args.seed,
        )
        gen_rows, filter_counts = run_generation_jobs(
            client,
            jobs,
            cache_path=cache_path,
            fail_path=fail_path,
            resume=args.resume,
            concurrency=args.concurrency,
            rps=args.rps,
            max_requests=args.max_requests,
        )
        ai_rows.extend(gen_rows)

    ai_df = pd.DataFrame(ai_rows) if ai_rows else pd.DataFrame(columns=SCHEMA_COLUMNS)

    # --- 조립 ---
    cols = [c for c in SCHEMA_COLUMNS if c in human_df.columns]
    human_part = human_df[cols].copy()
    merged = pd.concat([human_part, ai_df], ignore_index=True)
    if len(merged) == 0:
        raise RuntimeError("조립 결과가 비어 있습니다.")

    conflicts_path = data_dir / "label_conflicts.csv"
    merged, dup_stats, _ = remove_duplicates_and_conflicts(
        merged, conflicts_path=conflicts_path
    )
    assert_group_consistency(merged)

    if args.ood_generator:
        ood = merged[merged["generator"].astype(str) == args.ood_generator]
        ood_path = data_dir / f"dataset_ood_{args.ood_generator}.csv"
        ood.to_csv(ood_path, index=False, encoding="utf-8-sig")
        logger.info("OOD 저장: %s (%d)", ood_path, len(ood))

    shortcut = {"skipped": True}
    if not args.skip_shortcut:
        shortcut = shortcut_baseline_auc(merged, seed=args.seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("데이터셋 저장: %s (%d행)", out_path, len(merged))

    gens = set(merged["generator"].astype(str)) if len(merged) else set()
    # cursor/offline 템플릿만 offline으로 본다. xdac:/gemini-/lgc: 등 외부 LLM CSV는 실데이터.
    soft_gens = {"offline", "cursor", "mock"}
    has_soft_ai = any(
        (g in soft_gens) or g.startswith("offline") or g.startswith("cursor")
        for g in gens
    )
    offline_used = args.provider == "offline" or has_soft_ai
    # provider=file 이면서 외부 LLM generator 만 있으면 file_ai_used=False 로 취급
    file_ai_used = args.provider == "file" and (
        has_soft_ai or not any(
            ("xdac:" in g) or g.startswith("gemini") or g.startswith("lgc:") or "flash" in g
            for g in gens
        )
    )
    extras = {
        "removal_stats": {
            "generation_filters": filter_counts,
            "file_import": file_stats,
            "dedup": dup_stats,
            "human": human_report,
        },
        "hf_sources": (human_report.get("hf") or {}).get("sources"),
        "shortcut": shortcut,
        "provider": args.provider,
        "generator": generator_name,
        "output": str(out_path),
        "offline_used": offline_used,
        "file_ai_used": file_ai_used,
        "external_llm_files": bool(args.extra_ai_files) and not file_ai_used,
        "disclaimer_offline": (
            "offline/cursor 파일 기반 생성물은 외부 LLM API 결과가 아니며 "
            "실제 SNS AI 글 성능을 보장하지 않습니다."
            if offline_used or file_ai_used
            else None
        ),
        "key_status": {
            "gemini": mask_key_status("gemini"),
            "openai": mask_key_status("openai"),
            "anthropic": mask_key_status("anthropic"),
        },
    }
    write_reports(merged, data_dir, extras=extras)
    write_inspect_samples(
        merged, data_dir / "inspect_samples.csv", n=args.inspect_n, seed=args.seed
    )
    cli_safe = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            cli_safe[k] = str(v)
        elif isinstance(v, tuple):
            cli_safe[k] = list(v)
        else:
            cli_safe[k] = v
    save_json(
        {
            "cli": cli_safe,
            "n_rows": len(merged),
            "output": str(out_path),
            "provider": args.provider,
            "key_status": extras["key_status"],
        },
        data_dir / "build_run_config.json",
    )

    if client_name == "mock":
        print(
            "⚠ mock 데이터셋이 생성되었습니다. 실성능 평가에 사용하지 마세요: "
            f"{out_path}"
        )
    elif extras.get("offline_used") or extras.get("file_ai_used"):
        print(
            "⚠ offline/file(cursor) 경로로 데이터셋이 생성되었습니다. "
            f"외부 LLM API 결과가 아닙니다: {out_path}"
        )
    return out_path


def _synthetic_human_seed(n: int, seed: int) -> pd.DataFrame:
    """mock 전용: HF 실패 시 최소 사람 시드 (generator=human, origin=human_local)."""
    from data_build.schema import empty_row, make_human_group_id, make_row_id

    templates = [
        ("ㅋㅋㅋ 오늘 날씨 미쳤다 #일상", "post"),
        ("점심 뭐먹지 고민되다 김치찌개 먹음 ㅎㅎ", "caption"),
        ("와 이 사진 각도 개잘나옴", "comment"),
        ("카페 라떼 개맛있음 다음에 또 와야지", "review"),
        ("야근 끝... 집에 가서 기절할 듯", "chat"),
        ("시험 공부해야 하는데 넷플릭스부터 킴", "post"),
        ("주말에 한강 가고싶다 날씨만 좋으면", "caption"),
        ("헐 대박 축하해!!!", "comment"),
    ]
    rows = []
    for i in range(n):
        text, st = templates[i % len(templates)]
        text = f"{text} ({i})"
        rows.append(
            empty_row(
                text=text,
                label=0,
                source_type=st,
                group_id=make_human_group_id("synthetic", text=text),
                row_id=make_row_id(text, salt=f"syn-{i}"),
                origin="human_local",
                generator="human",
            )
        )
    df = pd.DataFrame(rows)
    df["length_bucket"] = df["text"].str.len().map(length_bucket)
    return df


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        run_build(args)
    except KeyboardInterrupt:
        print(
            "\n중단됨. 캐시는 보존되었습니다. 같은 명령에 --resume 을 붙여 이어가세요.",
            file=sys.stderr,
        )
        raise SystemExit(130)


if __name__ == "__main__":
    main()
