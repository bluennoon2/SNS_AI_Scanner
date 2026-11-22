# 이 파일의 역할:
# 학습/평가/추론에 쓰이는 모든 하이퍼파라미터와 매직 넘버를 dataclass로 모으고,
# argparse와 연동해 CLI에서 덮어쓸 수 있게 한다.
# 기존 인자 이름·기본값은 유지하고, 새 기능만 새 인자로 추가한다.

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class TrainConfig:
    """학습·평가·추론 공통 설정.

    기존 필드 기본값은 변경하지 않는다. 신규 필드는 기존 동작과 동일한 기본값을 둔다.
    """

    # --- 기존 필드 (기본값 유지) ---
    data_path: str = "data/sample.csv"
    output_dir: str = "outputs/run1"
    model_name: str = "beomi/KcELECTRA-base"
    max_length: int = 128
    epochs: int = 5
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    scheduler: str = "linear"
    grad_accum: int = 1
    fp16: bool = False
    bf16: bool = False
    seed: int = 42
    split_strategy: str = "random"
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    holdout_source: Optional[str] = None
    early_stopping_patience: int = 2
    metric_for_best: str = "f1"
    class_weight: str = "none"
    threshold_mode: str = "fixed"
    # target_fpr: 하위 호환을 위해 리스트. 단일 값도 [0.05] 형태.
    target_fpr: List[float] = field(default_factory=lambda: [0.05])
    fixed_threshold: float = 0.5
    length_bins: List[int] = field(default_factory=lambda: [20, 100])
    max_repeat: Optional[int] = None
    collapse_repeats: bool = False
    dedup: bool = False
    num_workers: int = 0
    logging_steps: int = 10
    save_total_limit: int = 2
    min_text_warn_len: int = 5

    # --- 신규 필드 (기본값은 기존 동작과 동일) ---
    max_rows: Optional[int] = None
    strict_split: bool = False
    fail_on_leak: bool = False
    skip_bad_lines: bool = False
    tokenize_num_proc: int = 1
    cache_dir: Optional[str] = None
    no_cache: bool = False
    group_by_length: bool = False
    compare_thresholds: bool = True
    bootstrap: int = 0
    loss: str = "ce"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    pos_weight: Optional[float] = None
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    deterministic: bool = False
    newline_token: bool = False
    device: Optional[str] = None

    def primary_target_fpr(self) -> float:
        """threshold_mode=target_fpr 일 때 사용할 대표 FPR."""
        if not self.target_fpr:
            return 0.05
        return float(self.target_fpr[0])

    def to_dict(self) -> Dict[str, Any]:
        """설정을 JSON 직렬화 가능한 dict로 변환한다."""
        return asdict(self)


def _parse_length_bins(value: str) -> List[int]:
    """'20,100' 형태의 문자열을 정수 리스트로 변환한다."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) < 1:
        raise argparse.ArgumentTypeError(
            "--length_bins 는 쉼표로 구분된 정수여야 합니다. 예: 20,100"
        )
    try:
        bins = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--length_bins 파싱 실패: '{value}'. 예: 20,100"
        ) from exc
    if any(b <= 0 for b in bins):
        raise argparse.ArgumentTypeError("--length_bins 값은 양수여야 합니다.")
    if sorted(bins) != bins:
        raise argparse.ArgumentTypeError("--length_bins 는 오름차순이어야 합니다.")
    return bins


def build_arg_parser(description: str = "SNS AI 텍스트 탐지기") -> argparse.ArgumentParser:
    """공통 CLI 인자를 가진 ArgumentParser를 만든다."""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=description)
    # --- 기존 인자 ---
    parser.add_argument("--data_path", type=str, default=defaults.data_path)
    parser.add_argument("--output_dir", type=str, default=defaults.output_dir)
    parser.add_argument("--model_name", type=str, default=defaults.model_name)
    parser.add_argument("--max_length", type=int, default=defaults.max_length)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--eval_batch_size", type=int, default=defaults.eval_batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["linear", "cosine"],
        default=defaults.scheduler,
    )
    parser.add_argument("--grad_accum", type=int, default=defaults.grad_accum)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--split_strategy",
        type=str,
        choices=["random", "group", "source"],
        default=defaults.split_strategy,
    )
    parser.add_argument("--val_ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--test_ratio", type=float, default=defaults.test_ratio)
    parser.add_argument("--holdout_source", type=str, default=None)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=defaults.early_stopping_patience,
    )
    parser.add_argument(
        "--metric_for_best",
        type=str,
        choices=["f1", "loss"],
        default=defaults.metric_for_best,
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        choices=["none", "balanced"],
        default=defaults.class_weight,
    )
    parser.add_argument(
        "--threshold_mode",
        type=str,
        choices=["fixed", "max_f1", "target_fpr", "youden"],
        default=defaults.threshold_mode,
    )
    # 확장: 여러 FPR 허용. 기본 [0.05] → 기존과 동일
    parser.add_argument(
        "--target_fpr",
        type=float,
        nargs="+",
        default=[0.05],
        help="target_fpr 모드(및 비교표)용 FPR 목표. 예: --target_fpr 0.01 0.05",
    )
    parser.add_argument(
        "--fixed_threshold", type=float, default=defaults.fixed_threshold
    )
    parser.add_argument(
        "--length_bins",
        type=_parse_length_bins,
        default=defaults.length_bins,
        help="길이 버킷 경계. 예: 20,100",
    )
    parser.add_argument(
        "--max_repeat",
        type=int,
        default=None,
        help="같은 문자 반복을 N회로 축약. 미지정 시 비활성.",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        default=False,
        help="분할 전 중복 text를 제거한다. 라벨 충돌 샘플은 모두 제거.",
    )
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)
    parser.add_argument("--logging_steps", type=int, default=defaults.logging_steps)

    # --- 신규 인자 ---
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="디버깅용 최대 행 수 (None=전체).",
    )
    parser.add_argument(
        "--strict_split",
        action="store_true",
        default=False,
        help="group 전략에서 group_id 누락 시 에러.",
    )
    parser.add_argument(
        "--fail_on_leak",
        action="store_true",
        default=False,
        help="group/text 누수 발견 시 에러로 종료.",
    )
    parser.add_argument(
        "--skip_bad_lines",
        action="store_true",
        default=False,
        help="JSONL 깨진 줄을 건너뛴다.",
    )
    parser.add_argument(
        "--tokenize_num_proc",
        type=int,
        default=1,
        help="datasets.map 프로세스 수. Windows/MPS에서는 1 권장.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="토큰화 캐시 디렉터리. 기본: outputs/<run>/cache",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        default=False,
        help="토큰화 캐시를 사용하지 않는다.",
    )
    parser.add_argument(
        "--group_by_length",
        action="store_true",
        default=False,
        help="길이별 배치 그룹핑 (Trainer group_by_length).",
    )
    parser.add_argument(
        "--compare_thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="여러 임계값 후보를 val/test에서 비교 표로 저장 (기본 ON).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="test 지표 95%% CI용 bootstrap 횟수 (0=비활성).",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["ce", "weighted_ce", "focal"],
        default="ce",
        help="손실 함수. class_weight=balanced 는 weighted_ce와 동일하게 동작.",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="positive(AI=1) 클래스 가중치 직접 지정 (weighted_ce).",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="CE 계열 label smoothing (0=off).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="torch.use_deterministic_algorithms(True, warn_only=True). MPS는 건너뜀.",
    )
    parser.add_argument(
        "--newline_token",
        action="store_true",
        default=False,
        help="줄바꿈을 특수 토큰 [NL]로 치환 (학습/추론 동일 적용 필요).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="강제 device (cuda|mps|cpu). 미지정 시 자동.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """argparse Namespace를 TrainConfig로 변환한다."""
    known = {f.name for f in fields(TrainConfig)}
    payload: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in known:
            payload[key] = value
    if payload.get("max_repeat") is not None:
        payload["collapse_repeats"] = True
    # target_fpr 단일 float 방어 (구버전 호출)
    tfp = payload.get("target_fpr")
    if tfp is not None and not isinstance(tfp, list):
        payload["target_fpr"] = [float(tfp)]
    # class_weight=balanced → loss를 weighted_ce로 정렬(명시적 loss=focal 등은 유지)
    if payload.get("class_weight") == "balanced" and payload.get("loss", "ce") == "ce":
        payload["loss"] = "weighted_ce"
    return TrainConfig(**payload)


def merge_cli_overrides(
    config: TrainConfig, overrides: Sequence[str] | None = None
) -> TrainConfig:
    """테스트용: 문자열 인자 목록으로 설정을 덮어쓴다."""
    parser = build_arg_parser()
    args = parser.parse_args(list(overrides) if overrides else [])
    return config_from_args(args)


def normalize_target_fpr(value: Union[float, Sequence[float], None]) -> List[float]:
    """target_fpr를 항상 float 리스트로 정규화한다."""
    if value is None:
        return [0.05]
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]
