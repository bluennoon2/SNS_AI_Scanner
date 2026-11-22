# 이 파일의 역할:
# 시드 고정, device 자동 감지, 로깅 설정, JSON 저장 등
# 학습/평가/추론에서 공통으로 쓰는 유틸리티를 제공한다.

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False, device: Optional[torch.device] = None) -> None:
    """random / numpy / torch / cuda 시드를 고정해 재현성을 확보한다.

    Args:
        seed: 정수 시드 값.
        deterministic: True이면 torch.use_deterministic_algorithms(True, warn_only=True).
            MPS에서는 건너뛴다.
        device: deterministic 판단용 device.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN 결정성 (약간의 성능 저하 가능)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DataLoader worker 재현성 (함수 속성으로 보관; 호출 측에서 참조 가능)
    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    set_seed.worker_init_fn = _worker_init_fn  # type: ignore[attr-defined]

    if deterministic:
        dev = device or detect_device()
        if getattr(dev, "type", None) == "mps":
            logging.getLogger("sns_ai_detector").warning(
                "MPS에서는 deterministic 모드를 건너뜁니다."
            )
        else:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)


def detect_device(preferred: Optional[str] = None) -> torch.device:
    """CUDA → MPS → CPU 순으로 device를 자동 선택한다.

    Args:
        preferred: 강제 device 문자열 (cuda|mps|cpu). None이면 자동.

    Returns:
        선택된 torch.device.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def pin_memory_for_device(device: torch.device) -> bool:
    """CUDA에서만 pin_memory=True. MPS/CPU는 False (경고 방지)."""
    return device.type == "cuda"


def get_git_commit() -> Optional[str]:
    """가능하면 현재 git commit 해시를 반환한다."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parent),
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def resolve_precision(
    device: torch.device,
    fp16: bool,
    bf16: bool,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, bool]:
    """요청된 정밀도 옵션을 device 지원 여부에 맞게 조정한다.

    - fp16: CUDA에서만 허용
    - bf16: CUDA(Ampere 이상)에서만 허용 (그 외는 폴백)
    - MPS: fp16/bf16을 끈다.
      이유: MPS 백엔드는 혼합정밀/일부 연산 지원이 불완전하여
      NaN·크래시가 자주 발생하므로 fp32를 기본으로 한다.

    Args:
        device: 현재 device.
        fp16: fp16 요청 여부.
        bf16: bf16 요청 여부.
        logger: 경고 출력용 로거.

    Returns:
        (use_fp16, use_bf16) 튜플.
    """
    log = logger or logging.getLogger(__name__)
    use_fp16 = fp16
    use_bf16 = bf16

    if device.type == "mps":
        # MPS에서는 혼합정밀을 끄는 것이 기본이다.
        # PyTorch MPS는 fp16/bf16 연산·스케일러 지원이 제한적이라
        # 학습 중 불안정하거나 미지원 연산으로 폴백/오류가 날 수 있다.
        if use_fp16 or use_bf16:
            log.warning(
                "MPS device에서는 fp16/bf16을 지원하지 않거나 불안정합니다. "
                "fp32로 폴백합니다."
            )
        return False, False

    if device.type != "cuda":
        if use_fp16 or use_bf16:
            log.warning(
                "CUDA가 아닌 device(%s)에서 fp16/bf16 요청 → fp32로 폴백합니다.",
                device.type,
            )
        return False, False

    # CUDA
    if use_fp16 and use_bf16:
        log.warning("fp16과 bf16을 동시에 켰습니다. bf16을 우선 사용합니다.")
        use_fp16 = False

    if use_bf16:
        # Ampere(SM 8.0+) 이상에서 bf16 권장
        major, _minor = torch.cuda.get_device_capability()
        if major < 8:
            log.warning(
                "현재 GPU(compute capability %s)는 bf16에 적합하지 않습니다. "
                "fp32로 폴백합니다.",
                major,
            )
            use_bf16 = False

    if use_fp16 and not torch.cuda.is_available():
        log.warning("fp16은 CUDA에서만 허용됩니다. fp32로 폴백합니다.")
        use_fp16 = False

    return use_fp16, use_bf16


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    name: str = "sns_ai_detector",
) -> logging.Logger:
    """콘솔 + (선택) 파일 핸들러를 가진 로거를 설정한다.

    Args:
        log_file: 로그 파일 경로. None이면 콘솔만 사용.
        level: 로그 레벨.
        name: 로거 이름.

    Returns:
        설정된 Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    """객체를 UTF-8 JSON 파일로 저장한다.

    Args:
        obj: 직렬화할 객체.
        path: 저장 경로.
        indent: JSON indent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                pass
        return str(o)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=_default)


def load_json(path: Union[str, Path]) -> Any:
    """UTF-8 JSON 파일을 로드한다."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Union[str, Path]) -> Path:
    """디렉터리를 생성하고 Path를 반환한다."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_env_info(device: torch.device) -> Dict[str, Any]:
    """실행 환경 정보(버전, device 등)를 dict로 반환한다."""
    import transformers

    info: Dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "git_commit": get_git_commit(),
    }
    try:
        import datasets as ds

        info["datasets"] = ds.__version__
    except Exception:
        info["datasets"] = None
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)
    return info


def get_training_args_eval_key() -> str:
    """transformers 버전에 따라 evaluation_strategy / eval_strategy 키를 반환한다.

    TrainingArguments 시그니처를 inspect하여 올바른 인자명을 고른다.
    """
    import inspect

    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    if "eval_strategy" in params:
        return "eval_strategy"
    if "evaluation_strategy" in params:
        return "evaluation_strategy"
    # 매우 오래된 버전 대비 기본값
    return "evaluation_strategy"
