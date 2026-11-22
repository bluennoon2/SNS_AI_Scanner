# LLM provider 추상화 + mock + 재시도/동시성/회로차단.
# API 키는 로그에 절대 출력하지 않는다. SDK는 lazy import.

from __future__ import annotations

import logging
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("sns_ai_detector")

# 환경변수 이름 (값은 절대 로그에 찍지 않음)
ENV_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def mask_key_status(provider: str) -> str:
    """키가 있는지 여부만 반환 (키 문자열 비노출)."""
    env = ENV_KEYS.get(provider)
    if not env:
        return "n/a"
    val = os.environ.get(env, "")
    return "감지됨" if bool(val.strip()) else "없음"


def detect_available_providers() -> List[str]:
    _try_load_dotenv()
    found = []
    for p, env in ENV_KEYS.items():
        if os.environ.get(env, "").strip():
            found.append(p)
    return found


class AuthError(Exception):
    """401/403 — 재시도하지 않음."""


class BadRequestError(Exception):
    """400 — 재시도하지 않음."""


class RateLimitError(Exception):
    """429."""


class ServerError(Exception):
    """5xx / timeout."""


@dataclass
class LLMClient(ABC):
    """공통 인터페이스."""

    name: str = "base"
    model: str = ""
    max_retries: int = 4
    base_backoff: float = 0.5
    circuit_fail_threshold: int = 8
    _consecutive_failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @abstractmethod
    def _generate_once(self, prompt: str, **opts: Any) -> str:
        ...

    def generate(self, prompt: str, **opts: Any) -> str:
        """재시도 포함 생성. 401/400은 즉시 실패, 429/5xx는 백오프."""
        if self._consecutive_failures >= self.circuit_fail_threshold:
            raise RuntimeError(
                f"circuit breaker 발동: {self.name} 연속 실패 "
                f"{self._consecutive_failures}회. "
                f"키/쿼터/네트워크를 확인하거나 --provider mock 을 사용하세요."
            )

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                out = self._generate_once(prompt, **opts)
                with self._lock:
                    self._consecutive_failures = 0
                return out
            except (AuthError, BadRequestError):
                with self._lock:
                    self._consecutive_failures += 1
                raise
            except (RateLimitError, ServerError, TimeoutError, ConnectionError) as exc:
                last_exc = exc
                # 429/쿼터는 대기 후 재시도 — circuit breaker에 누적하지 않음
                if not isinstance(exc, RateLimitError):
                    with self._lock:
                        self._consecutive_failures += 1
                if attempt >= self.max_retries:
                    break
                sleep_s = self.base_backoff * (2**attempt) + random.uniform(0, 0.25)
                if isinstance(exc, RateLimitError):
                    sleep_s = max(sleep_s, 20.0 + attempt * 10.0)
                # 테스트에서 time.sleep 모킹 가능
                time.sleep(sleep_s)
            except Exception as exc:
                # 알 수 없는 오류는 1회만 재시도 후 실패
                last_exc = exc
                with self._lock:
                    self._consecutive_failures += 1
                if attempt >= 1:
                    break
                time.sleep(self.base_backoff)
        raise RuntimeError(f"{self.name} 생성 실패: {last_exc}") from last_exc


@dataclass
class MockLLMClient(LLMClient):
    """규칙 기반 가짜 생성기 — 파이프라인 완주 검증용."""

    name: str = "mock"
    model: str = "mock"

    def _generate_once(self, prompt: str, **opts: Any) -> str:
        style = opts.get("style", "casual")
        topic = opts.get("topic", "일상")
        min_len = int(opts.get("min_len", 20))
        max_len = int(opts.get("max_len", 80))
        seed = str(opts.get("variant_seed", "0"))

        # paraphrase / hybrid 구간: 프롬프트에 원문이 있으면 어미 변환
        marker = "원문:"
        if marker in prompt:
            src = prompt.split(marker, 1)[1].strip().split("\n")[0]
            out = _mock_rewrite(src, seed)
        else:
            templates = [
                f"오늘 {topic} 얘기인데 진짜 괜찮았음 ㅋㅋ",
                f"{topic} 다녀옴. 생각보다 좋더라 #일상",
                f"와 {topic} 미쳤다 ㄹㅇ 추천",
                f"{topic}에 대해 간단히 남겨둡니다. 전반적으로 만족스러웠습니다.",
                f"{topic}...\n그냥 그런 하루였음 ㅠㅠ",
            ]
            # 결정적 인덱스 (내장 hash() 금지 — seed 문자열 기반)
            seed_key = seed or "0"
            idx = sum(ord(c) for c in seed_key) % len(templates)
            out = templates[idx]
            if "emoji" in style or "casual" in style:
                out += " 😂"
            if "hashtag" in style:
                out += " #일상 #기록"
            if "jamo" in style:
                out += " ㅋㅋㅋ"
            if "formal" in style or "plain" in style:
                out = f"{topic} 관련 소감을 정리합니다. 전반적으로 무난했습니다."
            if "rough" in style:
                out = f"{topic} 걍 뭐.. 그럭저럭? ㅋㅋ"
            # 작업마다 고유 접미로 텍스트 충돌 방지 (mock 전용)
            out = f"{out} [{seed_key[-8:]}]"

        # 길이 맞춤
        while len(out) < min_len:
            out += " 그래도 괜찮았음"
        if len(out) > max_len:
            out = out[: max_len - 1] + "…"
        return out


def _mock_rewrite(src: str, seed: str) -> str:
    """재작성 mock — 유사도 필터(sim_high≈0.82)를 통과할 만큼 변형."""
    t = (src or "").strip()
    tag = (seed or "0")[-6:]
    n = sum(ord(c) for c in (seed or "0"))
    modes = [
        f"솔직히 말하면 {t} 그런 느낌이었어 ㅋㅋ",
        f"다시 정리하면… {t} 요지가 이거임",
        f"{t} ← 이거 완전 그거네. 내 말로 하면 비슷함",
        f"한줄평: {t} 근데 표현은 다르게 씀 ({tag})",
        f"와 이거 보면 {t} 분위기인데 말투만 바꿈 ㄹㅇ",
    ]
    out = modes[n % len(modes)]
    if len(t) < 25:
        out = f"{out} 짧은 원문이라 조금 늘려 적음 #기록{tag}"
    else:
        out = f"{out} #기록{tag}"
    return out


@dataclass
class OfflineLLMClient(LLMClient):
    """API 없는 오프라인 생성기. generator=offline (mock과 구분).

    Cursor/로컬에서 미리 만든 파일을 쓸 때는 --extra_ai_files 를 권장.
    이 클라이언트는 파일이 없을 때 파이프라인을 채우기 위한 템플릿 경로다.
    """

    name: str = "offline"
    model: str = "offline"

    def _generate_once(self, prompt: str, **opts: Any) -> str:
        style = opts.get("style", "casual")
        topic = opts.get("topic", "일상")
        min_len = int(opts.get("min_len", 20))
        max_len = int(opts.get("max_len", 80))
        seed = str(opts.get("variant_seed", "0"))
        seed_key = seed or "0"
        n = sum(ord(c) for c in seed_key)

        marker = "원문:"
        if marker in prompt:
            # hybrid/paraphrase: 프롬프트에 여러 줄이 있어도 원문 블록 전체 사용
            src = prompt.split(marker, 1)[1].strip()
            # 첫 지시문 줄만 오지 않도록 이어진 본문 유지
            out = _offline_rewrite(src, seed)
        else:
            templates = [
                f"오늘 {topic} 얘기. 생각보다 괜찮아서 남김 ㅎㅎ",
                f"{topic} 다녀옴. 사람 좀 많았는데 분위기는 좋더라",
                f"와 {topic} 추천함… 다음에 또 갈 듯 ㅋㅋ",
                f"{topic} 후기: 전반적으로 무난. 재방문할 만함.",
                f"{topic}…\n그냥 그런 하루였음. ㅠㅠ",
                f"{topic} 걍 그럭저럭. 기대보단 별로",
                f"요즘 {topic} 관심 생겨서 찾아보는 중. 정보는 아직 부족함",
                f"{topic} 관련해서 친구랑 이야기하다가 적어봄. 의견 갈림",
                f"짧게: {topic} 괜찮았음. 사진도 대충 찍어둠",
                f"{topic} 때문에 일정 꼬였는데 결과는 나쁘지 않음",
            ]
            idx = n % len(templates)
            out = templates[idx]
            style_s = str(style)
            if "emoji" in style_s or "casual" in style_s:
                out += [" ✨", " 😂", " 🙂"][n % 3]
            if "hashtag" in style_s:
                out += " #일상 #후기"
            if "jamo" in style_s:
                out += " ㅋㅋㅋ"
            if "formal" in style_s or "plain" in style_s:
                out = (
                    f"{topic}에 대한 소감을 정리합니다. "
                    f"전체적으로 {'만족' if n % 2 == 0 else '보통'}이었습니다."
                )
            if "rough" in style_s:
                out = f"{topic} 뭐 그냥.. {'ㄱㅊ' if n % 2 else '별로'}? ㅋㅋ"
            if "multiline" in style_s and "\n" not in out:
                out = out + f"\n다음엔 다르게 해볼지도 ({n % 9})"
            # 길이 버킷 다양화: seed에 따라 문장 보강 (지름길 완화)
            extras = [
                "",
                " 디테일은 나중에.",
                " 위치는 그냥 근처였음.",
                " 가격대는 보통.",
                " 같이 간 사람도 비슷하게 느낌.",
                " 다음 주에도 비슷한 거 할 예정.",
            ]
            out = out + extras[n % len(extras)]
            out = f"{out} ·{seed_key[-6:]}"

        while len(out) < min_len:
            out += " 그래도 기록용으로 남김"
        if len(out) > max_len:
            out = out[: max_len - 1] + "…"
        return out


def _offline_rewrite(src: str, seed: str) -> str:
    """문장 재작성 강화: 어휘 교체 + 어미/어순 변형 + 부가 구. 의미는 유지."""
    import re

    t = (src or "").strip()
    if not t:
        return "내용을 다시 정리해 봤음"
    seed_key = seed or "0"
    n = sum(ord(c) for c in seed_key)

    # 1) 어휘 치환 (여러 개 적용)
    lexicon = [
        ("정말로", "진짜"),
        ("정말", "완전"),
        ("너무", "좀"),
        ("진짜", "ㄹㅇ"),
        ("오늘", "금일" if n % 5 == 0 else "오늘"),
        ("내일", "다음날"),
        ("좋다", "괜찮다"),
        ("좋음", "괜찮음"),
        ("맛있", "맛좋"),
        ("재미있", "재밌"),
        ("입니다", "이에요"),
        ("습니다", "어요"),
        ("했어요", "했음"),
        ("했습니다", "했음"),
        ("네요", "는데"),
        ("같아요", "같음"),
        ("거예요", "거임"),
        ("거예요", "거임"),
        ("근데", "그런데"),
        ("그리고", "또"),
        ("그래서", "그래서인지"),
    ]
    applied = 0
    for a, b in lexicon:
        if a in t and a != b:
            t = t.replace(a, b, 1)
            applied += 1
            if applied >= 3:
                break

    # 2) 문장 분할 후 일부 재배치/어미 변경
    parts = [p.strip() for p in re.split(r"(?<=[.!?。\n])\s+", t) if p.strip()]
    if len(parts) >= 2:
        # 마지막 문장 어미 변형
        last = parts[-1]
        if last.endswith("다"):
            parts[-1] = last[:-1] + ("네요" if n % 2 == 0 else "음")
        elif last.endswith("요"):
            parts[-1] = last[:-1] + ("음" if n % 2 == 0 else "다더라")
        else:
            parts[-1] = last + (" ㅋㅋ" if n % 2 == 0 else " …")
        if n % 3 == 0 and len(parts) >= 2:
            parts[0], parts[1] = parts[1], parts[0]
        t = " ".join(parts)
    else:
        if t.endswith("다"):
            t = t[:-1] + ("네요" if n % 2 == 0 else "음")
        elif t.endswith("요"):
            t = t[:-1] + ("음" if n % 2 == 0 else "다고 함")
        else:
            t = t + (" ㅎㅎ" if n % 2 == 0 else " ㅋㅋ")

    # 3) 부가 구(의미 유지 범위의 짧은 코멘트) — 접미만 붙이던 방식 탈피
    addons = [
        " 다시 보니 비슷한 느낌.",
        " 말로 바꾸면 이런 톤.",
        " 표현만 조금 다르게 적음.",
        " 요약하면 큰 틀은 같음.",
        " 뉘앙스만 살짝 조정.",
        " 같은 내용인데 문장만 손봄.",
    ]
    t = t + addons[n % len(addons)]
    # 과도한 해시태그 접미 대신 약한 마커
    if n % 4 == 0:
        t = t + f" ({seed_key[-4:]})"
    return t.strip()


@dataclass
class GeminiClient(LLMClient):
    """google-genai (genai.Client) 전용. 구형 google-generativeai 폴백 없음."""

    name: str = "gemini"

    def _generate_once(self, prompt: str, **opts: Any) -> str:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise AuthError("GEMINI_API_KEY 없음")
        model_name = opts.get("model") or self.model or "gemini-3.1-flash-lite"
        temperature = float(opts.get("temperature", 0.8))

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai 미설치. pip install -U google-genai"
            ) from exc

        client = genai.Client(api_key=api_key)
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            text = getattr(resp, "text", None) or ""
            if not text:
                # candidates/parts 폴백
                try:
                    cands = getattr(resp, "candidates", None) or []
                    parts = []
                    for c in cands:
                        content = getattr(c, "content", None)
                        for part in getattr(content, "parts", None) or []:
                            t = getattr(part, "text", None)
                            if t:
                                parts.append(t)
                    text = "\n".join(parts).strip()
                except Exception:
                    text = ""
            if not text:
                raise ServerError("빈 응답")
            return text
        except (AuthError, BadRequestError, RateLimitError, ServerError):
            raise
        except Exception as exc:
            raise self._classify_gemini_error(exc) from exc

    @staticmethod
    def _classify_gemini_error(exc: Exception) -> Exception:
        msg = str(exc)
        low = msg.lower()
        if (
            "api key" in low
            or "api_key_invalid" in low
            or "401" in low
            or "403" in low
            or "permission denied" in low
            or "unauthenticated" in low
        ):
            return AuthError(msg)
        if "429" in low or "quota" in low or "rate" in low or "resource_exhausted" in low:
            return RateLimitError(msg)
        if "404" in low or "not found" in low or "not_found" in low:
            return BadRequestError(msg)
        if "400" in low or "invalid_argument" in low or "invalid" in low:
            # 키 invalid는 위에서 이미 AuthError
            return BadRequestError(msg)
        return ServerError(msg)


@dataclass
class OpenAIClient(LLMClient):
    name: str = "openai"

    def _generate_once(self, prompt: str, **opts: Any) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai 미설치. pip install openai 또는 --provider mock"
            ) from exc
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise AuthError("OPENAI_API_KEY 없음")
        client = OpenAI(api_key=api_key)
        model_name = opts.get("model") or self.model or "gpt-4o-mini"
        temperature = float(opts.get("temperature", 0.8))
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            msg = str(exc).lower()
            if "401" in msg or "auth" in msg or "api key" in msg:
                raise AuthError(str(exc)) from exc
            if "429" in msg or "rate" in msg:
                raise RateLimitError(str(exc)) from exc
            if "400" in msg:
                raise BadRequestError(str(exc)) from exc
            raise ServerError(str(exc)) from exc


@dataclass
class AnthropicClient(LLMClient):
    name: str = "anthropic"

    def _generate_once(self, prompt: str, **opts: Any) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic 미설치. pip install anthropic 또는 --provider mock"
            ) from exc
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise AuthError("ANTHROPIC_API_KEY 없음")
        client = anthropic.Anthropic(api_key=api_key)
        model_name = opts.get("model") or self.model or "claude-3-5-haiku-latest"
        temperature = float(opts.get("temperature", 0.8))
        try:
            resp = client.messages.create(
                model=model_name,
                max_tokens=1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = []
            for block in resp.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts).strip()
        except Exception as exc:
            msg = str(exc).lower()
            if "401" in msg or "auth" in msg or "api key" in msg:
                raise AuthError(str(exc)) from exc
            if "429" in msg or "rate" in msg:
                raise RateLimitError(str(exc)) from exc
            if "400" in msg:
                raise BadRequestError(str(exc)) from exc
            raise ServerError(str(exc)) from exc


def build_client(
    provider: str = "auto",
    model: Optional[str] = None,
) -> LLMClient:
    """provider 선택. auto는 키가 있는 쪽 우선(gemini>openai>anthropic)."""
    _try_load_dotenv()
    provider = (provider or "auto").lower()

    statuses = {p: mask_key_status(p) for p in ENV_KEYS}
    logger.info(
        "API 키 상태: gemini=%s openai=%s anthropic=%s (값은 표시하지 않음)",
        statuses["gemini"],
        statuses["openai"],
        statuses["anthropic"],
    )

    if provider == "mock":
        return MockLLMClient(model=model or "mock")

    if provider == "offline":
        # API 키 없이 SNS 스타일 템플릿/어미 변환으로 생성. generator=offline (mock 아님).
        logger.warning(
            "⚠ --provider offline: 외부 LLM이 아닌 오프라인 템플릿 생성입니다. "
            "실제 상용 LLM·실제 SNS AI 글 성능을 대체하지 않습니다."
        )
        return OfflineLLMClient(model=model or "offline-cursor-v2")

    if provider == "file":
        raise RuntimeError(
            "--provider file 은 LLM 클라이언트를 쓰지 않습니다. "
            "build_dataset 에서 --extra_ai_files 로만 조립하세요."
        )

    if provider == "auto":
        available = detect_available_providers()
        if not available:
            raise RuntimeError(
                "사용 가능한 API 키가 없습니다.\n"
                "옵션:\n"
                "  1) GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY 설정 후 재실행\n"
                "  2) --provider offline  (API 없는 템플릿 생성, generator=offline)\n"
                "  3) --provider file --extra_ai_files <cursor가 만든 jsonl/csv>\n"
                "  4) --provider mock  (파이프라인 스모크 전용, 파일명에 _mock)\n"
                "⚠ mock/offline 결과는 실제 SNS AI 탐지 성능으로 해석하면 안 됩니다."
            )
        provider = available[0]
        logger.info("auto → provider=%s", provider)

    if provider == "gemini":
        return GeminiClient(model=model or "gemini-3.1-flash-lite")
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini")
    if provider == "anthropic":
        return AnthropicClient(model=model or "claude-3-5-haiku-latest")
    raise ValueError(
        f"알 수 없는 provider: {provider}. "
        f"auto|gemini|openai|anthropic|offline|file|mock 중 선택하세요."
    )


class RateLimitedPool:
    """간단한 동시성 + RPS 제한 실행기."""

    def __init__(self, concurrency: int = 4, rps: float = 2.0) -> None:
        self.concurrency = max(1, concurrency)
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self._last = 0.0
        self._lock = threading.Lock()

    def _throttle(self) -> None:
        with self._lock:
            now = time.time()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    def map(self, fn: Callable, items: Sequence[Any]) -> List[Any]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List[Any] = [None] * len(items)

        def _wrap(i: int, item: Any) -> None:
            self._throttle()
            results[i] = fn(item)

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futs = [ex.submit(_wrap, i, item) for i, item in enumerate(items)]
            for f in as_completed(futs):
                f.result()
        return results
