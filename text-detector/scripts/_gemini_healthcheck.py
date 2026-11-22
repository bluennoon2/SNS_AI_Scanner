"""Gemini 연결 헬스체크 (google-genai 전용). API 키 원문은 절대 출력하지 않음."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)

from data_build.llm_clients import GeminiClient, mask_key_status


def _scrub(msg: str) -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if key and key in msg:
        msg = msg.replace(key, "[REDACTED]")
    return msg[:240]


def _key_prefix_bucket() -> str:
    k = os.environ.get("GEMINI_API_KEY", "")
    if not k:
        return "missing"
    if k.startswith("AQ."):
        return "AQ."
    if k.startswith("AIza"):
        return "AIza"
    return "other"


def main() -> int:
    print("env_exists:", (ROOT / ".env").exists())
    print("GEMINI_API_KEY status:", mask_key_status("gemini"))
    print("key_prefix_bucket:", _key_prefix_bucket(), "(원문 미출력)")
    if mask_key_status("gemini") != "감지됨":
        print("FAIL: .env 에 GEMINI_API_KEY=... 가 없습니다.")
        return 1

    try:
        from google import genai
    except ImportError:
        print("FAIL: google-genai 미설치. pip install -U google-genai")
        return 1

    # 최신 Flash 우선 (사용자 지정) + list_models로 보강
    candidates = [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-flash-latest",
    ]

    listed: list[str] = []
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            name = name.replace("models/", "")
            low = name.lower()
            if "flash" in low and "embed" not in low:
                listed.append(name)
        # 3.x flash 우선 정렬
        listed_sorted = sorted(
            set(listed),
            key=lambda n: (
                0 if re.search(r"3\.[58]", n) else 1 if "2.5" in n else 2,
                n,
            ),
        )
        print("flash_models_listed:", listed_sorted[:25])
        for m in reversed(listed_sorted):
            if m not in candidates:
                candidates.insert(0, m)
    except Exception as exc:
        print("list_models_failed:", type(exc).__name__, _scrub(str(exc)))

    # 중복 제거, 순서 유지
    seen = set()
    ordered = []
    for m in candidates:
        if m not in seen:
            seen.add(m)
            ordered.append(m)

    prompt = "한국어로 '연결 확인' 네 글자만 출력하세요."
    for model in ordered:
        print(f"try_model: {model}")
        try:
            g = GeminiClient(model=model)
            out = g.generate(prompt, temperature=0.2)
            preview = (out or "").replace("\n", " ")[:40]
            print("OK model=", model, "resp_len=", len(out or ""), "preview=", preview)
            print("SELECTED_MODEL=", model)
            return 0
        except Exception as exc:
            print(
                "FAIL model=",
                model,
                "err=",
                type(exc).__name__,
                _scrub(str(exc)),
            )
    print("FAIL: 후보 모델 모두 실패")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
