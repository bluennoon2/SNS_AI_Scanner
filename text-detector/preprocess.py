# 이 파일의 역할:
# SNS 텍스트에 대한 "최소 정제"를 수행한다.
# 이모지·해시태그·구어체·맞춤법 오류 등은 AI 판별 신호이므로 보존한다.

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# 제어문자: \n(0x0A), \t(0x09)는 유지하고 나머지 C0/C1 제어문자 제거
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

# 제로폭 문자: U+200B~U+200D, U+FEFF
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")

# 연속 공백/탭 → 단일 공백 (줄바꿈은 별도 처리)
_MULTI_SPACE_RE = re.compile(r"[^\S\n]+")

# 3줄 이상 연속 줄바꿈 → 2줄
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# 같은 문자 반복 축약용 (옵션)
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}", re.DOTALL)


# 줄바꿈 보존용 특수 토큰 (옵션). vocab을 수정하지 않고 문자열 치환만 한다.
NEWLINE_PLACEHOLDER = "[NL]"


def clean_text(
    text: str,
    collapse_repeats: bool = False,
    max_repeat: Optional[int] = None,
    newline_token: bool = False,
) -> str:
    """SNS 텍스트를 최소 정제한다.

    수행 규칙 (각각 예시):
        1) NFC 정규화
           입력: "가" (조합형) → 출력: "가" (완성형 NFC)
           ※ NFKC는 호환 분해로 문자 변형이 커서 사용하지 않는다.
        2) 제어문자 제거 (\n, \t 유지)
           입력: "안녕\\x00하세요" → 출력: "안녕하세요"
        3) 제로폭 문자 제거 (U+200B~U+200D, U+FEFF)
           입력: "안\\u200b녕" → 출력: "안녕"
        4) 연속 공백/탭 → 단일 공백, 3줄+ 줄바꿈 → 2줄
           입력: "안녕   하세요\\n\\n\\n내일" → 출력: "안녕 하세요\\n\\n내일"
        5) 앞뒤 공백 trim
           입력: "  안녕  " → 출력: "안녕"
        6) (옵션) 같은 문자 반복 축약 — collapse_repeats=True, max_repeat=N
           입력: "ㅋㅋㅋㅋㅋㅋ", max_repeat=3 → 출력: "ㅋㅋㅋ"
        7) (옵션) newline_token=True이면 줄바꿈을 [NL]로 치환
           입력: "안녕\\n내일" → 출력: "안녕 [NL] 내일"
           ※ 학습/평가/추론에 동일하게 적용해야 한다. vocab 수정은 하지 않는다.

    유지(삭제·치환 금지, newline_token OFF 시):
        이모지, #해시태그, @멘션, URL, 줄바꿈,
        자음/모음 반복(ㅋㅋㅋ, ㅠㅠ), 구어체, 맞춤법 오류, !!! / ...

    Args:
        text: 원본 문자열. None/비문자형은 빈 문자열로 취급.
        collapse_repeats: True이면 반복 문자 축약을 켠다.
        max_repeat: 동일 문자 최대 반복 횟수. collapse_repeats가 True일 때 사용.
        newline_token: True이면 줄바꿈을 NEWLINE_PLACEHOLDER로 치환.

    Returns:
        정제된 문자열.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1) NFC 정규화 (NFKC 금지)
    text = unicodedata.normalize("NFC", text)

    # 2) 제어문자 제거 (\n, \t 유지)
    text = _CONTROL_RE.sub("", text)

    # 3) 제로폭 문자 제거
    text = _ZERO_WIDTH_RE.sub("", text)

    # 4) 연속 공백/탭 → 단일 공백, 3줄+ → 2줄
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    # 5) trim
    text = text.strip()

    # 6) 옵션: 반복 문자 축약
    if collapse_repeats and max_repeat is not None and max_repeat > 0:
        limit = max_repeat

        def _collapse(match: re.Match) -> str:
            ch = match.group(1)
            return ch * limit

        pattern = re.compile(rf"(.)\1{{{limit},}}", re.DOTALL)
        text = pattern.sub(_collapse, text)

    # 7) 옵션: 줄바꿈 → [NL] (기본 OFF)
    if newline_token:
        text = text.replace("\n", f" {NEWLINE_PLACEHOLDER} ")
        text = _MULTI_SPACE_RE.sub(" ", text).strip()

    return text


if __name__ == "__main__":
    # 자체 테스트 — 각 규칙별 입출력 확인
    cases = [
        ("NFC", "가", clean_text("가")),
        ("제어문자", "안녕하세요", clean_text("안녕\x00하세요")),
        ("제로폭", "안녕", clean_text("안\u200b녕")),
        ("공백축약", "안녕 하세요\n\n내일", clean_text("안녕   하세요\n\n\n내일")),
        ("trim", "안녕", clean_text("  안녕  ")),
        (
            "반복축약",
            "ㅋㅋㅋ",
            clean_text("ㅋㅋㅋㅋㅋㅋ", collapse_repeats=True, max_repeat=3),
        ),
        ("이모지유지", "오늘 날씨 미쳤다 😂 #일상", clean_text("오늘 날씨 미쳤다 😂 #일상")),
        ("멘션URL유지", "안녕 @user https://x.com/a !!!", clean_text("안녕 @user https://x.com/a !!!")),
        ("구어체유지", "ㅋㅋㅋ ㅠㅠ ㅎㅎ 진짜ㅋㅋ", clean_text("ㅋㅋㅋ ㅠㅠ ㅎㅎ 진짜ㅋㅋ")),
    ]

    print("=== preprocess.py 자체 테스트 ===")
    all_ok = True
    for name, expected, actual in cases:
        ok = actual == expected
        all_ok = all_ok and ok
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: expected={expected!r} actual={actual!r}")

    if not all_ok:
        raise SystemExit(1)
    print("모든 테스트 통과")
