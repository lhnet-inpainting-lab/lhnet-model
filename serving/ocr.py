"""사진 속 텍스트 개인정보 탐지 (베타).

EasyOCR(영문·숫자)로 글자를 읽고, 정규식으로 개인정보 패턴만 골라낸다.
한글 문장 인식은 대상이 아니다 — 개인정보의 핵심인 숫자·기호 패턴에 집중한다.
반환 형태는 detect.py 와 동일한 [{type, box:[x,y,w,h], score}] 에 label(분류명)이 추가된다.
"""

import re
import threading

import cv2
import numpy as np

# 구체적인(긴) 패턴을 먼저 검사한다 — 전화번호·주민번호도 계좌 패턴에 걸리기 때문.
PII_PATTERNS = [
    ("카드번호", re.compile(r"(?:\d{4}[- ]){3}\d{4}")),
    ("주민등록번호", re.compile(r"\d{6}-[0-4]\d{6}")),
    ("전화번호", re.compile(r"01[016789][- ]?\d{3,4}[- ]?\d{4}|0\d{1,2}-\d{3,4}-\d{4}")),
    ("이메일", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("계좌번호", re.compile(r"\d{2,6}-\d{2,6}-\d{2,8}")),
]

_lock = threading.Lock()
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr  # 무거운 import — 첫 사용 시점으로 미룬다

        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def classify_pii(text: str) -> str | None:
    """인식된 문자열이 개인정보 패턴이면 분류명을, 아니면 None을 반환."""
    for label, pattern in PII_PATTERNS:
        if pattern.search(text):
            return label
    return None


def detect_text(image_bgr: np.ndarray) -> list[dict]:
    """개인정보 텍스트 목록 반환: [{type:'text', label, box:[x,y,w,h], score}] (원본 픽셀 좌표)."""
    h, w = image_bgr.shape[:2]
    # 아주 큰 이미지는 축소해 인식 후 좌표 복원 (detect_faces 와 같은 전략)
    scale = min(1.0, 1600 / max(h, w))
    img = cv2.resize(image_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else image_bgr

    reader = _get_reader()
    with _lock:  # EasyOCR Reader는 스레드 안전이 보장되지 않는다
        results = reader.readtext(img)

    out = []
    for quad, text, conf in results:
        label = classify_pii(text)
        if label is None:
            continue
        xs = [p[0] / scale for p in quad]
        ys = [p[1] / scale for p in quad]
        x, y = max(0, int(min(xs))), max(0, int(min(ys)))
        out.append({
            "type": "text",
            "label": label,
            "box": [x, y, int(max(xs)) - x, int(max(ys)) - y],
            "score": round(float(conf), 3),
        })
    return out
