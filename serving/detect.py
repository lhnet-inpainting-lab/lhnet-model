"""개인정보 객체 자동 탐지.

- 얼굴: YuNet(OpenCV 공식 경량 ONNX, 232KB) — CPU 수십 ms
- 번호판: OpenCV 내장 캐스케이드(베타) — 추가 의존성 없음
텍스트(주민번호 등) 탐지는 OCR 모델 추가 예정.
"""

import os
import threading

import cv2
import numpy as np

MODEL = os.path.join(os.path.dirname(__file__), "face_detection_yunet_2023mar.onnx")
SCORE_THRESHOLD = 0.6
_lock = threading.Lock()
_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(MODEL, "", (320, 320), SCORE_THRESHOLD)
    return _detector


def detect_faces(image_bgr: np.ndarray) -> list[dict]:
    """얼굴 목록 반환: [{type, box:[x,y,w,h], score}] (원본 픽셀 좌표)."""
    h, w = image_bgr.shape[:2]
    # 아주 큰 이미지는 축소해 탐지 후 좌표 복원
    scale = min(1.0, 1280 / max(h, w))
    img = cv2.resize(image_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else image_bgr

    with _lock:  # FaceDetectorYN은 스레드 안전하지 않음
        det = _get_detector()
        det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = det.detect(img)

    out = []
    if faces is not None:
        for f in faces:
            x, y, bw, bh = f[:4] / scale
            out.append({
                "type": "face",
                "box": [max(0, int(x)), max(0, int(y)), int(bw), int(bh)],
                "score": round(float(f[-1]), 3),
            })
    return out


_plate_cascade = None


def _get_plate_cascade():
    global _plate_cascade
    if _plate_cascade is None:
        _plate_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        )
    return _plate_cascade


def detect_plates(image_bgr: np.ndarray) -> list[dict]:
    """차량 번호판 후보 목록(베타). 캐스케이드 기반이라 score는 고정값."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    with _lock:
        plates = _get_plate_cascade().detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(50, 16))
    return [
        {"type": "plate", "box": [int(x), int(y), int(w), int(h)], "score": 0.5}
        for x, y, w, h in (plates if plates is not None else [])
    ]


def detect_all(image_bgr: np.ndarray) -> list[dict]:
    """얼굴 + 번호판 통합 탐지."""
    return detect_faces(image_bgr) + detect_plates(image_bgr)


def boxes_to_mask(shape_hw: tuple, boxes: list, margin: float = 0.25) -> np.ndarray:
    """탐지 박스 목록을 인페인팅 마스크(흰색=제거 영역)로 변환. 박스를 margin만큼 키운 타원."""
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in boxes:
        cx, cy = x + bw / 2, y + bh / 2
        ax, ay = bw / 2 * (1 + margin), bh / 2 * (1 + margin)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 255, -1)
    return mask
