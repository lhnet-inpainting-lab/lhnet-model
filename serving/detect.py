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


def detect_face_landmarks(image_bgr: np.ndarray) -> list[dict]:
    """얼굴 박스와 5점 랜드마크 목록 반환 (원본 픽셀 좌표).

    랜드마크 순서는 YuNet 그대로: 오른눈, 왼눈, 코끝, 오른입꼬리, 왼입꼬리
    (화면 기준 왼쪽이 먼저) — 얼굴 복원의 FFHQ 템플릿 정렬에 쓴다.
    """
    h, w = image_bgr.shape[:2]
    scale = min(1.0, 1280 / max(h, w))
    img = cv2.resize(image_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else image_bgr

    with _lock:
        det = _get_detector()
        det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = det.detect(img)

    out = []
    if faces is not None:
        for f in faces:
            x, y, bw, bh = f[:4] / scale
            out.append({
                "box": [max(0, int(x)), max(0, int(y)), int(bw), int(bh)],
                "landmarks": (f[4:14].reshape(5, 2) / scale).tolist(),
                "score": round(float(f[-1]), 3),
            })
    return out


PLATE_MODEL = os.path.join(os.path.dirname(__file__), "plate_yolo11s.onnx")
PLATE_INPUT = 640
PLATE_SCORE_THRESHOLD = 0.35
PLATE_NMS_THRESHOLD = 0.45

_plate_session = None
_plate_cascade = None


def _get_plate_session():
    """YOLO11 번호판 모델(있으면). 모델 파일이 없으면 None — 캐스케이드로 폴백."""
    global _plate_session
    if _plate_session is None and os.path.exists(PLATE_MODEL):
        import onnxruntime as ort

        _plate_session = ort.InferenceSession(PLATE_MODEL, providers=["CPUExecutionProvider"])
    return _plate_session


def _get_plate_cascade():
    global _plate_cascade
    if _plate_cascade is None:
        _plate_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        )
    return _plate_cascade


def _detect_plates_yolo(session, image_bgr: np.ndarray) -> list[dict]:
    """YOLO11 번호판 탐지: 레터박스 640 → 추론 → NMS → 원본 좌표 복원."""
    h, w = image_bgr.shape[:2]
    scale = min(PLATE_INPUT / w, PLATE_INPUT / h)
    nw, nh = round(w * scale), round(h * scale)
    pad_x, pad_y = (PLATE_INPUT - nw) / 2, (PLATE_INPUT - nh) / 2

    canvas = np.full((PLATE_INPUT, PLATE_INPUT, 3), 114, np.uint8)
    canvas[int(pad_y):int(pad_y) + nh, int(pad_x):int(pad_x) + nw] = cv2.resize(image_bgr, (nw, nh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = rgb.transpose(2, 0, 1)[np.newaxis]

    with _lock:
        pred = session.run(None, {"images": tensor})[0][0]  # (5, N): cx, cy, w, h, conf

    keep = pred[4] >= PLATE_SCORE_THRESHOLD
    boxes, scores = [], []
    for cx, cy, bw, bh, conf in pred[:, keep].T:
        x = (cx - bw / 2 - pad_x) / scale
        y = (cy - bh / 2 - pad_y) / scale
        boxes.append([int(x), int(y), int(bw / scale), int(bh / scale)])
        scores.append(float(conf))

    picked = cv2.dnn.NMSBoxes(boxes, scores, PLATE_SCORE_THRESHOLD, PLATE_NMS_THRESHOLD)
    out = []
    for i in np.array(picked).flatten():
        x, y, bw, bh = boxes[i]
        out.append({
            "type": "plate",
            "box": [max(0, x), max(0, y), min(bw, w), min(bh, h)],
            "score": round(scores[i], 3),
        })
    return out


def detect_plates(image_bgr: np.ndarray) -> list[dict]:
    """차량 번호판 목록. YOLO11 전용 모델을 쓰고, 모델이 없으면 캐스케이드(베타)로 폴백."""
    session = _get_plate_session()
    if session is not None:
        return _detect_plates_yolo(session, image_bgr)

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
