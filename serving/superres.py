"""초해상도 확대 — FSRCNN(OpenCV dnn_superres)으로 2배·4배.

FSRCNN은 40KB급 경량 모델이라 CPU에서도 큰 사진을 1초 안에 처리한다.
모델 파일(FSRCNN_x2.pb, FSRCNN_x4.pb)은 레포에 커밋돼 있다.
"""

import os
import threading

import cv2
import numpy as np

SCALES = (2, 4)
MAX_OUTPUT_SIDE = 4096  # 결과 긴 변 상한 — 메모리·전송 보호

_lock = threading.Lock()
_models: dict[int, "cv2.dnn_superres.DnnSuperResImpl"] = {}


def _get_model(scale: int):
    if scale not in _models:
        path = os.path.join(os.path.dirname(__file__), f"FSRCNN_x{scale}.pb")
        model = cv2.dnn_superres.DnnSuperResImpl_create()
        model.readModel(path)
        model.setModel("fsrcnn", scale)
        _models[scale] = model
    return _models[scale]


def upscale(image_bgr: np.ndarray, scale: int) -> np.ndarray:
    """scale배로 키운 이미지를 반환. 지원하지 않는 배율·과대 출력은 ValueError."""
    if scale not in SCALES:
        raise ValueError(f"지원하는 배율은 {SCALES} 입니다.")
    h, w = image_bgr.shape[:2]
    if max(h, w) * scale > MAX_OUTPUT_SIDE:
        raise ValueError(f"결과가 너무 큽니다 — 긴 변 {MAX_OUTPUT_SIDE // scale}px 이하 사진만 {scale}배로 키울 수 있어요.")

    with _lock:
        return _get_model(scale).upsample(image_bgr)
