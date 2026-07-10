"""얼굴 복원 — 흐리거나 저화질인 얼굴을 GFPGAN 1.4(ONNX)로 선명하게.

파이프라인: YuNet 5점 랜드마크 → FFHQ 512 템플릿으로 유사 변환 정렬 →
GFPGAN 추론 → 역변환으로 원본 위치에 페더링 합성.

모델 파일(gfpgan_1.4.onnx, 340MB)은 커밋하지 않는다 — 없으면 다운로드 안내 에러를 낸다:
https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx
"""

import os
import threading

import cv2
import numpy as np

from detect import detect_face_landmarks

MODEL = os.path.join(os.path.dirname(__file__), "gfpgan_1.4.onnx")
CROP = 512
# FFHQ 512 표준 5점 템플릿 (왼눈, 오른눈, 코끝, 왼입꼬리, 오른입꼬리 — 화면 기준)
FFHQ_TEMPLATE = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.19360],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
], np.float32)
FEATHER_PX = 24  # 합성 경계 페더링

_lock = threading.Lock()
_session = None


def _get_session():
    global _session
    if _session is None:
        if not os.path.exists(MODEL):
            raise FileNotFoundError(
                "gfpgan_1.4.onnx 가 없습니다. model/serving 에 내려받아 주세요: "
                "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx"
            )
        import onnxruntime as ort

        _session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    return _session


def _run_gfpgan(crop_bgr: np.ndarray) -> np.ndarray:
    """정렬된 512 얼굴 크롭을 복원해 반환 (BGR uint8)."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[np.newaxis]

    with _lock:
        out = _get_session().run(None, {"input": tensor})[0][0]

    out = ((out.clip(-1, 1) + 1) / 2 * 255).astype(np.uint8).transpose(1, 2, 0)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _paste_mask() -> np.ndarray:
    """크롭 가장자리를 부드럽게 죽인 합성용 마스크 (float32 0~1)."""
    mask = np.ones((CROP, CROP), np.float32)
    mask[:FEATHER_PX, :] = 0
    mask[-FEATHER_PX:, :] = 0
    mask[:, :FEATHER_PX] = 0
    mask[:, -FEATHER_PX:] = 0
    return cv2.GaussianBlur(mask, (FEATHER_PX * 2 + 1, FEATHER_PX * 2 + 1), 0)


def restore_faces(image_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """사진 속 모든 얼굴을 복원해 (결과, 복원한 얼굴 수)를 반환."""
    faces = detect_face_landmarks(image_bgr)
    if not faces:
        return image_bgr, 0

    h, w = image_bgr.shape[:2]
    result = image_bgr.astype(np.float32)
    paste_mask = _paste_mask()

    for face in faces:
        src = np.array(face["landmarks"], np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(src, FFHQ_TEMPLATE, method=cv2.LMEDS)
        if matrix is None:
            continue
        crop = cv2.warpAffine(image_bgr, matrix, (CROP, CROP), borderMode=cv2.BORDER_REFLECT)
        restored = _run_gfpgan(crop)

        inverse = cv2.invertAffineTransform(matrix)
        back = cv2.warpAffine(restored, inverse, (w, h)).astype(np.float32)
        weight = cv2.warpAffine(paste_mask, inverse, (w, h))[..., np.newaxis]
        result = back * weight + result * (1 - weight)

    return result.clip(0, 255).astype(np.uint8), len(faces)
