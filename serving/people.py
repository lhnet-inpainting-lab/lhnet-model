"""사람 영역 세그멘테이션.

torchvision DeepLabV3(ResNet-50, VOC 학습)의 person 클래스로 사람 실루엣 마스크를 만든다.
얼굴 박스가 아니라 몸 전체를 잡아야 하는 '사람 전체 지우기'에 쓴다.
"""

import threading

import cv2
import numpy as np

PERSON_CLASS = 15  # PASCAL VOC 클래스 인덱스
SCORE_THRESHOLD = 0.5
INFER_SIZE = 520  # 긴 변 기준 추론 해상도 — CPU 속도와 경계 품질의 절충
MASK_DILATE_PX = 5  # 경계가 몸 안쪽으로 파고들지 않게 살짝 부풀린다

_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    if _model is None:
        import torch
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        model.eval()
        torch.set_grad_enabled(False)
        _model = model
    return _model


def segment_people(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """사람 마스크(uint8, 255=사람)와 화면 대비 비율(0~1)을 반환."""
    import torch

    h, w = image_bgr.shape[:2]
    scale = INFER_SIZE / max(h, w)
    small = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))))

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    tensor = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)).unsqueeze(0)

    with _lock:
        out = _get_model()(tensor)["out"][0]

    prob = out.softmax(0)[PERSON_CLASS].numpy()
    mask_small = (prob > SCORE_THRESHOLD).astype(np.uint8) * 255
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    if MASK_DILATE_PX > 0:
        kernel = np.ones((MASK_DILATE_PX * 2 + 1, MASK_DILATE_PX * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, kernel)

    coverage = float((mask > 0).mean())
    return mask, coverage
