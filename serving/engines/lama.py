"""사전학습 LaMa(big-lama torchscript) 엔진."""

import os

import cv2
import numpy as np

from engines.base import SERVING_DIR, module_available

PAD_MULTIPLE = 8  # LaMa 다운/업샘플 정렬을 위한 입력 해상도 배수
MASK_DILATE_KERNEL = 7  # 경계가 남지 않도록 마스크를 살짝 팽창


class LamaEngine:
    """사전학습 LaMa 기반 추론.

    학습 데이터 없이도 SOTA 수준의 객체 제거/복원 품질을 낸다. Cleanup.pictures·IOPaint가
    쓰는 그 엔진이며, big-lama.pt(약 205MB)가 있고 torch가 설치돼 있을 때만 로드된다.
    """

    id = "lama"
    name = id
    label = "LaMa · 고품질"
    version = "big-lama"
    desc = "대형 마스크·복잡한 배경에 강한 최상 품질 엔진 (권장 기본값)"
    _MODEL = os.environ.get("LAMA_MODEL", os.path.join(SERVING_DIR, "big-lama.pt"))

    @staticmethod
    def available() -> bool:
        return module_available("torch") and os.path.exists(LamaEngine._MODEL)

    def __init__(self):
        import torch

        if not os.path.exists(self._MODEL):
            raise FileNotFoundError(f"LaMa 모델 없음: {self._MODEL}")
        self._torch = torch
        self.model = torch.jit.load(self._MODEL, map_location="cpu")
        self.model.eval()

    def _pad(self, x):
        _, _, h, w = x.shape
        pad_w = (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE
        pad_h = (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE
        return self._torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = image_bgr.shape[:2]

        binary = (mask > 127).astype(np.uint8) * 255
        kernel = np.ones((MASK_DILATE_KERNEL, MASK_DILATE_KERNEL), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mask_tensor = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0).float() / 255.0
        mask_tensor = (mask_tensor >= 0.5).float()

        with torch.inference_mode():
            out = self.model(self._pad(image_tensor), self._pad(mask_tensor))
        result = (out[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()[:h, :w]
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # 마스크 영역만 결과로 합성 → 나머지 원본 픽셀 보존
        hole = binary[..., None] > 0
        return np.where(hole, result_bgr, image_bgr)
