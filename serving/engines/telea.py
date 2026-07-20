"""OpenCV Telea 폴백 엔진."""

import cv2
import numpy as np

INPAINT_RADIUS = 7


class OpenCVEngine:
    """cv2.inpaint(Telea) 폴백. 딥러닝 없이도 시연 흐름을 확인할 수 있다."""

    id = "telea"
    name = id
    label = "OpenCV Telea"
    version = "고전 알고리즘"
    desc = "딥러닝 없이 주변 픽셀로 메우는 고전 방식 — 가장 가볍고 항상 동작"

    @staticmethod
    def available() -> bool:
        return True

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        binary_mask = (mask > 127).astype(np.uint8) * 255
        return cv2.inpaint(image_bgr, binary_mask, inpaintRadius=INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)
