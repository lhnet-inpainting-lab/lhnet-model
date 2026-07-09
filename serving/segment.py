"""클릭 좌표 기반 객체 분리.

GrabCut을 클릭 지점 주변 사각형으로 초기화해 전경 마스크를 뽑고,
클릭 지점이 포함된 연결 성분만 남긴다. 세그먼트 전용 모델 없이
OpenCV만으로 동작하도록 만든 경량 구현.
"""

import cv2
import numpy as np

WORK_SIZE = 480      # GrabCut 처리 해상도 (속도)
RECT_RATIO = 0.42    # 클릭 지점 기준 초기 사각형 크기 (짧은 변 대비)
FALLBACK_RADIUS = 0.04  # 분리 실패 시 클릭 지점 원형 마스크 반경


def grabcut_at(image_bgr: np.ndarray, nx: float, ny: float) -> np.ndarray:
    """정규화 좌표 (nx, ny)의 객체 마스크(흰색=객체)를 원본 해상도로 반환."""
    h, w = image_bgr.shape[:2]
    scale = min(1.0, WORK_SIZE / max(h, w))
    small = cv2.resize(image_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else image_bgr
    sh, sw = small.shape[:2]
    px, py = int(nx * sw), int(ny * sh)

    side = int(min(sh, sw) * RECT_RATIO)
    x0, y0 = max(0, px - side // 2), max(0, py - side // 2)
    x1, y1 = min(sw - 1, px + side // 2), min(sh - 1, py + side // 2)

    mask = np.zeros((sh, sw), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, (x0, y0, x1 - x0, y1 - y0), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        fg = np.zeros((sh, sw), np.uint8)

    # 클릭 지점이 포함된 연결 성분만 유지
    n, labels = cv2.connectedComponents(fg)
    label = labels[min(py, sh - 1), min(px, sw - 1)]
    if label == 0:
        # 분리 실패 → 클릭 지점 주변 원형 폴백
        fg = np.zeros((sh, sw), np.uint8)
        cv2.circle(fg, (px, py), max(6, int(min(sh, sw) * FALLBACK_RADIUS)), 255, -1)
    else:
        fg = np.where(labels == label, 255, 0).astype(np.uint8)

    # 경계를 살짝 부드럽게 + 팽창해 인페인팅 시 잔상 방지
    fg = cv2.dilate(fg, np.ones((5, 5), np.uint8), iterations=1)
    fg = cv2.GaussianBlur(fg, (5, 5), 0)
    fg = (fg > 127).astype(np.uint8) * 255

    if scale < 1.0:
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
    return fg
