"""인페인팅 추론 엔진.

DeepFillv2 학습 체크포인트(training_checkpoints/)가 있으면 GeneratorMultiColumn으로 추론하고,
없으면 OpenCV Telea 인페인팅으로 폴백한다. 웹 시연 환경에는 대용량 체크포인트를
올리지 않으므로 기본은 폴백 모드다.
"""

import os

import cv2
import numpy as np

CHECKPOINT_DIR = os.environ.get("DEEPFILL_CKPT_DIR", "./training_checkpoints")
IMG_SIZE = 256  # 학습 시 입력 해상도


class DeepFillEngine:
    """DeepFillv2 GeneratorMultiColumn 기반 추론 (TensorFlow 필요)."""

    name = "deepfillv2"

    def __init__(self):
        import tensorflow as tf  # 체크포인트가 있을 때만 임포트

        from net import GeneratorMultiColumn

        self._tf = tf
        self.generator = GeneratorMultiColumn()
        checkpoint = tf.train.Checkpoint(generator=self.generator)
        latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if latest is None:
            raise FileNotFoundError(f"체크포인트 없음: {CHECKPOINT_DIR}")
        checkpoint.restore(latest).expect_partial()

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        tf = self._tf
        h, w = image_bgr.shape[:2]

        img = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
        msk = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_norm = (img_rgb / 127.5) - 1.0
        msk_norm = (msk > 127).astype(np.float32)[..., np.newaxis]

        masked = img_norm * (1.0 - msk_norm)
        x = tf.concat(
            [masked, tf.ones_like(msk_norm) * msk_norm], axis=-1
        )[tf.newaxis, ...]
        _, fine = self.generator(x, training=False)

        out = fine[0].numpy()
        out = img_norm * (1.0 - msk_norm) + out * msk_norm  # 마스크 영역만 합성
        out = ((out + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(out, (w, h))


class OpenCVEngine:
    """cv2.inpaint(Telea) 폴백. 체크포인트 없이도 시연 흐름을 확인할 수 있다."""

    name = "opencv-telea"

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        binary_mask = (mask > 127).astype(np.uint8) * 255
        return cv2.inpaint(image_bgr, binary_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)


def load_engine():
    """가능한 엔진을 우선순위대로 로드한다."""
    try:
        return DeepFillEngine()
    except Exception:
        return OpenCVEngine()
