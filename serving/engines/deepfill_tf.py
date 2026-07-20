"""직접 학습한 DeepFillv2 연구모델(TensorFlow) 엔진."""

import os

import cv2
import numpy as np

from engines.base import SERVING_DIR, module_available

CHECKPOINT_DIR = os.environ.get("DEEPFILL_CKPT_DIR", "./training_checkpoints")
IMG_SIZE = 256  # 학습 시 입력 해상도


def _has_checkpoint() -> bool:
    if not os.path.isdir(CHECKPOINT_DIR):
        return False
    return any(f.startswith("ckpt") for f in os.listdir(CHECKPOINT_DIR))


class DeepFillEngine:
    """DeepFillv2 GeneratorMultiColumn 기반 추론 (TensorFlow 필요).

    프로젝트 초기에 직접 학습한 '연구 모델'. training_checkpoints/의 체크포인트와
    net.py(GeneratorMultiColumn)가 있어야 로드된다. 시연 배포본에는 대용량
    체크포인트를 올리지 않으므로 보통 '사용 불가'로 표시된다.
    """

    id = "deepfillv2-tf"
    name = id
    label = "DeepFillv2 · 연구모델"
    version = "v1 · 직접 학습(TF)"
    desc = "프로젝트 초기에 직접 학습한 GeneratorMultiColumn 체크포인트 모델"

    @staticmethod
    def available() -> bool:
        has_net = os.path.exists(os.path.join(SERVING_DIR, "net.py"))
        return module_available("tensorflow") and has_net and _has_checkpoint()

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
