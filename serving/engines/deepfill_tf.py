"""직접 학습한 DeepFillv2 연구모델(TensorFlow) 엔진."""

import os

import cv2
import numpy as np

from engines.base import SERVING_DIR, module_available

CHECKPOINT_DIR = os.environ.get(
    "DEEPFILL_CKPT_DIR", os.path.join(SERVING_DIR, "training_checkpoints")
)
IMG_SIZE = 256  # 학습 시 입력 해상도
STAGE3_INDEX = 2  # GeneratorMultiColumn 반환값(stage1·2·3·flow×2) 중 최종 결과


def _has_checkpoint() -> bool:
    if not os.path.isdir(CHECKPOINT_DIR):
        return False
    return any(f.startswith("ckpt") for f in os.listdir(CHECKPOINT_DIR))


class DeepFillEngine:
    """2024 캡스톤에서 직접 학습한 GeneratorMultiColumn 연구모델 추론.

    네트워크 정의는 research_net.py(이력에서 복원·벤더링)에 있고, 가중치는
    training_checkpoints/의 tf.train.Checkpoint다. 체크포인트는 대용량이라
    레포에 없으므로, TensorFlow가 설치돼 있고 체크포인트를 넣었을 때만 활성화된다.
    """

    id = "deepfillv2-tf"
    name = id
    label = "DeepFillv2 · 연구모델"
    version = "v1 · 직접 학습(TF)"
    desc = "프로젝트 초기에 직접 학습한 GeneratorMultiColumn 체크포인트 모델"

    @staticmethod
    def available() -> bool:
        return module_available("tensorflow") and _has_checkpoint()

    def __init__(self):
        import tensorflow as tf  # 체크포인트가 있을 때만 임포트

        from research_net import GeneratorMultiColumn

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
        img_norm = (img_rgb / 127.5) - 1.0  # [-1, 1]
        msk_norm = (msk > 127).astype(np.float32)[..., np.newaxis]  # 1=구멍

        # 학습 코드(generate_images)와 같은 관례: 구멍을 지운 이미지 + 마스크를 입력
        image_batch = tf.convert_to_tensor(img_norm[np.newaxis])
        mask_batch = tf.convert_to_tensor(msk_norm[np.newaxis])
        masked_batch = image_batch * (1.0 - mask_batch)

        outputs = self.generator(masked_batch, mask_batch, training=False)
        stage3 = outputs[STAGE3_INDEX]

        composited = masked_batch + stage3 * mask_batch  # 마스크 영역만 생성 결과로 합성
        out = ((composited[0].numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(out, (w, h))
