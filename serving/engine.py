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


class LamaEngine:
    """사전학습 LaMa(big-lama torchscript) 기반 추론.

    학습 데이터 없이도 SOTA 수준의 객체 제거/복원 품질을 낸다. Cleanup.pictures·IOPaint가
    쓰는 그 엔진이며, big-lama.pt(약 205MB)가 있고 torch가 설치돼 있을 때만 로드된다.
    """

    name = "lama"
    _MODEL = os.environ.get(
        "LAMA_MODEL", os.path.join(os.path.dirname(__file__), "big-lama.pt")
    )

    def __init__(self):
        import torch

        if not os.path.exists(self._MODEL):
            raise FileNotFoundError(f"LaMa 모델 없음: {self._MODEL}")
        self._torch = torch
        self.model = torch.jit.load(self._MODEL, map_location="cpu")
        self.model.eval()

    def _pad(self, x, mod=8):
        _, _, h, w = x.shape
        return self._torch.nn.functional.pad(
            x, (0, (mod - w % mod) % mod, 0, (mod - h % mod) % mod), mode="reflect"
        )

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = image_bgr.shape[:2]

        # 경계가 남지 않도록 마스크를 살짝 팽창
        binary = (mask > 127).astype(np.uint8) * 255
        binary = cv2.dilate(binary, np.ones((7, 7), np.uint8), iterations=1)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        it = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mt = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0).float() / 255.0
        mt = (mt >= 0.5).float()

        with torch.inference_mode():
            out = self.model(self._pad(it), self._pad(mt))
        res = (out[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()[:h, :w]
        res_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

        # 마스크 영역만 결과로 합성 → 나머지 원본 픽셀 보존
        m3 = (binary[..., None] > 0)
        return np.where(m3, res_bgr, image_bgr)


def load_engine():
    """가능한 엔진을 우선순위대로 로드한다: DeepFill 체크포인트 → LaMa → Telea 폴백."""
    for factory in (DeepFillEngine, LamaEngine):
        try:
            engine = factory()
            print(f"[engine] loaded: {engine.name}")
            return engine
        except Exception as exc:  # noqa: BLE001
            print(f"[engine] {factory.__name__} 사용 불가: {exc}")
    print("[engine] loaded: opencv-telea (fallback)")
    return OpenCVEngine()
