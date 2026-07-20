"""PyTorch DeepFillv2(원본 아키텍처) 엔진."""

import os

import cv2
import numpy as np

from engines.base import SERVING_DIR, module_available

PAD_MULTIPLE = 8  # 입력 해상도를 8의 배수로 맞춰 다운/업샘플 정렬


class DeepFillTorchEngine:
    """PyTorch DeepFillv2(gated conv + contextual attention) 추론.

    프로젝트 이름값을 하는 '실제 DeepFillv2' 엔진. TF 원본 가중치를 변환한
    states_tf_places2.pth(약 40MB)를 쓰며, 파일이 없으면 최초 로드 시 gdown으로
    자동 내려받는다. 네트워크 정의는 deepfill_net.py(벤더링)에 있다.
    """

    id = "deepfillv2-torch"
    name = id
    label = "DeepFillv2 · PyTorch"
    version = "places2 (pretrained)"
    desc = "gated conv + contextual attention 기반 실제 DeepFillv2 — 최초 사용 시 가중치 자동 다운로드"
    _WEIGHTS = os.environ.get(
        "DEEPFILL_TORCH_WEIGHTS", os.path.join(SERVING_DIR, "states_tf_places2.pth")
    )
    _GDRIVE_ID = "1tvdQRmkphJK7FYveNAKSMWC6K09hJoyt"  # Places2 변환 가중치

    @staticmethod
    def available() -> bool:
        # torch + 네트워크 정의가 있으면 가용(가중치는 없으면 최초 사용 시 자동 다운로드)
        has_net = os.path.exists(os.path.join(SERVING_DIR, "deepfill_net.py"))
        has_weights_or_gdown = (
            os.path.exists(DeepFillTorchEngine._WEIGHTS) or module_available("gdown")
        )
        return module_available("torch") and has_net and has_weights_or_gdown

    def __init__(self):
        import torch

        from deepfill_net import Generator

        self._torch = torch
        weights = self._ensure_weights()
        state = torch.load(weights, map_location="cpu")["G"]
        generator = Generator(cnum_in=5, cnum=48, return_flow=False)
        generator.load_state_dict(state, strict=True)
        generator.eval()
        self.generator = generator

    def _ensure_weights(self) -> str:
        if os.path.exists(self._WEIGHTS):
            return self._WEIGHTS
        import gdown

        url = f"https://drive.google.com/uc?id={self._GDRIVE_ID}"
        print(f"[engine] DeepFillv2 가중치 다운로드 중(약 40MB): {url}")
        gdown.download(url, self._WEIGHTS, quiet=True)
        if not os.path.exists(self._WEIGHTS):
            raise FileNotFoundError("DeepFillv2 가중치 다운로드 실패")
        return self._WEIGHTS

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        torch = self._torch
        F = torch.nn.functional
        h, w = image_bgr.shape[:2]

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        image_tensor = image_tensor * 2.0 - 1.0  # [-1, 1]
        binary = (mask > 127).astype(np.uint8)
        mask_tensor = torch.from_numpy(binary).float().unsqueeze(0).unsqueeze(0)  # 1=구멍

        # 8의 배수로 패딩(이미지 reflect, 마스크 0) → 추론 → 원 크기로 크롭
        pad_h = (PAD_MULTIPLE - h % PAD_MULTIPLE) % PAD_MULTIPLE
        pad_w = (PAD_MULTIPLE - w % PAD_MULTIPLE) % PAD_MULTIPLE
        image_padded = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")
        mask_padded = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        masked = image_padded * (1.0 - mask_padded)
        ones = torch.ones_like(mask_padded)
        x = torch.cat([masked, ones, ones * mask_padded], dim=1)  # 5채널 입력

        with torch.inference_mode():
            out = self.generator(x, mask_padded)
        stage2 = out[1] if isinstance(out, (tuple, list)) else out
        composited = image_padded * (1.0 - mask_padded) + stage2 * mask_padded
        composited = composited[:, :, :h, :w]

        result = ((composited[0].permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255)
        return cv2.cvtColor(result.byte().cpu().numpy(), cv2.COLOR_RGB2BGR)
