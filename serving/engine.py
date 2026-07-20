"""인페인팅 추론 엔진.

DeepFillv2 학습 체크포인트(training_checkpoints/)가 있으면 GeneratorMultiColumn으로 추론하고,
없으면 OpenCV Telea 인페인팅으로 폴백한다. 웹 시연 환경에는 대용량 체크포인트를
올리지 않으므로 기본은 폴백 모드다.
"""

import importlib.util
import os
import threading

import cv2
import numpy as np

SERVING_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.environ.get("DEEPFILL_CKPT_DIR", "./training_checkpoints")
IMG_SIZE = 256  # 학습 시 입력 해상도


def _has(module: str) -> bool:
    """무거운 임포트 없이 모듈 설치 여부만 확인."""
    return importlib.util.find_spec(module) is not None


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
        idx_exists = os.path.isdir(CHECKPOINT_DIR) and any(
            f.startswith("ckpt") for f in os.listdir(CHECKPOINT_DIR)
        ) if os.path.isdir(CHECKPOINT_DIR) else False
        return _has("tensorflow") and os.path.exists(os.path.join(SERVING_DIR, "net.py")) and idx_exists

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
        return cv2.inpaint(image_bgr, binary_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)


class LamaEngine:
    """사전학습 LaMa(big-lama torchscript) 기반 추론.

    학습 데이터 없이도 SOTA 수준의 객체 제거/복원 품질을 낸다. Cleanup.pictures·IOPaint가
    쓰는 그 엔진이며, big-lama.pt(약 205MB)가 있고 torch가 설치돼 있을 때만 로드된다.
    """

    id = "lama"
    name = id
    label = "LaMa · 고품질"
    version = "big-lama"
    desc = "대형 마스크·복잡한 배경에 강한 최상 품질 엔진 (권장 기본값)"
    _MODEL = os.environ.get(
        "LAMA_MODEL", os.path.join(os.path.dirname(__file__), "big-lama.pt")
    )

    @staticmethod
    def available() -> bool:
        return _has("torch") and os.path.exists(LamaEngine._MODEL)

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
        "DEEPFILL_TORCH_WEIGHTS",
        os.path.join(os.path.dirname(__file__), "states_tf_places2.pth"),
    )
    _GDRIVE_ID = "1tvdQRmkphJK7FYveNAKSMWC6K09hJoyt"  # Places2 변환 가중치
    _GRID = 8  # 입력 해상도를 8의 배수로 맞춰 다운/업샘플 정렬

    @staticmethod
    def available() -> bool:
        # torch + 네트워크 정의가 있으면 가용(가중치는 없으면 최초 사용 시 자동 다운로드)
        has_net = os.path.exists(os.path.join(SERVING_DIR, "deepfill_net.py"))
        has_weights_or_gdown = os.path.exists(DeepFillTorchEngine._WEIGHTS) or _has("gdown")
        return _has("torch") and has_net and has_weights_or_gdown

    def __init__(self):
        import torch

        from deepfill_net import Generator

        self._torch = torch
        weights = self._ensure_weights()
        state = torch.load(weights, map_location="cpu")["G"]
        gen = Generator(cnum_in=5, cnum=48, return_flow=False)
        gen.load_state_dict(state, strict=True)
        gen.eval()
        self.generator = gen

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
        it = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        it = it * 2.0 - 1.0  # [-1, 1]
        binary = (mask > 127).astype(np.uint8)
        mt = torch.from_numpy(binary).float().unsqueeze(0).unsqueeze(0)  # 1=구멍

        # 8의 배수로 패딩(이미지 reflect, 마스크 0) → 추론 → 원 크기로 크롭
        ph = (self._GRID - h % self._GRID) % self._GRID
        pw = (self._GRID - w % self._GRID) % self._GRID
        it_p = F.pad(it, (0, pw, 0, ph), mode="reflect")
        mt_p = F.pad(mt, (0, pw, 0, ph), mode="constant", value=0.0)

        masked = it_p * (1.0 - mt_p)
        ones = torch.ones_like(mt_p)
        x = torch.cat([masked, ones, ones * mt_p], dim=1)  # 5채널 입력

        with torch.inference_mode():
            out = self.generator(x, mt_p)
        stage2 = out[1] if isinstance(out, (tuple, list)) else out
        comp = it_p * (1.0 - mt_p) + stage2 * mt_p
        comp = comp[:, :, :h, :w]

        res = ((comp[0].permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255).byte().cpu().numpy()
        return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)


# 사용자에게 노출할 모델 카탈로그. 위에서부터 기본값 우선순위(품질 순).
ENGINE_CLASSES = [LamaEngine, DeepFillTorchEngine, DeepFillEngine, OpenCVEngine]
ENGINE_BY_ID = {cls.id: cls for cls in ENGINE_CLASSES}
_ALIASES = {"deepfill": "deepfillv2-torch", "opencv-telea": "telea", "deepfillv2": "deepfillv2-tf"}


class EngineManager:
    """여러 인페인팅 모델을 카탈로그로 관리하고 요청마다 골라 쓴다.

    엔진은 무겁고(가중치 로드·다운로드) 재사용 가능하므로 처음 쓸 때 한 번만 만들어
    캐시한다. 사용자는 id로 원하는 모델을 선택하고, 지정이 없으면 기본값을 쓴다.
    """

    def __init__(self):
        self._cache: dict[str, object] = {}
        self._lock = threading.Lock()

    def resolve(self, engine_id: str | None) -> str | None:
        """별칭·대소문자를 정규화해 유효한 id로 바꾼다. 알 수 없으면 None."""
        if not engine_id:
            return None
        key = engine_id.strip().lower()
        key = _ALIASES.get(key, key)
        return key if key in ENGINE_BY_ID else None

    def default_id(self) -> str:
        """환경변수(INPAINT_ENGINE) 또는 가용한 것 중 품질 우선순위로 기본 모델 결정."""
        forced = self.resolve(os.environ.get("INPAINT_ENGINE"))
        if forced and ENGINE_BY_ID[forced].available():
            return forced
        for cls in ENGINE_CLASSES:
            if cls.available():
                return cls.id
        return OpenCVEngine.id

    def catalog(self) -> list[dict]:
        """프론트 모델 선택기에 내려줄 목록(사용 가능 여부 포함)."""
        default = self.default_id()
        items = []
        for cls in ENGINE_CLASSES:
            try:
                ok = cls.available()
            except Exception:  # noqa: BLE001
                ok = False
            items.append({
                "id": cls.id, "label": cls.label, "version": cls.version,
                "desc": cls.desc, "available": ok, "default": cls.id == default,
            })
        return items

    def get(self, engine_id: str | None = None):
        """요청한 모델(없으면 기본값)을 로드해 반환. 로드 실패 시 Telea로 폴백."""
        target = self.resolve(engine_id) or self.default_id()
        with self._lock:
            if target in self._cache:
                return self._cache[target]
            try:
                engine = ENGINE_BY_ID[target]()
                print(f"[engine] loaded: {engine.name}")
            except Exception as exc:  # noqa: BLE001
                print(f"[engine] {target} 로드 실패: {exc} → opencv-telea 폴백")
                engine = self._cache.get("telea") or OpenCVEngine()
            self._cache[target] = engine
            return engine


# 하위 호환: 단일 엔진을 원하는 기존 호출부를 위해 기본 엔진을 반환한다.
def load_engine():
    return EngineManager().get()
