"""인페인팅 모델 카탈로그·선택 관리."""

import os
import threading

from engines.deepfill_tf import DeepFillEngine
from engines.deepfill_torch import DeepFillTorchEngine
from engines.lama import LamaEngine
from engines.telea import OpenCVEngine

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
