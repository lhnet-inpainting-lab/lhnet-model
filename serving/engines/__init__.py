"""인페인팅 추론 엔진 패키지.

엔진별 모듈(telea·lama·deepfill_torch·deepfill_tf)과 이를 카탈로그로 묶는
EngineManager로 구성된다. 서비스 코드는 EngineManager만 쓰면 된다.
"""

from engines.manager import ENGINE_CLASSES, EngineManager


def load_engine():
    """하위 호환: 단일 엔진을 원하는 기존 호출부를 위해 기본 엔진을 반환한다."""
    return EngineManager().get()
