"""엔진 공통 상수·헬퍼."""

import importlib.util
import os

SERVING_DIR = os.path.dirname(os.path.dirname(__file__))


def module_available(module: str) -> bool:
    """무거운 임포트 없이 모듈 설치 여부만 확인."""
    return importlib.util.find_spec(module) is not None
