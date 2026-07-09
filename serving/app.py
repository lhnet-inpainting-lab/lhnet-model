"""인페인팅 추론 API 서버.

Spring Boot 백엔드가 호출하는 내부 서비스. 원본 이미지와 마스크(흰색=복원 영역)를
받아 인페인팅된 PNG를 반환한다.

실행: uvicorn app:app --port 8000  (model/serving 디렉터리에서)
"""

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from engine import load_engine
from segment import grabcut_at

app = FastAPI(title="DeepFillv2 Inference Service")
engine = load_engine()


def _decode(data: bytes, flags: int) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if img is None:
        raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다.")
    return img


@app.get("/health")
def health():
    return {"status": "ok", "engine": engine.name}


@app.post("/inpaint")
async def inpaint(image: UploadFile = File(...), mask: UploadFile = File(...)):
    img = _decode(await image.read(), cv2.IMREAD_COLOR)
    msk = _decode(await mask.read(), cv2.IMREAD_GRAYSCALE)

    if msk.shape[:2] != img.shape[:2]:
        msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    result = engine.inpaint(img, msk)

    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="결과 인코딩에 실패했습니다.")
    return Response(content=encoded.tobytes(), media_type="image/png")


@app.post("/segment")
async def segment(image: UploadFile = File(...), x: float = Form(...), y: float = Form(...)):
    """클릭 지점(0~1 정규화 좌표)의 객체를 GrabCut으로 분리해 마스크 PNG를 반환한다."""
    img = _decode(await image.read(), cv2.IMREAD_COLOR)
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise HTTPException(status_code=400, detail="좌표는 0~1 범위여야 합니다.")

    mask = grabcut_at(img, x, y)

    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise HTTPException(status_code=500, detail="마스크 인코딩에 실패했습니다.")
    return Response(content=encoded.tobytes(), media_type="image/png")
