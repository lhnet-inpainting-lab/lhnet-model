"""인페인팅 추론 API 서버.

Spring Boot 백엔드가 호출하는 내부 서비스. 원본 이미지와 마스크(흰색=복원 영역)를
받아 인페인팅된 PNG를 반환한다.

실행: uvicorn app:app --port 8000  (model/serving 디렉터리에서)
"""

import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from detect import boxes_to_mask, detect_all, detect_faces, detect_plates
from engines import EngineManager
from enhance import restore_faces
from ocr import detect_text
from people import segment_people
from segment import grabcut_at
from superres import upscale

app = FastAPI(title="DeepFillv2 Inference Service")
engines = EngineManager()


def _decode(data: bytes, flags: int) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if img is None:
        raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다.")
    return img


@app.get("/health")
def health():
    return {"status": "ok", "engine": engines.default_id()}


@app.get("/engines")
def list_engines():
    """사용자 모델 선택기용 카탈로그. 각 모델의 사용 가능 여부·기본값을 함께 내려준다."""
    return {"default": engines.default_id(), "engines": engines.catalog()}


@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    engine: str = Form(None),
):
    """마스크 영역을 인페인팅한다. engine으로 사용할 모델을 고를 수 있다(없으면 기본값)."""
    img = _decode(await image.read(), cv2.IMREAD_COLOR)
    msk = _decode(await mask.read(), cv2.IMREAD_GRAYSCALE)

    if msk.shape[:2] != img.shape[:2]:
        msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    eng = engines.get(engine)
    started = time.perf_counter()
    result = eng.inpaint(img, msk)
    elapsed_ms = (time.perf_counter() - started) * 1000

    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="결과 인코딩에 실패했습니다.")
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={"X-Engine": eng.name, "X-Elapsed-Ms": f"{elapsed_ms:.0f}"},
    )


DETECTORS = {"face": detect_faces, "plate": detect_plates, "text": detect_text}


@app.post("/detect")
async def detect(image: UploadFile = File(...), targets: str = Form("face,plate")):
    """사진 속 개인정보(얼굴·번호판·텍스트)를 탐지해 좌표 목록을 반환한다.

    targets: 쉼표로 구분한 탐지 대상 (face | plate | text). 기본값은 기존 동작과 같은 face,plate.
    """
    img = _decode(await image.read(), cv2.IMREAD_COLOR)
    detections = []
    for target in targets.split(","):
        detector = DETECTORS.get(target.strip())
        if detector is None:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 탐지 대상입니다: {target.strip()}")
        detections += detector(img)
    return {"width": img.shape[1], "height": img.shape[0], "detections": detections}


@app.post("/redact")
async def redact(image: UploadFile = File(...), engine: str = Form(None)):
    """원콜 비식별화: 얼굴·번호판 탐지 → 마스크 → 인페인팅 복원까지 한 번에 처리한다."""
    img = _decode(await image.read(), cv2.IMREAD_COLOR)
    detections = detect_all(img)

    eng = engines.get(engine)
    if detections:
        mask = boxes_to_mask(img.shape[:2], [d["box"] for d in detections])
        result = eng.inpaint(img, mask)
    else:
        result = img

    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="결과 인코딩에 실패했습니다.")
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={"X-Redacted-Count": str(len(detections)), "X-Engine": eng.name},
    )


@app.post("/upscale")
async def upscale_endpoint(image: UploadFile = File(...), scale: int = Form(2)):
    """FSRCNN 초해상도로 scale배(2·4) 확대한 PNG를 반환한다."""
    img = _decode(await image.read(), cv2.IMREAD_COLOR)
    try:
        result = upscale(img, scale)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="결과 인코딩에 실패했습니다.")
    return Response(content=encoded.tobytes(), media_type="image/png")


@app.post("/restore-faces")
async def restore_faces_endpoint(image: UploadFile = File(...)):
    """사진 속 모든 얼굴을 GFPGAN으로 복원한 PNG를 반환한다. X-Restored-Count 헤더에 얼굴 수."""
    img = _decode(await image.read(), cv2.IMREAD_COLOR)
    result, count = restore_faces(img)

    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="결과 인코딩에 실패했습니다.")
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={"X-Restored-Count": str(count)},
    )


@app.post("/segment-people")
async def segment_people_endpoint(image: UploadFile = File(...)):
    """사람 실루엣 마스크 PNG(255=사람)를 반환한다. X-Person-Coverage 헤더에 화면 대비 비율."""
    img = _decode(await image.read(), cv2.IMREAD_COLOR)
    mask, coverage = segment_people(img)

    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise HTTPException(status_code=500, detail="마스크 인코딩에 실패했습니다.")
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={"X-Person-Coverage": f"{coverage:.4f}"},
    )


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
