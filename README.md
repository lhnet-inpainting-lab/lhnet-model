# LHNet Model Serving

FastAPI 추론 서비스 — 인페인팅·탐지·세그먼트.

## 구성
- `serving/` — API 서버 (`app.py`), 엔진(`engine.py`), 탐지(`detect.py`, YuNet), 세그먼트(`segment.py`, GrabCut)
- `masks/` — 학습용 마스크 생성 스크립트 (원형·사각·한글·스크리블 등)
- `train/` — 학습 유틸리티

## 엔진 우선순위
DeepFillv2 체크포인트(`DEEPFILL_CKPT_DIR`) → 사전학습 LaMa(`big-lama.pt`) → OpenCV Telea 폴백

## 실행
```bash
pip install -r serving/requirements.txt
# big-lama.pt(약 205MB)는 requirements.txt 안내의 링크에서 serving/에 다운로드
cd serving && uvicorn app:app --port 8000
```

## 엔드포인트
`POST /inpaint` · `POST /detect` · `POST /segment` · `POST /redact` · `GET /health`
