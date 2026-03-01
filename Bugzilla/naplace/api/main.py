from __future__ import annotations

import time
from typing import List

from fastapi import FastAPI, HTTPException, Request, Response
from gradio.routes import mount_gradio_app
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from naplace.api.models import PredictionRequest, PredictionResponse
from naplace.observability.metrics import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
)
from naplace.ui.gradio import build_gradio_app

app = FastAPI(
    title="Naplace Bug Component Classifier API",
    version="0.1.0",
    description=(
        "Web API per la classificazione dei bug report "
        "nei componenti di Mozilla Bugzilla, basata su modelli GRU, LSTM e SetFit."
    ),
)

gradio_app = build_gradio_app()
app = mount_gradio_app(app, gradio_app, path="/label")


@app.middleware("http")
async def prometheus_http_middleware(request: Request, call_next):
    method = request.method
    path = request.url.path

    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    # Metriche Prometheus
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(duration)
    HTTP_REQUESTS_TOTAL.labels(
        method=method, path=path, status_code=str(response.status_code)
    ).inc()

    return response


@app.get("/metrics")
def metrics():
    """
    Endpoint scrape per Prometheus.
    Restituisce tutte le metriche registrate dal prometheus_client.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict/gru", response_model=PredictionResponse)
def gru_predict(request: PredictionRequest):
    from naplace.api.inference import predict_gru

    texts: List[str] = [item.text for item in request.texts]
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided for prediction.")
    predictions = predict_gru(texts)
    return PredictionResponse(model_name="gru", predictions=predictions)


@app.post("/predict/tfidf", response_model=PredictionResponse)
def tfidf_predict(request: PredictionRequest):
    from naplace.api.inference import predict_tfidf

    texts: List[str] = [item.text for item in request.texts]
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided for prediction.")
    predictions = predict_tfidf(texts)
    return PredictionResponse(model_name="tfidf", predictions=predictions)


@app.post("/predict/lstm", response_model=PredictionResponse)
def lstm_predict(request: PredictionRequest):
    from naplace.api.inference import predict_lstm

    texts: List[str] = [item.text for item in request.texts]
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided for prediction.")
    predictions = predict_lstm(texts)
    return PredictionResponse(model_name="lstm", predictions=predictions)


@app.post("/predict/setfit", response_model=PredictionResponse)
def setfit_predict(request: PredictionRequest):
    from naplace.api.inference import predict_setfit

    texts: List[str] = [item.text for item in request.texts]
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided for prediction.")
    predictions = predict_setfit(texts)
    return PredictionResponse(model_name="setfit", predictions=predictions)
