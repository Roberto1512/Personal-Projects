from __future__ import annotations

import time
from typing import Callable

from prometheus_client import Counter, Histogram

# Quante richieste riceviamo (con endpoint e status code)
HTTP_REQUESTS_TOTAL = Counter(
    "naplace_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

# Latenza richiesta (tempo totale in secondi)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "naplace_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

# Quante predizioni facciamo per modello
MODEL_PREDICTIONS_TOTAL = Counter(
    "naplace_model_predictions_total",
    "Total number of predictions per model",
    ["model_name"],
)

# Latenza inferenza per modello
MODEL_INFERENCE_DURATION_SECONDS = Histogram(
    "naplace_model_inference_duration_seconds",
    "Model inference duration in seconds",
    ["model_name"],
)


def time_it_seconds() -> Callable[[], float]:
    """
    Ritorna una funzione che, se chiamata dopo, restituisce i secondi trascorsi.
    Pattern comodo per misurare durata senza ripetere codice.
    """
    start = time.perf_counter()

    def _elapsed() -> float:
        return time.perf_counter() - start

    return _elapsed
