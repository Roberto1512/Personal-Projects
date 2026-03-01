from __future__ import annotations

import json
from pathlib import Path
from typing import List

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from setfit import SetFitModel

from naplace import config
from naplace.api.models import PredictionItem
from naplace.observability.metrics import (
    MODEL_INFERENCE_DURATION_SECONDS,
    MODEL_PREDICTIONS_TOTAL,
    time_it_seconds,
)

# Root e cartella modelli
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = getattr(config, "MODELS_DIR", ROOT_DIR / "models")

# --- Percorsi GRU / LSTM ---
GRU_MODEL_PATH = MODELS_DIR / "gru.h5"
GRU_TOKENIZER_PATH = MODELS_DIR / "gru_tokenizer.pkl"
GRU_LABELS_PATH = MODELS_DIR / "gru_label_classes.npy"

LSTM_MODEL_PATH = MODELS_DIR / "lstm.h5"
LSTM_TOKENIZER_PATH = MODELS_DIR / "lstm_tokenizer.pkl"
LSTM_LABELS_PATH = MODELS_DIR / "lstm_label_classes.npy"

# --- Percorsi SetFit ---
SETFIT_DIR = MODELS_DIR / "setfit_component"
SETFIT_MAPPING_PATH = SETFIT_DIR / "label_mapping.json"

# --- Percorso TF-IDF (sklearn) ---
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_sgd.joblib"
_tfidf_model = None


# Cache in memoria (lazy load)
_gru_model = None
_gru_tokenizer = None
_gru_label_classes = None

_lstm_model = None
_lstm_tokenizer = None
_lstm_label_classes = None

_setfit_model = None
_setfit_id2label = None


def _load_pickle(path: Path):
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def _record_metrics(model_name: str, elapsed_seconds: float, n_predictions: int) -> None:
    if n_predictions <= 0:
        return
    MODEL_PREDICTIONS_TOTAL.labels(model_name=model_name).inc(n_predictions)
    MODEL_INFERENCE_DURATION_SECONDS.labels(model_name=model_name).observe(elapsed_seconds)


# =========================
#    GRU / LSTM HELPERS
# =========================


def _lazy_load_gru():
    global _gru_model, _gru_tokenizer, _gru_label_classes
    if _gru_model is None:
        if not GRU_MODEL_PATH.is_file():
            raise SystemExit(f"❌ Modello GRU non trovato in {GRU_MODEL_PATH}")
        if not GRU_TOKENIZER_PATH.is_file():
            raise SystemExit(f"❌ Tokenizer GRU non trovato in {GRU_TOKENIZER_PATH}")
        if not GRU_LABELS_PATH.is_file():
            raise SystemExit(f"❌ Label classes GRU non trovate in {GRU_LABELS_PATH}")

        _gru_model = load_model(GRU_MODEL_PATH)
        _gru_tokenizer = _load_pickle(GRU_TOKENIZER_PATH)
        _gru_label_classes = np.load(GRU_LABELS_PATH, allow_pickle=True)

    return _gru_model, _gru_tokenizer, _gru_label_classes


def _lazy_load_lstm():
    global _lstm_model, _lstm_tokenizer, _lstm_label_classes
    if _lstm_model is None:
        if not LSTM_MODEL_PATH.is_file():
            raise SystemExit(f"❌ Modello LSTM non trovato in {LSTM_MODEL_PATH}")
        if not LSTM_TOKENIZER_PATH.is_file():
            raise SystemExit(f"❌ Tokenizer LSTM non trovato in {LSTM_TOKENIZER_PATH}")
        if not LSTM_LABELS_PATH.is_file():
            raise SystemExit(f"❌ Label classes LSTM non trovate in {LSTM_LABELS_PATH}")

        _lstm_model = load_model(LSTM_MODEL_PATH)
        _lstm_tokenizer = _load_pickle(LSTM_TOKENIZER_PATH)
        _lstm_label_classes = np.load(LSTM_LABELS_PATH, allow_pickle=True)

    return _lstm_model, _lstm_tokenizer, _lstm_label_classes


def _preprocess_texts(tokenizer, texts: List[str], max_len: int = 200):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")


def _predict_keras(
    model,
    tokenizer,
    label_classes: np.ndarray,
    texts: List[str],
    max_len: int = 200,
) -> List[PredictionItem]:
    x = _preprocess_texts(tokenizer, texts, max_len=max_len)
    proba = model.predict(x)
    indices = proba.argmax(axis=1)
    labels = label_classes[indices]

    results: List[PredictionItem] = []
    for text, label, p in zip(texts, labels, proba.max(axis=1)):
        results.append(
            PredictionItem(
                input_text=text,
                predicted_label=str(label),
                probability=float(p),
            )
        )
    return results


def predict_gru(texts: List[str]) -> List[PredictionItem]:
    if not texts:
        return []
    model, tokenizer, label_classes = _lazy_load_gru()
    elapsed = time_it_seconds()
    results = _predict_keras(model, tokenizer, label_classes, texts)
    _record_metrics("gru", elapsed(), len(results))
    return results


def predict_lstm(texts: List[str]) -> List[PredictionItem]:
    if not texts:
        return []
    model, tokenizer, label_classes = _lazy_load_lstm()
    elapsed = time_it_seconds()
    results = _predict_keras(model, tokenizer, label_classes, texts)
    _record_metrics("lstm", elapsed(), len(results))
    return results


# =========================
#        SETFIT
# =========================


def _lazy_load_setfit():
    """Carica SetFitModel + mapping id->label."""
    global _setfit_model, _setfit_id2label

    if _setfit_model is None:
        if not SETFIT_DIR.is_dir():
            raise SystemExit(f"❌ Directory SetFit non trovata: {SETFIT_DIR}")

        if not SETFIT_MAPPING_PATH.is_file():
            raise SystemExit(
                f"❌ File label_mapping.json non trovato: {SETFIT_MAPPING_PATH}\n"
                "Assicurati di aver eseguito scripts/train_setfit.py."
            )

        _setfit_model = SetFitModel.from_pretrained(str(SETFIT_DIR))

        with SETFIT_MAPPING_PATH.open(encoding="utf-8") as f:
            mapping = json.load(f)

        raw_id2label = mapping.get("id2label", {})
        # chiavi salvate come stringhe → convertiamo a int
        _setfit_id2label = {int(k): v for k, v in raw_id2label.items()}

    return _setfit_model, _setfit_id2label


def _lazy_load_tfidf():
    global _tfidf_model
    if _tfidf_model is None:
        if not TFIDF_MODEL_PATH.is_file():
            raise SystemExit(f"❌ Modello TF-IDF non trovato in {TFIDF_MODEL_PATH}")
        import joblib

        _tfidf_model = joblib.load(TFIDF_MODEL_PATH)
    return _tfidf_model


def predict_tfidf(texts: List[str]) -> List[PredictionItem]:
    if not texts:
        return []

    model = _lazy_load_tfidf()
    elapsed = time_it_seconds()

    pred_labels = model.predict(texts)

    try:
        proba = model.predict_proba(texts)
        pmax = proba.max(axis=1)
    except Exception:
        pmax = [None] * len(texts)

    results: List[PredictionItem] = []
    for t, lab, p in zip(texts, pred_labels, pmax):
        results.append(
            PredictionItem(
                input_text=t,
                predicted_label=str(lab),
                probability=(float(p) if p is not None else None),
            )
        )

    _record_metrics("tfidf", elapsed(), len(results))
    return results


def predict_setfit(texts: List[str]) -> List[PredictionItem]:
    """
    Predice la componente usando il modello SetFit addestrato
    (livello definito da config.label_field, di solito 'component').
    """
    if not texts:
        return []

    model, id2label = _lazy_load_setfit()

    elapsed = time_it_seconds()

    # id predetti
    pred_ids = model.predict(texts)

    # proviamo a ottenere anche le probabilità (se disponibile)
    try:
        proba = model.predict_proba(texts)
        # SetFit può restituire torch.Tensor: convertiamo a numpy per evitare crash con np.max
        if hasattr(proba, "detach"):
            proba = proba.detach().cpu().numpy()
        else:
            proba = np.asarray(proba)
    except Exception:
        proba = None

    results: List[PredictionItem] = []
    for i, text in enumerate(texts):
        label_id = int(pred_ids[i])
        label = id2label.get(label_id, str(label_id))

        p: float | None
        if proba is not None:
            # proba[i] è il vettore di probabilità su tutte le classi
            p = float(np.max(proba[i]))
        else:
            p = None

        results.append(
            PredictionItem(
                input_text=text,
                predicted_label=label,
                probability=p,
            )
        )

    _record_metrics("setfit", elapsed(), len(results))

    return results
