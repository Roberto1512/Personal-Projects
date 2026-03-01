# naplace/modeling/predict.py

import pickle

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from naplace.config import MODELS

# Lunghezza max delle sequenze: usa lo stesso valore che hai usato in training.
# Se in training hai usato un altro valore (es. 200), cambialo qui per coerenza.
MAX_LEN = 200


def _load_lstm_artifacts():
    """Carica modello, tokenizer e classi salvati per l'LSTM."""
    model_path = MODELS / "lstm.h5"
    tok_path = MODELS / "lstm_tokenizer.pkl"
    classes_path = MODELS / "lstm_label_classes.npy"

    if not model_path.exists():
        raise SystemExit(f"❌ Modello LSTM non trovato in {model_path}")
    if not tok_path.exists():
        raise SystemExit(f"❌ Tokenizer LSTM non trovato in {tok_path}")
    if not classes_path.exists():
        raise SystemExit(f"❌ Label classes LSTM non trovate in {classes_path}")

    model = load_model(model_path)

    with tok_path.open("rb") as f:
        tokenizer = pickle.load(f)

    classes = np.load(classes_path, allow_pickle=True)

    return model, tokenizer, classes


def predict_component_lstm(texts: list[str]) -> list[str]:
    """
    Predice la componente per una lista di descrizioni di bug usando l'LSTM salvato.
    `texts` è una lista di stringhe (summary+description).
    Ritorna una lista di label (component).
    """
    if not texts:
        return []

    model, tokenizer, classes = _load_lstm_artifacts()

    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    probs = model.predict(X, verbose=0)
    idx = probs.argmax(axis=1)

    labels = [classes[i] for i in idx]
    return labels
