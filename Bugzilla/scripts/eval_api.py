from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import joblib

from naplace.api.inference import predict_gru, predict_lstm, predict_setfit
from naplace.api.models import PredictionItem

# Percorso del file di test
TEST_PATH = Path("data/interim/test.jsonl")

# Quanti esempi campionare
N_SAMPLES = 50

# Seed per riproducibilità
SEED = 42

# Percorsi modelli TF-IDF
BASELINE_TFIDF_DIR = Path("models/baseline_tfidf")
TFIDF_SGD_PATH = Path("models/tfidf_sgd.joblib")


def _truncate(text: str, max_len: int = 160) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def extract_text_and_label(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Estrae testo e label dal JSONL "interim".
    - Testo: summary + text (se diversi), così è più informativo e simile a un bug report.
    - Label: component
    """
    summary = (obj.get("summary") or "").strip()
    text = (obj.get("text") or "").strip()

    if not summary and not text:
        return None, None

    if summary and text and summary != text:
        combined = f"{summary}\n\n{text}"
    else:
        combined = summary or text

    label = obj.get("component")
    label = str(label).strip() if label else None

    return combined, label


def load_dataset(path: Path) -> List[Tuple[str, Optional[str]]]:
    """Carica il dataset JSONL e restituisce una lista di (text, label)."""
    examples: List[Tuple[str, Optional[str]]] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text, label = extract_text_and_label(obj)
            if text is None:
                continue
            examples.append((text, label))

    return examples


def _load_baseline_tfidf_predictor(model_dir: Path):
    """
    Prova a caricare il baseline TF-IDF da una directory.
    Supporta:
    - Pipeline salvata in un .joblib/.pkl
    - Coppia vectorizer + model salvati separatamente (nomi vari)
    """
    if not model_dir.exists():
        return None

    candidates = [
        model_dir / "pipeline.joblib",
        model_dir / "pipeline.pkl",
        model_dir / "model.joblib",
        model_dir / "model.pkl",
        model_dir / "tfidf_pipeline.joblib",
        model_dir / "tfidf_model.joblib",
        model_dir / "baseline_tfidf.joblib",
    ]

    # 1) tenta candidati noti
    for p in candidates:
        if p.is_file():
            return joblib.load(p)

    # 2) fallback: se c'è UN solo file joblib/pkl nella cartella, carica quello
    files = sorted(list(model_dir.glob("*.joblib")) + list(model_dir.glob("*.pkl")))
    if len(files) == 1:
        return joblib.load(files[0])

    # 3) fallback: prova a caricare vectorizer + classifier separati
    vec_candidates = [
        model_dir / "vectorizer.joblib",
        model_dir / "vectorizer.pkl",
        model_dir / "tfidf_vectorizer.joblib",
        model_dir / "tfidf_vectorizer.pkl",
    ]
    clf_candidates = [
        model_dir / "classifier.joblib",
        model_dir / "classifier.pkl",
        model_dir / "tfidf_classifier.joblib",
        model_dir / "tfidf_classifier.pkl",
        model_dir / "logreg.joblib",
        model_dir / "logreg.pkl",
    ]

    vec = next((p for p in vec_candidates if p.is_file()), None)
    clf = next((p for p in clf_candidates if p.is_file()), None)

    if vec and clf:
        vectorizer = joblib.load(vec)
        classifier = joblib.load(clf)
        return (vectorizer, classifier)

    return None


def _predict_sklearn(
    predictor: Any, texts: List[str]
) -> List[PredictionItem]:
    """
    predictor può essere:
    - una sklearn Pipeline con predict/predict_proba
    - una tupla (vectorizer, classifier)
    """
    if not texts:
        return []

    if isinstance(predictor, tuple) and len(predictor) == 2:
        vectorizer, classifier = predictor
        x = vectorizer.transform(texts)
        y_pred = classifier.predict(x)

        proba = None
        if hasattr(classifier, "predict_proba"):
            try:
                proba = classifier.predict_proba(x)
            except Exception:
                proba = None

        out: List[PredictionItem] = []
        for i, t in enumerate(texts):
            p = float(max(proba[i])) if proba is not None else None
            out.append(
                PredictionItem(
                    input_text=t,
                    predicted_label=str(y_pred[i]),
                    probability=p,
                )
            )
        return out

    # Pipeline / estimator unico
    y_pred = predictor.predict(texts)

    proba = None
    if hasattr(predictor, "predict_proba"):
        try:
            proba = predictor.predict_proba(texts)
        except Exception:
            proba = None

    out2: List[PredictionItem] = []
    for i, t in enumerate(texts):
        p = float(max(proba[i])) if proba is not None else None
        out2.append(
            PredictionItem(
                input_text=t,
                predicted_label=str(y_pred[i]),
                probability=p,
            )
        )
    return out2


def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"File non trovato: {TEST_PATH}")

    all_examples = load_dataset(TEST_PATH)
    if not all_examples:
        raise RuntimeError("Nessun esempio valido trovato nel dataset di test.")

    print(f"Trovati {len(all_examples)} esempi nel test set.")
    n = min(N_SAMPLES, len(all_examples))

    random.seed(SEED)
    sampled = random.sample(all_examples, k=n)

    texts = [t for t, _ in sampled]
    true_labels = [lbl for _, lbl in sampled]

    print(f"\nEseguo prediction su {n} esempi casuali...\n")

    # --- Modelli API (già integrati) ---
    gru_preds = predict_gru(texts)
    lstm_preds = predict_lstm(texts)
    setfit_preds = predict_setfit(texts)

    # --- TF-IDF baseline ---
    baseline_predictor = _load_baseline_tfidf_predictor(BASELINE_TFIDF_DIR)
    baseline_preds: Optional[List[PredictionItem]] = None
    if baseline_predictor is None:
        print(f"[WARN] Baseline TF-IDF non trovato in {BASELINE_TFIDF_DIR} (salto).")
    else:
        baseline_preds = _predict_sklearn(baseline_predictor, texts)

    # --- TF-IDF SGD ---
    tfidf_sgd_preds: Optional[List[PredictionItem]] = None
    if not TFIDF_SGD_PATH.is_file():
        print(f"[WARN] TF-IDF SGD non trovato in {TFIDF_SGD_PATH} (salto).")
    else:
        sgd_pipeline = joblib.load(TFIDF_SGD_PATH)
        tfidf_sgd_preds = _predict_sklearn(sgd_pipeline, texts)

    # Helper per check correttezza
    def is_correct(true_label: Optional[str], pred: PredictionItem) -> bool:
        return true_label is not None and pred.predicted_label == str(true_label)

    # Stats
    totals = {"gru": 0, "lstm": 0, "setfit": 0, "tfidf": 0, "tfidf_sgd": 0}

    for idx in range(n):
        text = texts[idx]
        true_label = true_labels[idx]

        gru_pred = gru_preds[idx]
        lstm_pred = lstm_preds[idx]
        setfit_pred = setfit_preds[idx]

        gru_ok = is_correct(true_label, gru_pred)
        lstm_ok = is_correct(true_label, lstm_pred)
        setfit_ok = is_correct(true_label, setfit_pred)

        totals["gru"] += int(gru_ok)
        totals["lstm"] += int(lstm_ok)
        totals["setfit"] += int(setfit_ok)

        tfidf_line = ""
        tfidf_sgd_line = ""

        if baseline_preds is not None:
            tfidf_pred = baseline_preds[idx]
            tfidf_ok = is_correct(true_label, tfidf_pred)
            totals["tfidf"] += int(tfidf_ok)
            p = tfidf_pred.probability
            tfidf_line = (
                f"TF-IDF PRED:  {tfidf_pred.predicted_label} "
                f"(p={'-' if p is None else f'{p:.3f}'}) "
                f"{'✔' if tfidf_ok else '✘' if true_label is not None else ''}"
            )

        if tfidf_sgd_preds is not None:
            sgd_pred = tfidf_sgd_preds[idx]
            sgd_ok = is_correct(true_label, sgd_pred)
            totals["tfidf_sgd"] += int(sgd_ok)
            p = sgd_pred.probability
            tfidf_sgd_line = (
                f"TFIDF-SGD:   {sgd_pred.predicted_label} "
                f"(p={'-' if p is None else f'{p:.3f}'}) "
                f"{'✔' if sgd_ok else '✘' if true_label is not None else ''}"
            )

        print("=" * 90)
        print(f"ESEMPIO #{idx + 1}")
        print(f"TESTO:        {_truncate(text)}")
        print(f"LABEL VERA:   {true_label}")
        print(
            f"GRU PRED:     {gru_pred.predicted_label} "
            f"(p={'-' if gru_pred.probability is None else f'{gru_pred.probability:.3f}'}) "
            f"{'✔' if gru_ok else '✘' if true_label is not None else ''}"
        )
        print(
            f"LSTM PRED:    {lstm_pred.predicted_label} "
            f"(p={'-' if lstm_pred.probability is None else f'{lstm_pred.probability:.3f}'}) "
            f"{'✔' if lstm_ok else '✘' if true_label is not None else ''}"
        )
        print(
            f"SETFIT PRED:  {setfit_pred.predicted_label} "
            f"(p={'-' if setfit_pred.probability is None else f'{setfit_pred.probability:.3f}'}) "
            f"{'✔' if setfit_ok else '✘' if true_label is not None else ''}"
        )
        if tfidf_line:
            print(tfidf_line)
        if tfidf_sgd_line:
            print(tfidf_sgd_line)
        print()

    print("=" * 90)
    print("RIEPILOGO (accuracy su campione)")
    print(f"GRU:       {totals['gru']}/{n} = {totals['gru'] / n:.3f}")
    print(f"LSTM:      {totals['lstm']}/{n} = {totals['lstm'] / n:.3f}")
    print(f"SetFit:    {totals['setfit']}/{n} = {totals['setfit'] / n:.3f}")
    if baseline_preds is not None:
        print(f"TF-IDF:    {totals['tfidf']}/{n} = {totals['tfidf'] / n:.3f}")
    if tfidf_sgd_preds is not None:
        print(f"TFIDF-SGD: {totals['tfidf_sgd']}/{n} = {totals['tfidf_sgd'] / n:.3f}")
    print("Fine valutazione qualitativa.")


if __name__ == "__main__":
    main()
