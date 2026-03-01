from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


@dataclass
class TFIDFConfig:
    text_fields: Tuple[str, ...] = ("summary", "text")
    label_field: str = "component"
    max_features: int = 50000
    ngram_range: Tuple[int, int] = (1, 2)
    C: float = 3.0
    penalty: str = "l2"
    max_iter: int = 300

    def __post_init__(self) -> None:
        # Ensure text_fields is always a tuple of field names (not a single string).
        if isinstance(self.text_fields, str):
            self.text_fields = (self.text_fields,)
        else:
            self.text_fields = tuple(self.text_fields)


def _load_texts_and_labels(
    path: Path,
    text_fields: Iterable[str],
    label_field: str,
) -> Tuple[List[str], List[str]]:
    df = pd.read_json(path, lines=True)

    if isinstance(text_fields, str):
        text_fields = (text_fields,)
    else:
        text_fields = tuple(text_fields)

    missing = [c for c in text_fields + (label_field,) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    texts = df[list(text_fields)].fillna("").agg(" ".join, axis=1).tolist()
    labels = df[label_field].astype(str).tolist()
    return texts, labels


def train_tfidf_classifier(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    metrics_path: Path,
    config: TFIDFConfig,
) -> Dict[str, Any]:
    """Allena TF-IDF + LogisticRegression, salva modello+metriche e restituisce le metriche."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Load train and test splits
    train_texts, train_labels = _load_texts_and_labels(
        train_path, config.text_fields, config.label_field
    )
    test_texts, test_labels = _load_texts_and_labels(
        test_path, config.text_fields, config.label_field
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Classifier
    clf = LogisticRegression(
        C=config.C,
        penalty=config.penalty,
        max_iter=config.max_iter,
        n_jobs=-1,
    )
    clf.fit(X_train, train_labels)

    preds = clf.predict(X_test)

    acc = accuracy_score(test_labels, preds)
    f1_macro = f1_score(test_labels, preds, average="macro")
    f1_micro = f1_score(test_labels, preds, average="micro")

    print(f"[TFIDF] Accuracy:  {acc:.4f}")
    print(f"[TFIDF] F1 macro: {f1_macro:.4f}")
    print(f"[TFIDF] F1 micro: {f1_micro:.4f}")

    report = classification_report(test_labels, preds, output_dict=True)

    # Save metrics (in reports/)
    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "classification_report": report,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save model + vectorizer (in models/baseline_tfidf/)
    joblib.dump(clf, output_dir / "tfidf_logreg.pkl")
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.pkl")

    print(f"[TFIDF] Model + vectorizer saved to {output_dir}")
    print(f"[TFIDF] Metrics saved to {metrics_path}")

    return {
        "metrics": metrics,
        "model_dir": str(output_dir),
        "metrics_path": str(metrics_path),
    }
