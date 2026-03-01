from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


def read_jsonl_xy(path: Path) -> Tuple[List[str], List[str]]:
    X: List[str] = []
    y: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            summary = (obj.get("summary") or "").strip()
            text = (obj.get("text") or "").strip()
            label = (obj.get("component") or "").strip()

            if not label:
                continue

            combined = (summary + "\n\n" + text).strip()
            if not combined:
                continue

            X.append(combined)
            y.append(label)

    return X, y


def parse_args():
    p = argparse.ArgumentParser(
        description="Train TF-IDF + SGD(log_loss) model and save as joblib."
    )
    p.add_argument("--train-path", type=Path, default=Path("data/interim/train.jsonl"))
    p.add_argument("--test-path", type=Path, default=Path("data/interim/test.jsonl"))
    p.add_argument("--model-out", type=Path, default=Path("models/tfidf_sgd.joblib"))
    p.add_argument("--metrics-out", type=Path, default=Path("reports/metrics_tfidf_sgd.json"))
    p.add_argument("--max-features", type=int, default=200000)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--alpha", type=float, default=1e-5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    X_train, y_train = read_jsonl_xy(args.train_path)

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, args.ngram_max),
                    max_features=args.max_features,
                    min_df=args.min_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                    dtype=np.float32,
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=args.alpha,
                    penalty="l2",
                    max_iter=2000,
                    tol=1e-3,
                    n_jobs=-1,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    metrics: Dict[str, Any] = {
        "model": "tfidf_sgd_log_loss",
        "train_samples": len(X_train),
        "n_classes_train": len(set(y_train)),
        "params": {
            "max_features": args.max_features,
            "min_df": args.min_df,
            "ngram_range": [1, args.ngram_max],
            "alpha": args.alpha,
        },
    }

    if args.test_path.is_file():
        X_test, y_test = read_jsonl_xy(args.test_path)
        y_pred = pipeline.predict(X_test)

        metrics.update(
            {
                "test_samples": len(X_test),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
                "f1_micro": float(f1_score(y_test, y_pred, average="micro")),
            }
        )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[OK] Saved model: {args.model_out}")
    print(f"[OK] Saved metrics: {args.metrics_out}")
    if "f1_macro" in metrics:
        print(
            f"[TEST] acc={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f} f1_micro={metrics['f1_micro']:.4f}"
        )
