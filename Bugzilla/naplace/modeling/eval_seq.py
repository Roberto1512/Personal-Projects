import argparse
import json
from pathlib import Path
import pickle

import dagshub
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def read_jsonl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def main():
    # Init tracking only when running as a script to avoid side effects on import.
    dagshub.init(repo_owner="se4ai2526-uniba", repo_name="Naplace", mlflow=True)
    mlflow.set_experiment("Naplace Bug Report Classification")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--data", default="data/interim/test.jsonl")
    ap.add_argument("--maxlen", type=int, default=200)
    ap.add_argument("--out", default="reports/metrics.json")
    ap.add_argument("--run_name", default="eval")
    args = ap.parse_args()

    Path("reports").mkdir(parents=True, exist_ok=True)

    test = read_jsonl(args.data)
    X_text = [r.get("text", "") for r in test]
    y_lbl = [r.get("component", "Unknown") for r in test]

    tok = pickle.load(open(args.tok, "rb"))
    X = pad_sequences(tok.texts_to_sequences(X_text), maxlen=args.maxlen)
    classes = np.load(args.classes, allow_pickle=True)
    lab2id = {lab: i for i, lab in enumerate(classes)}
    y = np.array([lab2id.get(v, -1) for v in y_lbl])
    keep = y >= 0
    X, y = X[keep], y[keep]

    model = load_model(args.model)
    y_pred = model.predict(X, verbose=0).argmax(axis=1)

    metrics = {
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "f1_micro": float(f1_score(y, y_pred, average="micro")),
        "accuracy": float(accuracy_score(y, y_pred)),
    }
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Metrics:", metrics)

    with mlflow.start_run(run_name=args.run_name):
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_artifact(args.out)


if __name__ == "__main__":
    main()
