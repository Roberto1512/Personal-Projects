from __future__ import annotations

import argparse
from pathlib import Path

import dagshub
import mlflow

from naplace.modeling.baseline_tfidf import TFIDFConfig, train_tfidf_classifier


def parse_args():
    p = argparse.ArgumentParser(description="Train a TF-IDF + Logistic Regression baseline model.")
    p.add_argument("--train-path", type=Path, default=Path("data/interim/train.jsonl"))
    p.add_argument("--test-path", type=Path, default=Path("data/interim/test.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("models/baseline_tfidf"))
    p.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("reports/metrics_baseline_tfidf.json"),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # MLflow / DagsHub
    dagshub.init(repo_owner="se4ai2526-uniba", repo_name="Naplace", mlflow=True)
    mlflow.set_experiment("Naplace Bug Report Classification")

    config = TFIDFConfig(
        text_fields=("text",),
        label_field="component",
    )

    with mlflow.start_run(run_name="baseline_tfidf"):
        # log parametri principali
        mlflow.log_param("model", "tfidf_logreg")
        mlflow.log_param("text_fields", ",".join(config.text_fields))
        mlflow.log_param("label_field", config.label_field)
        mlflow.log_param("max_features", config.max_features)
        mlflow.log_param("ngram_range", config.ngram_range)
        mlflow.log_param("C", config.C)
        mlflow.log_param("penalty", config.penalty)
        mlflow.log_param("max_iter", config.max_iter)
        mlflow.log_param("train_path", str(args.train_path))
        mlflow.log_param("test_path", str(args.test_path))

        result = train_tfidf_classifier(
            train_path=args.train_path,
            test_path=args.test_path,
            output_dir=args.output_dir,
            metrics_path=args.metrics_path,
            config=config,
        )

        metrics = result["metrics"]
        # log metriche principali su MLflow
        for key in ("accuracy", "f1_macro", "f1_micro"):
            if key in metrics and isinstance(metrics[key], (int, float, float)):
                mlflow.log_metric(key, float(metrics[key]))

        # log artifacts
        mlflow.log_artifacts(result["model_dir"], artifact_path="baseline_tfidf")
        mlflow.log_artifact(result["metrics_path"])

        print("[TFIDF] Run logged to MLflow.")
