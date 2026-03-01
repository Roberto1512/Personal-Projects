from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

import dagshub
import mlflow

from naplace.modeling.setfit_model import (
    SetFitConfig,
    eval_setfit_on_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SetFit classifier on the test split."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("naplace/modeling/setfit_component"),
        help="Directory where the trained SetFit model is saved.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/interim/test.jsonl"),
        help="Path to the test JSONL file.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("reports/metrics_setfit_eval.json"),
        help="Where to save evaluation metrics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    # Avoid MLflow's emoji log messages breaking on Windows cp1252 consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    # Inizializza DagsHub + MLflow (stessa convenzione degli altri script)
    dagshub.init(repo_owner="se4ai2526-uniba", repo_name="Naplace", mlflow=True)
    mlflow.set_experiment("Naplace Bug Report Classification")

    config = SetFitConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        num_epochs=4,
        batch_size=16,
        learning_rate=2e-5,
        seed=42,
        text_fields=("text",),
        label_field="component",
    )

    with mlflow.start_run(run_name="setfit_eval_test"):
        # Log parametri principali dell’eval
        mlflow.log_param("model", "setfit")
        mlflow.log_param("run_type", "eval_test")
        mlflow.log_param("model_dir", str(args.model_dir))
        mlflow.log_param("test_path", str(args.test_path))
        mlflow.log_param("metrics_path", str(args.metrics_path))
        mlflow.log_param("text_fields", ",".join(config.text_fields))
        mlflow.log_param("label_field", config.label_field)

        # Esegui valutazione su test.jsonl
        metrics: Dict[str, Any] = eval_setfit_on_test(
            model_dir=args.model_dir,
            test_path=args.test_path,
            metrics_path=args.metrics_path,
            config=config,
        )

        # Log delle metriche numeriche su MLflow
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))

        # Salva il file JSON di metriche come artifact
        mlflow.log_artifact(str(args.metrics_path))

        print("[SetFit-EVAL] Run logged to MLflow.")


if __name__ == "__main__":
    main()
