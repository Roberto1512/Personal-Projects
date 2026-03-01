from __future__ import annotations

import argparse
from pathlib import Path
import sys

import dagshub
import mlflow

from naplace.modeling.setfit_model import SetFitConfig, train_setfit_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SetFit text classifier on bug reports.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/interim/train.jsonl"),
        help="Path to the training JSONL file.",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=Path("data/interim/test.jsonl"),
        help="Path to the validation JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/setfit_component"),
        help="Where to save the trained SetFit model.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("reports/metrics_setfit_component.json"),
        help="Where to save the SetFit metrics JSON.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base sentence-transformer to use inside SetFit.",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Body learning rate for the embedding training phase.",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-2,
        help="Head learning rate for the classifier phase (differentiable head).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples-per-label",
        type=int,
        default=64,
        help="Cap training examples per label to keep SetFit contrastive pairs manageable.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of contrastive iterations; each iteration adds ~2 * n_sentences pairs. Lower = less memory.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Avoid MLflow's emoji log messages breaking on Windows cp1252 consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    dagshub.init(repo_owner="se4ai2526-uniba", repo_name="Naplace", mlflow=True)
    mlflow.set_experiment("Naplace Bug Report Classification")

    # 🔴 Disattiva l’autolog di MLflow per evitare conflitti con Transformers/SetFit
    mlflow.autolog(disable=True)

    config = SetFitConfig(
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        head_learning_rate=args.head_lr,
        seed=args.seed,
        text_fields=("text",),
        label_field="component",
        max_samples_per_label=args.max_samples_per_label,
        num_iterations=args.num_iterations,
    )

    with mlflow.start_run(run_name="setfit_component"):
        # parametri
        mlflow.log_param("model", "setfit")
        mlflow.log_param("base_model", config.model_name)
        mlflow.log_param("num_epochs", config.num_epochs)
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("learning_rate", config.learning_rate)
        mlflow.log_param("head_learning_rate", config.head_learning_rate)
        mlflow.log_param("seed", config.seed)
        mlflow.log_param("max_samples_per_label", config.max_samples_per_label)
        mlflow.log_param("num_iterations", config.num_iterations)
        mlflow.log_param("text_fields", ",".join(config.text_fields))
        mlflow.log_param("label_field", config.label_field)
        mlflow.log_param("train_path", str(args.train_path))
        mlflow.log_param("val_path", str(args.val_path))

        result = train_setfit_classifier(
            train_path=args.train_path,
            val_path=args.val_path,
            output_dir=args.output_dir,
            metrics_path=args.metrics_path,
            config=config,
        )

        metrics = result["metrics"]
        for k, v in metrics.items():
            if isinstance(v, (int, float, float)):
                mlflow.log_metric(k, float(v))

        mlflow.log_artifacts(result["model_dir"], artifact_path="setfit_component")
        mlflow.log_artifact(result["metrics_path"])

        print("[SetFit] Run logged to MLflow.")
