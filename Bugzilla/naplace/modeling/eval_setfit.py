from __future__ import annotations

import argparse
from pathlib import Path

from naplace.modeling.setfit_model import SetFitConfig, eval_setfit_on_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Valuta il modello SetFit su data/interim/test.jsonl.")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/setfit_component"),
        help="Cartella dove è salvato il modello SetFit (save_pretrained).",
    )
    p.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/interim/test.jsonl"),
        help="File JSONL di test.",
    )
    p.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("reports/metrics_setfit_eval.json"),
        help="Dove salvare le metriche di test.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = SetFitConfig(
        text_fields=("text",),
        label_field="component",
    )

    eval_setfit_on_test(
        model_dir=args.model_dir,
        test_path=args.test_path,
        metrics_path=args.metrics_path,
        config=config,
    )
