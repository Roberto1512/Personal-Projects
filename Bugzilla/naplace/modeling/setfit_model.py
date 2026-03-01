from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Disable Transformers/SetFit MLflow auto-integration (causes callback crash on Windows HTTPS URIs).
os.environ.setdefault("DISABLE_MLFLOW_INTEGRATION", "1")
os.environ.setdefault("HF_MLFLOW_LOGGING_ENABLED", "0")
from datasets import Dataset
import pandas as pd
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, f1_score


@dataclass
class SetFitConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    num_epochs: int = 4
    batch_size: int = 16
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-2
    seed: int = 42
    text_fields: Tuple[str, ...] = ("summary", "text")
    label_field: str = "component"
    max_samples_per_label: Optional[int] = 8  # limit few-shot pairs to avoid OOM on large datasets
    num_iterations: Optional[int] = 10  # limit contrastive pairs to avoid huge datasets

    def __post_init__(self) -> None:
        # Normalize text_fields so a single string becomes a tuple.
        if isinstance(self.text_fields, str):
            self.text_fields = (self.text_fields,)
        else:
            self.text_fields = tuple(self.text_fields)


def _load_split(
    path: Path,
    text_fields: Iterable[str],
    label_field: str,
    label_vocab: Optional[List[str]] = None,
    max_samples_per_label: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Carica uno split da JSONL e restituisce:
      - texts: lista di stringhe (summary + text, ecc.)
      - label_ids: codici interi coerenti con label_vocab
      - label_vocab: lista ordinata di tutte le label (solo per il train)
    Se label_vocab è None → viene costruita da questo file (tipicamente train).
    Se label_vocab è fornita → le label vengono mappate usando quella vocab
    e gli esempi con label sconosciute vengono scartati.
    """
    df = pd.read_json(path, lines=True)

    if isinstance(text_fields, str):
        text_fields = (text_fields,)
    else:
        text_fields = tuple(text_fields)

    missing = [f for f in text_fields + (label_field,) if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    # Se abbiamo già una vocab (caso VAL), scartiamo le label non presenti
    if label_vocab is not None:
        label_series = df[label_field].astype(str)
        known = set(label_vocab)
        mask = label_series.isin(known)
        if not mask.all():
            dropped = (~mask).sum()
            print(
                f"[SetFit] Warning: dropping {dropped} examples in {path} "
                f"with labels not seen in train."
            )
            df = df[mask]

    texts = df[list(text_fields)].fillna("").agg(" ".join, axis=1).tolist()

    if label_vocab is None:
        # Train: costruiamo la vocab dalle categorie presenti
        labels_cat = df[label_field].astype("category")
        labels_cat = labels_cat.cat.remove_unused_categories()
        label_vocab = labels_cat.cat.categories.astype(str).tolist()

        if max_samples_per_label is not None and max_samples_per_label > 0:
            # Downsample per label to keep SetFit pairs manageable on large datasets.
            df = df.assign(_label=labels_cat.cat.codes)
            df = df.groupby("_label", group_keys=False).apply(
                lambda g: g.sample(
                    n=min(len(g), max_samples_per_label),
                    random_state=seed,
                )
            )
            labels_cat = df["_label"].astype("int")
            texts = df[list(text_fields)].fillna("").agg(" ".join, axis=1).tolist()

        label_ids = (
            labels_cat.cat.codes.tolist() if hasattr(labels_cat, "cat") else labels_cat.tolist()
        )
    else:
        # Val/Test: usiamo la vocab fissata dal train
        label_series = df[label_field].astype(str)
        mapping = {lab: i for i, lab in enumerate(label_vocab)}
        label_ids = [mapping[lab] for lab in label_series]

    return texts, label_ids, label_vocab


def _make_hf_dataset(
    texts: List[str],
    label_ids: List[int],
) -> Dataset:
    return Dataset.from_dict({"text": texts, "label": label_ids})


def train_setfit_classifier(
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    metrics_path: Path,
    config: SetFitConfig,
) -> Dict[str, Any]:
    """
    Allena SetFit, salva modello + mapping id↔label + metriche di validazione,
    e restituisce un dict con paths e metriche.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) TRAIN → testi, id e vocab comune
    train_texts, train_labels_ids, label_vocab = _load_split(
        train_path,
        text_fields=config.text_fields,
        label_field=config.label_field,
        label_vocab=None,
        max_samples_per_label=config.max_samples_per_label,
        seed=config.seed,
    )

    # 2) VAL → codifica label usando la stessa vocab del train
    val_texts, val_labels_ids, _ = _load_split(
        val_path,
        text_fields=config.text_fields,
        label_field=config.label_field,
        label_vocab=label_vocab,
        max_samples_per_label=None,
        seed=config.seed,
    )

    train_dataset = _make_hf_dataset(train_texts, train_labels_ids)
    val_dataset = _make_hf_dataset(val_texts, val_labels_ids)

    # ==============
    # LABEL MAPPING
    # ==============
    # Creiamo mapping id↔label a partire dalla vocab fissata
    id2label = {int(i): lab for i, lab in enumerate(label_vocab)}
    label2id = {lab: int(i) for i, lab in enumerate(label_vocab)}

    label_mapping = {"id2label": id2label, "label2id": label2id}
    mapping_path = output_dir / "label_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    print(f"[SetFit] Label mapping saved to {mapping_path}")

    # Modello SetFit
    model = SetFitModel.from_pretrained(
        config.model_name,
        use_differentiable_head=True,
        head_params={"out_features": len(label_vocab)},
        num_labels=len(label_vocab),
        id2label=id2label,
        label2id=label2id,
    )

    # Usa la nuova API Trainer con TrainingArguments espliciti e report_to disabilitato
    num_iterations = (
        None
        if config.num_iterations is not None and config.num_iterations <= 0
        else config.num_iterations
    )

    train_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        body_learning_rate=config.learning_rate,
        head_learning_rate=config.head_learning_rate,
        seed=config.seed,
        num_iterations=num_iterations,
        report_to=[],
        logging_strategy="no",
        eval_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        metric="accuracy",
    )

    # Rimuovi callback di tracking (MLflow/DagsHub) che su Windows rompono per via dei path con os.sep.
    def _strip_tracking_callbacks(callback_handler):
        if not hasattr(callback_handler, "callbacks"):
            return
        keep = []
        for cb in list(callback_handler.callbacks):
            name = cb.__class__.__name__.lower()
            module = getattr(cb.__class__, "__module__", "").lower()
            if "mlflow" in name or "mlflow" in module or "dagshub" in name or "dagshub" in module:
                continue
            keep.append(cb)
        callback_handler.callbacks = keep

    try:
        trainer.st_trainer.args.report_to = []
        trainer.args.report_to = []
        _strip_tracking_callbacks(trainer.st_trainer.callback_handler)
        _strip_tracking_callbacks(trainer.callback_handler)
    except Exception:
        pass

    # Compatibility patch: newer transformers pass num_items_in_batch to compute_loss.
    try:
        import inspect

        from sentence_transformers.trainer import SentenceTransformerTrainer

        if (
            "num_items_in_batch"
            not in inspect.signature(SentenceTransformerTrainer.compute_loss).parameters
        ):
            _orig_compute_loss = SentenceTransformerTrainer.compute_loss

            def _compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return _orig_compute_loss(
                    self,
                    model,
                    inputs,
                    return_outputs=return_outputs,
                )

            SentenceTransformerTrainer.compute_loss = _compute_loss
    except Exception:
        pass

    trainer.train()

    # accuracy (da trainer)
    acc_metrics = trainer.evaluate()  # es. {"accuracy": 0.81}
    accuracy = float(acc_metrics.get("accuracy", 0.0))

    # f1_macro calcolato a mano
    val_preds_ids = model.predict(val_texts)
    f1_macro = f1_score(val_labels_ids, val_preds_ids, average="macro")

    print(f"[SetFit] Validation accuracy: {accuracy:.4f}")
    print(f"[SetFit] Validation F1 macro: {f1_macro:.4f}")

    metrics = {
        "accuracy": accuracy,
        "f1_macro": float(f1_macro),
        "num_labels": len(label_vocab),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Salva modello
    model.save_pretrained(str(output_dir))
    print(f"[SetFit] Model saved to {output_dir}")
    print(f"[SetFit] Metrics saved to {metrics_path}")

    return {
        "metrics": metrics,
        "model_dir": str(output_dir),
        "metrics_path": str(metrics_path),
    }


def eval_setfit_on_test(
    model_dir: Path,
    test_path: Path,
    metrics_path: Path,
    config: SetFitConfig,
) -> Dict[str, Any]:
    """
    Valuta il modello SetFit salvato su data/interim/test.jsonl.

    Usa il mapping id↔label salvato in model_dir/label_mapping.json,
    confronta le label STRINGA (component) e salva:
      - accuracy
      - f1_macro
      - f1_micro
      - classification_report
    in metrics_path (es. reports/metrics_setfit_eval.json).
    """
    model_dir = Path(model_dir)
    test_path = Path(test_path)
    metrics_path = Path(metrics_path)

    if not model_dir.is_dir():
        raise SystemExit(f"❌ model_dir non esiste: {model_dir}")
    if not test_path.is_file():
        raise SystemExit(f"❌ test_path non esiste: {test_path}")

    # Carichiamo mapping id↔label
    mapping_file = model_dir / "label_mapping.json"
    if not mapping_file.is_file():
        raise SystemExit(
            f"❌ label_mapping.json non trovato in {mapping_file}. "
            "Assicurati di aver rieseguito il training SetFit aggiornato."
        )

    with mapping_file.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    id2label_raw = mapping.get("id2label", {})
    # Chiavi possono essere stringhe nel JSON → convertiamo a int
    id2label: Dict[int, str] = {int(k): str(v) for k, v in id2label_raw.items()}

    label2id_raw = mapping.get("label2id", {})
    label2id: Dict[str, int] = {str(k): int(v) for k, v in label2id_raw.items()}

    # Carichiamo test: testi + label stringa
    df = pd.read_json(test_path, lines=True)

    if isinstance(config.text_fields, str):
        text_fields: Tuple[str, ...] = (config.text_fields,)
    else:
        text_fields = tuple(config.text_fields)

    missing = [c for c in text_fields + (config.label_field,) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {test_path}: {missing}")

    texts_all = df[list(text_fields)].fillna("").agg(" ".join, axis=1).tolist()
    labels_all = df[config.label_field].astype(str).tolist()

    # Filtriamo esempi con label NON presenti nel mapping (per sicurezza)
    X: List[str] = []
    y_true_labels: List[str] = []
    skipped = 0

    for txt, lbl in zip(texts_all, labels_all):
        if lbl in label2id:
            X.append(txt)
            y_true_labels.append(lbl)
        else:
            skipped += 1

    if not X:
        raise SystemExit(
            "❌ Nessun esempio nel test set con label compatibili col mapping SetFit."
        )

    print(f"[SetFit-EVAL] Esempi totali nel test: {len(texts_all)}")
    print(f"[SetFit-EVAL] Esempi usati (label note al modello): {len(X)}")
    print(f"[SetFit-EVAL] Esempi scartati (label sconosciute): {skipped}")

    # Carichiamo modello SetFit salvato
    model = SetFitModel.from_pretrained(str(model_dir))

    # Predizioni (id) e poi label stringa
    pred_ids = model.predict(X)
    y_pred_labels = [id2label[int(i)] for i in pred_ids]

    # Metriche su label stringa
    acc = accuracy_score(y_true_labels, y_pred_labels)
    f1_macro = f1_score(y_true_labels, y_pred_labels, average="macro")
    f1_micro = f1_score(y_true_labels, y_pred_labels, average="micro")

    print(f"[SetFit-EVAL] Test accuracy:  {acc:.4f}")
    print(f"[SetFit-EVAL] Test F1 macro: {f1_macro:.4f}")
    print(f"[SetFit-EVAL] Test F1 micro: {f1_micro:.4f}")

    report = classification_report(y_true_labels, y_pred_labels, output_dict=True)

    metrics: Dict[str, Any] = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "n_test_total": int(len(texts_all)),
        "n_test_used": int(len(X)),
        "n_test_skipped_unknown_labels": int(skipped),
        "classification_report": report,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[SetFit-EVAL] Metrics salvate in {metrics_path}")

    return metrics
