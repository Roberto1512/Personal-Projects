import json
from pathlib import Path
import pickle

import dagshub
import mlflow
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def read_jsonl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]


if __name__ == "__main__":
    # Init DagsHub + MLflow
    dagshub.init(repo_owner="se4ai2526-uniba", repo_name="Naplace", mlflow=True)
    mlflow.set_experiment("Naplace Bug Report Classification")

    Path("models").mkdir(exist_ok=True, parents=True)
    Path("reports").mkdir(exist_ok=True, parents=True)

    # Caricamento dataset
    train = read_jsonl("data/interim/train.jsonl")
    X_text = [r.get("text", "") for r in train]
    y_lbl = [r.get("component", "Unknown") for r in train]

    max_words, max_len, emb_dim = 30000, 200, 128

    # Tokenizer
    tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tok.fit_on_texts(X_text)
    X = pad_sequences(tok.texts_to_sequences(X_text), maxlen=max_len)

    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(y_lbl)
    n_classes = len(le.classes_)

    # =============================
    # CLASS WEIGHTS (ANTI-COLLASSO)
    # =============================
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y,
    )
    class_weights = {int(cls): float(w) for cls, w in zip(np.unique(y), class_weights_array)}

    print("Class weights generati (GRU):")
    print("Esempio primi 10:", list(class_weights.items())[:10])

    # Modello GRU
    model = models.Sequential(
        [
            layers.Embedding(max_words, emb_dim, input_length=max_len),
            layers.GRU(128),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Training con MLflow
    with mlflow.start_run(run_name="gru_train"):
        mlflow.log_params(
            {
                "model": "gru",
                "max_words": max_words,
                "max_len": max_len,
                "emb_dim": emb_dim,
                "n_classes": n_classes,
            }
        )

        # TRAIN
        history = model.fit(
            X,
            y,
            epochs=5,
            batch_size=64,
            validation_split=0.1,
            verbose=2,
            class_weight=class_weights,
        )

        # Estrazione metriche finali
        train_acc = history.history.get("accuracy", [None])[-1]
        val_acc = history.history.get("val_accuracy", [None])[-1]
        train_loss = history.history.get("loss", [None])[-1]
        val_loss = history.history.get("val_loss", [None])[-1]

        if train_acc is not None:
            mlflow.log_metric("train_accuracy", float(train_acc))
        if val_acc is not None:
            mlflow.log_metric("val_accuracy", float(val_acc))
        if train_loss is not None:
            mlflow.log_metric("train_loss", float(train_loss))
        if val_loss is not None:
            mlflow.log_metric("val_loss", float(val_loss))

        # Salvataggio artefatti modello
        np.save("models/gru_label_classes.npy", le.classes_)
        with open("models/gru_tokenizer.pkl", "wb") as f:
            pickle.dump(tok, f)
        model.save("models/gru.h5")

        # Salvo un JSON di metriche (coerente con metrics_gru.json)
        metrics = {
            "train_accuracy": float(train_acc) if train_acc is not None else None,
            "val_accuracy": float(val_acc) if val_acc is not None else None,
            "train_loss": float(train_loss) if train_loss is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
        }
        metrics_path = Path("reports/metrics_gru.json")
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Log MLflow artifact
        mlflow.log_artifact("models/gru.h5")
        mlflow.log_artifact("models/gru_tokenizer.pkl")
        mlflow.log_artifact("models/gru_label_classes.npy")
        mlflow.log_artifact(str(metrics_path))

        print("GRU training completato e loggato su MLflow.")
