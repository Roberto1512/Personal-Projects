from __future__ import annotations

import json
from pathlib import Path
import random
from typing import List

from alibi_detect.cd import KSDrift
import numpy as np
from sentence_transformers import SentenceTransformer

INTERIM_TRAIN = Path("data/interim/train.jsonl")


def read_texts(path: Path, max_n: int = 500) -> List[str]:
    """
    Legge fino a max_n testi dal JSONL.
    Usa il campo 'text' se presente.
    """
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            t = (obj.get("text") or "").strip()
            if t:
                texts.append(t)
            if len(texts) >= max_n:
                break
    return texts


def make_shifted_batch(texts: List[str], n: int = 200) -> List[str]:
    """
    Simula drift: prende testi e li altera in modo sistematico.
    Qui facciamo una cosa semplice ma efficace:
    - aggiungiamo parole "fuori dominio"
    - cambiamo stile (es: parole di contesto diverso)
    """
    noise_tokens = [
        "cryptocurrency",
        "blockchain",
        "investment",
        "trading",
        "football",
        "stadium",
        "championship",
        "recipe",
        "kitchen",
        "ingredients",
        "travel",
        "hotel",
        "flight",
        "reservation",
    ]

    random.seed(42)
    sampled = random.sample(texts, k=min(n, len(texts)))

    shifted = []
    for t in sampled:
        extra = " ".join(random.sample(noise_tokens, k=4))
        shifted.append(f"{t} {extra}")
    return shifted


def main() -> None:
    if not INTERIM_TRAIN.exists():
        raise SystemExit(
            f"Train split not found: {INTERIM_TRAIN}. "
            "Run your data pipeline first (convert + split)."
        )

    # 1) Reference data (baseline)
    ref_texts = read_texts(INTERIM_TRAIN, max_n=800)
    if len(ref_texts) < 200:
        raise SystemExit("Not enough reference texts to run a drift demo.")

    # 2) Current data (two cases)
    # Case A: no drift (same distribution)
    cur_texts_no_drift = random.sample(ref_texts, k=200)

    # Case B: drift (simulated)
    cur_texts_drift = make_shifted_batch(ref_texts, n=200)

    # 3) Embeddings (turn text into numeric vectors)
    print("[AlibiDetect] Loading sentence-transformer embedder...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("[AlibiDetect] Computing reference embeddings...")
    X_ref = embedder.encode(ref_texts[:500], convert_to_numpy=True)
    X_ref = X_ref.astype(np.float32)

    print("[AlibiDetect] Building KSDrift detector...")
    detector = KSDrift(X_ref, p_val=0.05)

    # ---- NO DRIFT TEST ----
    print("\n=== TEST A: Expected NO drift ===")
    X_cur_a = embedder.encode(cur_texts_no_drift, convert_to_numpy=True).astype(np.float32)
    pred_a = detector.predict(X_cur_a)

    print("Drift detected:", bool(pred_a["data"]["is_drift"]))
    print("p-value (min over features):", float(np.min(pred_a["data"]["p_val"])))

    # ---- DRIFT TEST ----
    print("\n=== TEST B: Expected DRIFT ===")
    X_cur_b = embedder.encode(cur_texts_drift, convert_to_numpy=True).astype(np.float32)
    pred_b = detector.predict(X_cur_b)

    print("Drift detected:", bool(pred_b["data"]["is_drift"]))
    print("p-value (min over features):", float(np.min(pred_b["data"]["p_val"])))

    print("\nDone.")


if __name__ == "__main__":
    main()
