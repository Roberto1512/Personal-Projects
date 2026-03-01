# naplace/cli/split.py
from collections import Counter
from hashlib import md5
import json
from pathlib import Path

from naplace.config import INTERIM

RAW = INTERIM / "bugbug_converted.jsonl"
TRAIN = INTERIM / "train.jsonl"
TEST = INTERIM / "test.jsonl"


TEST_RATIO = 0.20  # 20% test
HASH_SALT = "naplace_v1"  # cambia se vuoi rendere *diversa* la partizione


def read_jsonl_lines(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line


def parse_record(line: str):
    try:
        return json.loads(line)
    except Exception:
        return None


def extract_label(rec: dict):
    # Etichetta: prova a leggere il campo "component"
    # Se non c'è, metti "Unknown"
    return rec.get("component", "Unknown")


def build_text(rec: dict, max_desc_len: int = 400):
    # Se il record ha già 'text', usiamo quello
    if "text" in rec and rec["text"]:
        return (rec["text"] or "").strip()

    # Fallback: ricostruisci da summary + primo commento / description
    summary = rec.get("summary") or ""
    desc = ""
    if "comments" in rec and isinstance(rec["comments"], list) and rec["comments"]:
        first = rec["comments"][0]
        if isinstance(first, dict):
            desc = first.get("text") or ""
    elif "description" in rec:
        desc = rec.get("description") or ""

    if max_desc_len is not None and len(desc) > max_desc_len:
        desc = desc[:max_desc_len]

    return (summary + " " + desc).strip()


def stable_bucket(key: str, ratio: float) -> str:
    """
    Assegna in modo deterministico 'train' o 'test' in base a un hash.
    Usa un sale (HASH_SALT) per rendere la partizione ripetibile ma modificabile.
    """
    h = md5((key + HASH_SALT).encode("utf-8")).hexdigest()
    # usa i primi 8 char per un intero in [0, 2^32)
    val = int(h[:8], 16) / 0xFFFFFFFF
    return "test" if val < ratio else "train"


def main():
    # --- PASSO 1: Conta esempi per classe (streaming) ---
    counts = Counter()
    for line in read_jsonl_lines(RAW):
        rec = parse_record(line)
        if rec is None:
            continue
        lab = extract_label(rec)
        counts[lab] += 1

    # Filtra le classi con almeno 2 esempi
    valid_labels = {lab for lab, c in counts.items() if c >= 2}
    if not valid_labels:
        raise SystemExit("❌ Nessuna classe con >= 2 esempi: impossibile fare split.")

    TRAIN.parent.mkdir(parents=True, exist_ok=True)
    TEST.parent.mkdir(parents=True, exist_ok=True)

    # --- PASSO 2: Rileggi e scrivi su train/test in streaming (deterministico) ---
    with TRAIN.open("w", encoding="utf-8") as ftrain, TEST.open("w", encoding="utf-8") as ftest:
        n_train = n_test = 0
        for line in read_jsonl_lines(RAW):
            rec = parse_record(line)
            if rec is None:
                continue
            lab = extract_label(rec)
            if lab not in valid_labels:
                # salta classi troppo piccole per stratify
                continue

            # costruisci il campo 'text' (summary + primo commento/descrizione)
            rec["text"] = build_text(rec)

            # chiave stabile: preferisci 'id', altrimenti fallback su summary
            key = str(rec.get("id")) if rec.get("id") is not None else (rec.get("summary") or "")
            bucket = stable_bucket(key, TEST_RATIO)

            if bucket == "test":
                ftest.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_test += 1
            else:
                ftrain.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_train += 1

    print(f"✅ Train: {n_train}  Test: {n_test}  (ratio ~{TEST_RATIO})")


if __name__ == "__main__":
    main()
