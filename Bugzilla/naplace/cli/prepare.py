# naplace/cli/prepare.py

import io
import json
from pathlib import Path

from naplace.config import RAW

# Sorgenti "esterni" (se un domani ti danno il dump BugBug in JSON/JSONL)
SRC_JSONL = Path("data/external/bugzilla/bugbug_dataset.jsonl")
SRC_JSON = Path("data/external/bugzilla/bugbug_dataset.json")

# Destinazione "ufficiale" usata dal progetto
DST = RAW / "bugbug_dataset.jsonl"


def detect_encoding(path: Path) -> str:
    """
    Rileva in modo semplice l'encoding di un file di testo, con supporto a BOM.
    Questa funzione è quella usata nei test.
    """
    with path.open("rb") as f:
        head = f.read(4)

    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        return "utf-16"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    # fallback: tentativi comuni
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            path.read_text(encoding=enc)
            return enc
        except Exception:
            continue

    return "latin-1"


def normalize_jsonl(src: Path, dst: Path) -> None:
    """
    Legge un file JSONL con encoding potenzialmente strano e scrive un JSONL
    pulito UTF-8, filtrando righe vuote o non-JSON.
    """
    enc = detect_encoding(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with (
        io.open(src, "r", encoding=enc, errors="ignore") as fin,
        io.open(dst, "w", encoding="utf-8", newline="\n") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # linea non-JSON: la saltiamo (comportamento atteso dai test)
                continue
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def json_to_jsonl(src_json: Path, dst_jsonl: Path) -> None:
    """
    Converte un file JSON (array o singolo oggetto) in JSONL.
    Questa è esattamente la funzione che i test usano.
    """
    enc = detect_encoding(src_json)
    text = src_json.read_text(encoding=enc, errors="ignore")
    obj = json.loads(text)

    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with dst_jsonl.open("w", encoding="utf-8") as f:
        if isinstance(obj, list):
            for item in obj:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    """
    Entry point "pratico":
    - Se esiste un JSONL esterno in data/external/bugzilla, lo normalizza in RAW.
    - Se esiste un JSON (array), lo converte in JSONL in RAW.
    - Se non c'è nulla, fallisce con messaggio chiaro.
    """
    if SRC_JSONL.exists():
        normalize_jsonl(SRC_JSONL, DST)
        print(f"Normalized {SRC_JSONL} -> {DST} (UTF-8)")
    elif SRC_JSON.exists():
        json_to_jsonl(SRC_JSON, DST)
        print(f"Converted {SRC_JSON} -> {DST} (UTF-8)")
    else:
        raise SystemExit(
            "❌ No external BugBug dataset found in data/external/bugzilla/.\n"
            "Per il flusso attuale stai già usando data/raw/bugbug_dataset.jsonl, "
            "quindi non serve lanciare questo script."
        )
