# tests/test_cli_prepare_unit.py

import json

from naplace.cli.prepare import (
    detect_encoding,
    normalize_jsonl,
    json_to_jsonl,
)


def test_json_to_jsonl_from_list(tmp_path):
    # Creiamo un JSON "lista di dict"
    src = tmp_path / "data.json"
    dst = tmp_path / "out.jsonl"

    data = [
        {"id": 1, "summary": "First"},
        {"id": 2, "summary": "Second"},
    ]
    src.write_text(json.dumps(data), encoding="utf-8")

    json_to_jsonl(src, dst)

    assert dst.exists()

    lines = dst.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1 == data[0]
    assert rec2 == data[1]


def test_normalize_jsonl_filters_invalid_lines(tmp_path):
    src = tmp_path / "raw.jsonl"
    dst = tmp_path / "clean.jsonl"

    # 2 righe valide, 1 vuota, 1 non-JSON
    lines = [
        json.dumps({"id": 1, "a": "ok"}),
        "",
        "NOT JSON",
        json.dumps({"id": 2, "a": "ok2"}),
    ]
    src.write_text("\n".join(lines), encoding="utf-8")

    normalize_jsonl(src, dst)

    assert dst.exists()

    out_lines = [l for l in dst.read_text(encoding="utf-8").split("\n") if l.strip()]
    assert len(out_lines) == 2

    objs = [json.loads(l) for l in out_lines]
    ids = [o["id"] for o in objs]
    assert ids == [1, 2]


def test_detect_encoding_simple_utf8(tmp_path):
    # Caso base: file UTF-8 normale
    src = tmp_path / "file.txt"
    src.write_text("ciao", encoding="utf-8")

    enc = detect_encoding(src)
    # Può restituire "utf-8" o "utf-8-sig" a seconda dei controlli,
    # l'importante è che non esploda e che sia una stringa "sensata".
    assert isinstance(enc, str)
    assert "utf-8" in enc.lower()
