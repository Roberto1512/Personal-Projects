# naplace/cli/convert_bugbug.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from naplace.config import INTERIM, RAW
from naplace.labeling import label_bug

SRC = RAW / "bugbug_dataset.jsonl"
DST = INTERIM / "bugbug_converted.jsonl"


def _first_comment_text(obj: Dict[str, Any], max_len: int = 1000) -> str:
    comments = obj.get("comments", [])
    if not isinstance(comments, list):
        return ""
    for comment in comments:
        if isinstance(comment, dict):
            text = comment.get("text") or ""
        elif isinstance(comment, str):
            text = comment
        else:
            continue
        text = text.strip()
        if text:
            return text[:max_len]
    return ""


def convert_bugbug_dataset(src: Path, dst: Path) -> None:
    if not src.exists():
        raise SystemExit(f"❌ Sorgente non trovata: {src} (esegui prima naplace.cli.prepare)")

    dst.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    skipped_no_label = 0
    skipped_no_text = 0

    print(f"[Naplace] Converting BugBug dataset: {src} -> {dst}")

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                obj: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Campi base
            bug_id = obj.get("id")
            product = obj.get("product", "")
            component_raw = obj.get("component", "")

            summary = obj.get("summary", "") or ""
            # il campo descrizione può chiamarsi in modi diversi a seconda della sorgente
            description = obj.get("description") or obj.get("text") or obj.get("short_desc") or ""

            comment_text = _first_comment_text(obj)
            text_parts = [summary, description, comment_text]
            text = " ".join(part for part in text_parts if part).strip()
            if not text:
                skipped_no_text += 1
                continue

            # Etichette gerarchiche
            lbl = label_bug(product, component_raw)
            component_label = lbl.component_label
            macro_label = lbl.macro_component

            if not component_label:
                skipped_no_label += 1
                continue

            rec_out: Dict[str, Any] = {
                "id": bug_id,
                "product": product,
                "summary": summary,
                "text": text,
                "component": component_label,
                "macro_component": macro_label,
            }

            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[Naplace] Totale bug letti:        {total}")
    print(f"[Naplace] Bug convertiti (kept):  {kept}")
    print(f"[Naplace] Scartati senza testo:   {skipped_no_text}")
    print(f"[Naplace] Scartati senza label:   {skipped_no_label}")
    print(f"[Naplace] Output scritto in:      {dst}")


if __name__ == "__main__":
    convert_bugbug_dataset(SRC, DST)
