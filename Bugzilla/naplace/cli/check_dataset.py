# naplace/cli/check_dataset.py

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict

from naplace.config import INTERIM


def _check_split(path: Path, split_name: str, max_examples: int = 3) -> Dict[str, Any]:
    if not path.is_file():
        print(f"❌ [{split_name}] File non trovato: {path}")
        return {"exists": False}

    print(f"\n=== [{split_name.upper()}] Analisi file: {path} ===")

    n_rows = 0
    n_with_summary = 0
    n_with_text = 0
    n_with_both = 0

    components = Counter()
    macros = Counter()

    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print("[WARN] Riga non valida (JSON) – salto.")
                continue

            n_rows += 1

            summary = rec.get("summary", "") or ""
            text = rec.get("text", "") or ""

            has_summary = bool(summary.strip())
            has_text = bool(text.strip())

            if has_summary:
                n_with_summary += 1
            if has_text:
                n_with_text += 1
            if has_summary and has_text:
                n_with_both += 1

            comp = rec.get("component")
            if comp is not None:
                components[str(comp)] += 1

            macro = rec.get("macro_component")
            if macro is not None:
                macros[str(macro)] += 1

            if len(examples) < max_examples:
                examples.append(
                    {
                        "id": rec.get("id"),
                        "product": rec.get("product"),
                        "component": rec.get("component"),
                        "macro_component": rec.get("macro_component"),
                        "summary": summary[:120],
                        "text": text[:120],
                    }
                )

    if n_rows == 0:
        print(f"[{split_name}] ⚠ Nessuna riga valida trovata.")
        return {"exists": True, "n_rows": 0}

    print(f"[{split_name}] Num. righe valide: {n_rows}")
    print(f"[{split_name}] summary non vuoto: {n_with_summary} ({n_with_summary / n_rows:.2%})")
    print(f"[{split_name}] text non vuoto:    {n_with_text} ({n_with_text / n_rows:.2%})")
    print(f"[{split_name}] entrambi:          {n_with_both} ({n_with_both / n_rows:.2%})")

    print("\n[Component] numero classi distinte:", len(components))
    print("[Component] top 15 classi per frequenza:")
    for comp, cnt in components.most_common(15):
        print(f"  - {comp}: {cnt}")

    if macros:
        print("\n[Macro_component] distribuzione:")
        for macro, cnt in macros.most_common():
            print(f"  - {macro}: {cnt}")
    else:
        print("\n[Macro_component] ⚠ Nessun campo 'macro_component' trovato nel file.")

    print("\nEsempi (troncati):")
    for ex in examples:
        print("-" * 60)
        print(
            f"id={ex['id']} | product={ex['product']} | "
            f"component={ex['component']} | macro={ex['macro_component']}"
        )
        print(f"summary: {ex['summary']!r}")
        print(f"text:    {ex['text']!r}")

    return {
        "exists": True,
        "n_rows": n_rows,
        "n_components": len(components),
        "components_counter": components,
        "macros_counter": macros,
    }


def main() -> None:
    train_path = INTERIM / "train.jsonl"
    test_path = INTERIM / "test.jsonl"

    print("=== Naplace dataset sanity check ===")
    print(f"INTERIM dir: {INTERIM}\n")

    train_stats = _check_split(train_path, "train")
    test_stats = _check_split(test_path, "test")

    print("\n=== Riepilogo finale ===")
    if not train_stats.get("exists") or not test_stats.get("exists"):
        print("❌ Mancano uno o entrambi gli split (train/test).")
        return

    print(
        f"- train: {train_stats.get('n_rows', 0)} righe, "
        f"{train_stats.get('n_components', 0)} classi component"
    )
    print(
        f"- test:  {test_stats.get('n_rows', 0)} righe, "
        f"{test_stats.get('n_components', 0)} classi component"
    )


if __name__ == "__main__":
    main()
