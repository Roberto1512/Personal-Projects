


import json

from bugbug import bugzilla, db
from bugbug.models import component

from naplace.config import RAW


def _macro_from_label(label: str) -> str | None:
    """
    Ricava il macro-component (livello 1) dalla label di BugBug.

    Esempi:
      - "Core::DOM"               -> "Core"
      - "Firefox for Android"     -> "Firefox for Android"
      - "DevTools"                -> "DevTools"
    """
    products = component.ComponentModel.PRODUCTS

    if "::" in label:
        macro = label.split("::", 1)[0]
        return macro if macro in products else None


    if label in products:
        return label


    return None


def download_bugbug_dataset():
    """
    Downloads a subset of Mozilla bug reports using Bugbug.
    Each bug includes:
      - id
      - summary
      - description (first comment)
      - text (summary + description)
      - component (label di BugBug, livello 2)
      - macro_component (prod principale, livello 1: Core, Firefox, ...)
    """
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / "bugbug_dataset.jsonl"

    print("[Naplace] Downloading Mozilla bugs via Bugbug...")
    db.download(bugzilla.BUGS_DB)


    classes, class_names = component.ComponentModel().get_labels()
    data = []

    kept, skipped_no_label, skipped_no_macro = 0, 0, 0

    for bug in bugzilla.get_bugs():
        bug_id = bug["id"]


        if bug_id not in classes:
            skipped_no_label += 1
            continue

        label = classes[bug_id]
        macro = _macro_from_label(label)
        if macro is None:
            skipped_no_macro += 1
            continue

        summary = bug.get("summary", "") or ""
        comments = bug.get("comments", []) or []
        description = comments[0]["text"] if comments else ""


        text = f"{summary}\n\n{description}".strip()

        data.append(
            {
                "id": bug_id,
                "summary": summary,
                "description": description,
                "text": text,
                "component": label,
                "macro_component": macro,
            }
        )
        kept += 1

    with dest.open("w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[Naplace] Saved {kept} bugs to {dest}")
    print(f"  Skipped (no label): {skipped_no_label}")
    print(f"  Skipped (no macro): {skipped_no_macro}")

    return dest
