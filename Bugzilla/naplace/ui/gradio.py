from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr


def _to_dict(pred) -> Dict[str, Any]:
    d = pred.model_dump()
    if d.get("probability") is None:
        d["probability"] = None
    return d


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_inference(model_name: str, text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"error": "Inserisci un testo (bug summary/description)."}

    # Lazy import: evita di caricare TensorFlow/SetFit all'avvio della UI
    # (molto utile in container/Spaces) :contentReference[oaicite:1]{index=1}
    from naplace.api.inference import predict_gru, predict_lstm, predict_setfit, predict_tfidf

    if model_name == "setfit":
        preds = predict_setfit([text])
    elif model_name == "gru":
        preds = predict_gru([text])
    elif model_name == "lstm":
        preds = predict_lstm([text])
    elif model_name == "tfidf":
        preds = predict_tfidf([text])
    else:
        return {"error": f"Modello non supportato: {model_name}"}

    return _to_dict(preds[0]) if preds else {"error": "Nessuna predizione."}


def _make_curl(model_name: str, text: str) -> str:
    base = os.getenv("NAPLACE_BASE_URL", "http://localhost:8000")
    endpoint = f"{base}/predict/{model_name}"
    payload = {"texts": [{"text": (text or "").strip()}]}
    body = json.dumps(payload, ensure_ascii=False)
    # Markdown code block (Gradio lo renderizza bene)
    return (
        "```bash\n"
        f'curl -X POST "{endpoint}" -H "Content-Type: application/json" -d \'{body}\'\n'
        "```"
    )


def _predict_ui(
    model_name: str, text: str, save: bool
) -> Tuple[Dict[str, Any], List[List[Any]], str]:
    result = _run_inference(model_name, text)

    summary: List[List[Any]] = []
    if "error" not in result:
        summary = [[result.get("predicted_label"), result.get("probability")]]

    curl_snippet = _make_curl(model_name, text)

    if save and "error" not in result:
        labels_path = Path(os.getenv("NAPLACE_LABELS_PATH", "data/labels/manual_labels.jsonl"))
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "input_text": result.get("input_text", (text or "").strip()),
            "predicted_label": result.get("predicted_label"),
            "probability": result.get("probability"),
        }
        _append_jsonl(labels_path, record)
        result["saved_to"] = str(labels_path)

    return result, summary, curl_snippet


def build_gradio_app() -> gr.Interface:
    title = "Naplace — Bug Component Classifier"
    description = (
        "Inserisci un bug report (summary/description) e scegli il modello.\n\n"
        "Endpoint API disponibili: `/predict/setfit`, `/predict/gru`, `/predict/lstm`, `/predict/tfidf`.\n"
        "Questa UI è montata su FastAPI in `/label`."
    )
    article = (
        "Note:\n"
        "- `title`, `description`, `article` rendono la demo leggibile (come richiesto nelle slide Gradio).\n"
        "- “Save” appende una riga JSONL in `NAPLACE_LABELS_PATH`.\n"
        "- Lo snippet `curl` usa `NAPLACE_BASE_URL` (default: http://localhost:8000)."
    )

    examples = [
        ["setfit", "Crash when opening settings panel", False],
        ["gru", "UI freezes after clicking save button", False],
        ["lstm", "Unexpected behavior when switching tabs", False],
        ["tfidf", "Crash when opening settings panel", False],
    ]

    return gr.Interface(
        fn=_predict_ui,
        inputs=[
            gr.Dropdown(choices=["setfit", "gru", "lstm", "tfidf"], value="setfit", label="Model"),
            gr.Textbox(lines=4, placeholder="Scrivi qui il bug report...", label="Bug report"),
            gr.Checkbox(value=False, label="Save prediction to JSONL (manual labels file)"),
        ],
        outputs=[
            gr.JSON(label="Prediction (raw)"),
            gr.Dataframe(headers=["predicted_label", "probability"], label="Prediction (summary)"),
            gr.Markdown(label="Use via API"),
        ],
        title=title,
        description=description,
        article=article,
        examples=examples,
        flagging_mode="never",
    )
