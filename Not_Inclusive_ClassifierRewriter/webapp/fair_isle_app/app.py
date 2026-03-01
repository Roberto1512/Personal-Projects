import os
import re
import json
import html
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template, request, jsonify, redirect
import torch
from transformers import pipeline


# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

SUPPORTED_LANGS = ("en", "it")
DEFAULT_LANG = "en"
LANG_COOKIE = "fairisle_lang"

MODEL_SPECS = {
    "en": {
        "clf_dir": "model_classifier_en",
        "s2s_dir": "model_inclusive_rewriter_en",
        "prefix": "Rewrite the sentence using inclusive language: ",
    },
    "it": {
        "clf_dir": "model_classifier_it",
        "s2s_dir": "model_inclusive_rewriter_it",
        # ✅ prefix "buono" (come nella versione vecchia che funzionava)
        "prefix": (
            "[INCLUSIVO] "
            "Riformula la frase in modo inclusivo e rispettoso, mantenendo il tema. "
            "Non introdurre il tema del genere se non è presente. "
            "Output: una sola frase. Testo: "
        ),
    },
}

MODELS_CACHE = {}


# ------------------ Feedback storage (single JSON) ------------------
FEEDBACK_DIR = Path(BASE_DIR) / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_FILE = FEEDBACK_DIR / "feedback.json"
if not FEEDBACK_FILE.exists():
    FEEDBACK_FILE.write_text("[]", encoding="utf-8")


# ------------------ Device ------------------
DEVICE = 0 if torch.cuda.is_available() else -1


# ------------------ Load label map (optional) ------------------
def load_label_map(clf_dir: str):
    label_map_path = os.path.join(clf_dir, "label_map.json")
    if not os.path.exists(label_map_path):
        return None
    with open(label_map_path, "r", encoding="utf-8") as f:
        tmp = json.load(f)
    try:
        return {int(k): v for k, v in tmp.items()}
    except Exception:
        return None


def normalize_label(raw_label: str, id2label) -> str:
    if raw_label is None:
        return "unknown"
    lab = raw_label.strip()
    m = re.match(r"^LABEL_(\d+)$", lab)
    if m and id2label is not None:
        idx = int(m.group(1))
        return id2label.get(idx, lab)
    return lab


LABEL_ALIASES = {
    "inclusiva": "inclusive",
    "non_inclusiva": "not_inclusive",
    "non inclusiva": "not_inclusive",
    "non-inclusiva": "not_inclusive",
}


def canonicalize_label(label: str) -> str:
    if not label:
        return "unknown"
    lab = label.strip().lower()
    return LABEL_ALIASES.get(lab, lab)


# ------------------ Models ------------------
def load_models(lang: str):
    lang = lang if lang in SUPPORTED_LANGS else DEFAULT_LANG
    if lang in MODELS_CACHE:
        return MODELS_CACHE[lang]

    spec = MODEL_SPECS[lang]
    clf_dir = os.path.join(MODELS_DIR, spec["clf_dir"])
    s2s_dir = os.path.join(MODELS_DIR, spec["s2s_dir"])

    if not os.path.isdir(clf_dir):
        raise FileNotFoundError(f"Missing classifier dir: {clf_dir}")
    if not os.path.isdir(s2s_dir):
        raise FileNotFoundError(f"Missing rewriter dir: {s2s_dir}")

    id2label = load_label_map(clf_dir)

    clf = pipeline(
        "text-classification",
        model=clf_dir,
        tokenizer=clf_dir,
        device=DEVICE,
        truncation=True,
        max_length=256,
    )

    rewriter = pipeline(
        "text2text-generation",
        model=s2s_dir,
        tokenizer=s2s_dir,
        device=DEVICE,
    )

    MODELS_CACHE[lang] = {
        "clf": clf,
        "rewriter": rewriter,
        "id2label": id2label,
        "prefix": spec["prefix"],
    }
    return MODELS_CACHE[lang]


def normalize_lang(raw: str, fallback: str = DEFAULT_LANG) -> str:
    if raw:
        candidate = raw.strip().lower()
        if candidate in SUPPORTED_LANGS:
            return candidate
    return fallback


def get_current_lang() -> str:
    return normalize_lang(request.cookies.get(LANG_COOKIE), DEFAULT_LANG)


def get_texts(lang: str) -> dict:
    return TEXTS.get(lang, TEXTS[DEFAULT_LANG])


CONFIDENCE_THRESHOLD = 0.60
MAX_INPUT_CHARS = 50000


TEXTS = {
    "en": {
        "header": {
            "writing_assistant": "Writing Assistant",
            "language_label": "Language",
        },
        "footer": {
            "tagline": "A Writing Assistant Tool for Inclusive Italian/English Language",
            "byline": "by Rosita Maglie, Francesca Filograsso, Lucia Siciliani, Pierpaolo Basile. With the support of",
            "oss_prefix": "This project re-uses open source code from the",
            "oss_suffix": "project.",
        },
        "labels": {
            "inclusive": "Inclusive",
            "not_inclusive": "Not inclusive",
            "not_pertinent": "Not pertinent",
            "suggestion": "Suggestion",
        },
        "index": {
            "subtitle": "An AI-assisted writing tool to detect and correct non-inclusive language.",
            "what_it_does_title": "What it does",
            "what_it_does_body": (
                "FAIR-ISLE analyzes your text sentence by sentence and highlights potentially non-inclusive expressions. "
                "When needed, it suggests more inclusive rewrites you can review and adapt."
            ),
            "detect_label": "Detect",
            "detect_detail": "Non-inclusive expressions",
            "detect_example": "Example: a sentence that may contain biased or non-inclusive wording.",
            "rewrite_label": "Rewrite",
            "rewrite_detail": "Inclusive alternative",
            "rewrite_example": "Example: a clearer, more inclusive version of the same sentence.",
            "cta": "Try the Writing Assistant",
            "pipeline_title": "Pipeline",
            "pipeline_detection_label": "Detection:",
            "pipeline_detection_desc": "identifies non-inclusive language.",
            "pipeline_rewriting_label": "Rewriting:",
            "pipeline_rewriting_desc": "proposes inclusive reformulations.",
            "pipeline_feedback_label": "Feedback:",
            "pipeline_feedback_desc": "collects user ratings to improve the system.",
            "languages_title": "Languages",
            "lang_it": "Italian",
            "lang_en": "English",
            "good_to_know_title": "Good to know",
            "good_to_know_body": "Suggestions are generated automatically and may require human review depending on context.",
        },
        "testing": {
            "title": "Writing Assistant",
            "subtitle": "Detect and correct non-inclusive language with AI assistance.",
            "original_text": "ORIGINAL TEXT",
            "clear": "Clear",
            "placeholder": "Type or paste your text here to begin analysis.",
            "tip": "Tip: the system analyzes sentence by sentence.",
            "analyze": "Analyze Text",
            "suggestions": "SUGGESTIONS",
            "structured_results": "Structured results",
            "output_placeholder": "The analyzed text and suggestions will appear here.",
            "legend_not_inclusive": "Not inclusive",
            "legend_inclusive": "Inclusive",
            "feedback_title": "Feedback",
            "feedback_q1": "Is the rewritten sentence correct?",
            "feedback_q2": "Is the sentence classified correctly?",
            "yes": "Yes",
            "no": "No",
            "proposed_label": "Your proposed rewrite",
            "proposed_placeholder": "Write your improved rewrite here...",
            "send_feedback": "Send Feedback",
            "modal_saved": "Saved.",
        },
        "evaluation": {
            "title": "Human Evaluation and Annotation",
            "intro": (
                "This page aims at collecting feedback from expert users on the quality of the classification and rewriting models. "
                "The input text can be inserted on the left side of the page and the output text will be displayed on the right side of the page."
            ),
            "input_label": "Input",
            "input_placeholder": "Insert the full text to be analyzed here.",
            "submit_button": "Submit",
            "human_feedback_label": "Human Feedback",
            "no_text_submitted": "No text submitted yet.",
            "color_coding_title": "Color coding:",
            "inclusive_explain": "The sentence is inclusive.",
            "not_inclusive_explain": "The sentence is not inclusive.",
            "not_pertinent_explain": "The sentence is not containing any ambiguous expression.",
            "js": {
                "classification_not_correct": "&nbsp;Classification is <b>not</b> correct",
                "classification_feedback": "Classification feedback",
                "not_inclusive": "Not inclusive",
                "inclusive": "Inclusive",
                "not_pertinent": "Not pertinent",
                "rewriting_correct": "&nbsp;Rewriting is correct&nbsp;&nbsp;",
                "rewriting_not_correct": "&nbsp;Rewriting is <b>not</b> correct",
                "rewriting_placeholder": "Insert possible rewriting",
                "submit_feedback": "Submit feedback",
                "no_data": "No data returned from the server.",
                "submit_error": "Error during the submission of the feedback.",
            },
        },
        "messages": {
            "no_text_submitted": "No text submitted yet.",
            "text_too_long": "Text too long (max {max} chars).",
            "no_text_provided": "No text provided.",
            "request_failed": "Request failed. Please try again.",
            "feedback_saved": "Feedback saved.",
            "feedback_failed": "Could not save feedback.",
            "rewrite_choice_missing": "Please select Yes or No for the rewrite question.",
            "classification_choice_missing": "Please select Yes or No for the classification question.",
            "proposed_required": "If you selected No, please provide your proposed rewrite.",
            "invalid_rewrite_correct": "Invalid rewrite_correct (use yes/no).",
            "invalid_classification_correct": "Invalid classification_correct (use yes/no).",
            "no_system_rewrite": "No system rewrite/target sentence to evaluate.",
            "proposed_required_server": "Proposed rewrite is required when rewrite is not correct.",
            "failed_to_save_feedback": "Failed to save feedback: {error}",
            "language_not_supported": "Language '{language}' not supported. Use {supported}.",
        },
    },
    "it": {
        "header": {
            "writing_assistant": "Assistente di scrittura",
            "language_label": "Lingua",
        },
        "footer": {
            "tagline": "Uno strumento di scrittura per un linguaggio inclusivo in italiano e inglese",
            "byline": "di Rosita Maglie, Francesca Filograsso, Lucia Siciliani, Pierpaolo Basile. Con il supporto di",
            "oss_prefix": "Questo progetto riutilizza codice open source del",
            "oss_suffix": "progetto.",
        },
        "labels": {
            "inclusive": "Inclusivo",
            "not_inclusive": "Non inclusivo",
            "not_pertinent": "Non pertinente",
            "suggestion": "Suggerimento",
        },
        "index": {
            "subtitle": "Uno strumento di scrittura con IA per rilevare e correggere il linguaggio non inclusivo.",
            "what_it_does_title": "Cosa fa",
            "what_it_does_body": (
                "FAIR-ISLE analizza il testo frase per frase e mette in evidenza espressioni potenzialmente non inclusive. "
                "Quando necessario, suggerisce riscritture piu inclusive che puoi rivedere e adattare."
            ),
            "detect_label": "Rileva",
            "detect_detail": "Espressioni non inclusive",
            "detect_example": "Esempio: una frase che puo contenere formulazioni parziali o non inclusive.",
            "rewrite_label": "Riscrive",
            "rewrite_detail": "Alternativa inclusiva",
            "rewrite_example": "Esempio: una versione piu chiara e inclusiva della stessa frase.",
            "cta": "Prova l'assistente di scrittura",
            "pipeline_title": "Pipeline",
            "pipeline_detection_label": "Rilevamento:",
            "pipeline_detection_desc": "identifica il linguaggio non inclusivo.",
            "pipeline_rewriting_label": "Riscrittura:",
            "pipeline_rewriting_desc": "propone riformulazioni inclusive.",
            "pipeline_feedback_label": "Feedback:",
            "pipeline_feedback_desc": "raccoglie valutazioni degli utenti per migliorare il sistema.",
            "languages_title": "Lingue",
            "lang_it": "Italiano",
            "lang_en": "Inglese",
            "good_to_know_title": "Da sapere",
            "good_to_know_body": "I suggerimenti sono generati automaticamente e possono richiedere una revisione umana in base al contesto.",
        },
        "testing": {
            "title": "Assistente di scrittura",
            "subtitle": "Rileva e corregge il linguaggio non inclusivo con l'aiuto dell'IA.",
            "original_text": "TESTO ORIGINALE",
            "clear": "Cancella",
            "placeholder": "Scrivi o incolla qui il testo per iniziare l'analisi.",
            "tip": "Suggerimento: il sistema analizza frase per frase.",
            "analyze": "Analizza il testo",
            "suggestions": "SUGGERIMENTI",
            "structured_results": "Risultati strutturati",
            "output_placeholder": "Qui appariranno il testo analizzato e i suggerimenti.",
            "legend_not_inclusive": "Non inclusivo",
            "legend_inclusive": "Inclusivo",
            "feedback_title": "Feedback",
            "feedback_q1": "La frase riscritta risulta corretta?",
            "feedback_q2": "La classificazione della frase risulta corretta?",
            "yes": "Si",
            "no": "No",
            "proposed_label": "La tua proposta di riscrittura",
            "proposed_placeholder": "Scrivi qui la tua riscrittura migliorata...",
            "send_feedback": "Invia feedback",
            "modal_saved": "Salvato.",
        },
        "evaluation": {
            "title": "Valutazione e annotazione umana",
            "intro": (
                "Questa pagina raccoglie feedback da utenti esperti sulla qualita dei modelli di classificazione e riscrittura. "
                "Il testo di input puo essere inserito a sinistra e il testo di output verra mostrato a destra."
            ),
            "input_label": "Input",
            "input_placeholder": "Inserisci qui il testo completo da analizzare.",
            "submit_button": "Invia",
            "human_feedback_label": "Feedback umano",
            "no_text_submitted": "Nessun testo ancora inviato.",
            "color_coding_title": "Codifica colori:",
            "inclusive_explain": "La frase risulta inclusiva.",
            "not_inclusive_explain": "La frase risulta non inclusiva.",
            "not_pertinent_explain": "La frase non contiene espressioni ambigue.",
            "js": {
                "classification_not_correct": "&nbsp;La classificazione non risulta corretta",
                "classification_feedback": "Feedback classificazione",
                "not_inclusive": "Non inclusivo",
                "inclusive": "Inclusivo",
                "not_pertinent": "Non pertinente",
                "rewriting_correct": "&nbsp;La riscrittura risulta corretta&nbsp;&nbsp;",
                "rewriting_not_correct": "&nbsp;La riscrittura non risulta corretta",
                "rewriting_placeholder": "Inserisci una possibile riscrittura",
                "submit_feedback": "Invia feedback",
                "no_data": "Nessun dato restituito dal server.",
                "submit_error": "Errore durante l'invio del feedback.",
            },
        },
        "messages": {
            "no_text_submitted": "Nessun testo ancora inviato.",
            "text_too_long": "Testo troppo lungo (max {max} caratteri).",
            "no_text_provided": "Nessun testo fornito.",
            "request_failed": "Richiesta non riuscita. Riprova.",
            "feedback_saved": "Feedback salvato.",
            "feedback_failed": "Impossibile salvare il feedback.",
            "rewrite_choice_missing": "Seleziona Si o No per la domanda sulla riscrittura.",
            "classification_choice_missing": "Seleziona Si o No per la domanda sulla classificazione.",
            "proposed_required": "Se hai selezionato No, inserisci la tua proposta di riscrittura.",
            "invalid_rewrite_correct": "Valore non valido per rewrite_correct (usa yes/no).",
            "invalid_classification_correct": "Valore non valido per classification_correct (usa yes/no).",
            "no_system_rewrite": "Nessuna riscrittura/frase target da valutare.",
            "proposed_required_server": "La riscrittura proposta e richiesta quando quella di sistema non e corretta.",
            "failed_to_save_feedback": "Impossibile salvare il feedback: {error}",
            "language_not_supported": "Lingua '{language}' non supportata. Usa {supported}.",
        },
    },
}


# ------------------ Helpers ------------------
def split_sentences_simple(text: str):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]


def clean_rewrite_output(raw: str, original: str = "", lang: str = "en") -> str:
    if not raw:
        return ""

    s = raw.strip()

    # 1) rimuovi "Suggestion:" / "Suggerimento:" ovunque compaia
    s = re.sub(r"(?i)\b(suggestion|suggerimento)\s*:\s*", "", s).strip()

    # 2) se il modello ricomincia a ripetere la consegna, tronca il rumore
    if lang == "it":
        pat = r"\b(riscrivi\s+la\s+frase|riformula\s+la\s+frase)\b"
    else:
        pat = r"\b(rewrite\s+the\s+sentence)\b"

    m = re.search(pat, s, flags=re.IGNORECASE)
    if m:
        if m.start() <= 2:
            s = s[m.end():].strip(" \t\n:-")
        else:
            s = s[:m.start()].strip()

    # 3) elimina eventuali code tipo "usando un linguaggio inclusivo"
    s = re.sub(r"(?i)\busando\s+un\s+linguaggio\s+inclusivo\b\s*:?\s*", "", s).strip()
    s = re.sub(r"(?i)\busing\s+inclusive\s+language\b\s*:?\s*", "", s).strip()

    # 4) se ci sono più righe, tieni l'ultima significativa
    lines = [x.strip() for x in s.splitlines() if x.strip()]
    if lines:
        s = lines[-1].strip()

    # 5) se l'output ripete l'originale all'inizio, rimuovilo
    if original:
        o = original.strip().lower()
        sl = s.strip().lower()
        if sl.startswith(o):
            s = s[len(original.strip()):].strip(" \t\n:-")

    # 6) normalizza spazi
    s = re.sub(r"\s+", " ", s).strip()

    return s


def classify_sentence(s: str, model_bundle: dict):
    pred = model_bundle["clf"](s)[0]
    raw_label = normalize_label(pred.get("label", "unknown"), model_bundle["id2label"])
    label = canonicalize_label(raw_label)
    score = float(pred.get("score", 0.0))
    return label, score


def rewrite_sentence(s: str, model_bundle: dict, lang: str):
    raw = model_bundle["rewriter"](
        model_bundle["prefix"] + s,
        max_new_tokens=96,          # ✅ come la versione vecchia
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        length_penalty=1.0,
        early_stopping=True,
    )[0]["generated_text"]

    return clean_rewrite_output(raw, original=s, lang=lang)


def analyze_text_structured(text: str, lang: str):
    model_bundle = load_models(lang)
    out = []
    for s in split_sentences_simple(text):
        label, score = classify_sentence(s, model_bundle)
        final_label = label
        rw = ""

        if score < CONFIDENCE_THRESHOLD:
            final_label = "not_pertinent"
        elif label == "not_inclusive":
            rw = rewrite_sentence(s, model_bundle, lang)

        out.append({
            "sentence": s,
            "label": final_label,
            "score": score,
            "rewrite": rw
        })
    return out


def pick_feedback_target(analysis_rows):
    for row in analysis_rows:
        if (row.get("label") == "not_inclusive") and (row.get("rewrite") or "").strip():
            return {
                "sentence": row.get("sentence", ""),
                "system_rewrite": row.get("rewrite", ""),
                "label": row.get("label", ""),
                "score": float(row.get("score", 0.0)),
            }
    return None


def build_output_html(analysis_rows, labels: dict):
    parts = []
    for row in analysis_rows:
        s = row.get("sentence", "")
        label = (row.get("label", "unknown") or "unknown").lower()
        score = float(row.get("score", 0.0))
        rewrite_txt = row.get("rewrite", "") or ""

        css = {
            "inclusive": "inclusive",
            "not_inclusive": "not_inclusive",
            "not_pertinent": "not_pertinent",
        }.get(label, "not_pertinent")

        s_html = html.escape(s)
        rw_html = html.escape(rewrite_txt) if rewrite_txt else ""

        status_label = {
            "inclusive": labels.get("inclusive", "Inclusive"),
            "not_inclusive": labels.get("not_inclusive", "Not inclusive"),
            "not_pertinent": labels.get("not_pertinent", "Not pertinent"),
        }.get(css, css)

        badge_class = {
            "inclusive": "result-badge result-badge--inclusive",
            "not_inclusive": "result-badge result-badge--not-inclusive",
            "not_pertinent": "result-badge result-badge--not-pertinent",
        }.get(css, "result-badge")

        item_class = {
            "inclusive": "result-item result-item--inclusive",
            "not_inclusive": "result-item result-item--not-inclusive",
            "not_pertinent": "result-item result-item--not-pertinent",
        }.get(css, "result-item")

        score_pct = f"{score * 100:.0f}%"

        suggestion_label = labels.get("suggestion", "Suggestion")
        suggestion_block = (
            f'<div class="result-suggestion"><b>{suggestion_label}:</b> {rw_html}</div>'
            if rewrite_txt
            else ""
        )

        parts.append(
            f'<div class="{item_class}">'
            f'  <div class="result-meta">'
            f'    <span class="{badge_class}">{status_label}</span>'
            f'    <span class="result-score">{score_pct}</span>'
            f'  </div>'
            f'  <div class="result-sentence {css}">{s_html}</div>'
            f'  {suggestion_block}'
            f'</div>'
        )
    return "\n".join(parts)


def load_feedback_list():
    try:
        raw = FEEDBACK_FILE.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = FEEDBACK_DIR / f"feedback_corrupt_{ts}.json"
        try:
            FEEDBACK_FILE.replace(backup)
        except Exception:
            pass
        FEEDBACK_FILE.write_text("[]", encoding="utf-8")
        return []


def append_feedback(payload: dict):
    data = load_feedback_list()
    data.append(payload)
    tmp = FEEDBACK_DIR / ".feedback.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, FEEDBACK_FILE)


# ------------------ Flask App ------------------
app = Flask(__name__)


@app.context_processor
def inject_language():
    lang = get_current_lang()
    return {"lang": lang, "t": get_texts(lang)}


@app.route("/set_language", methods=["POST"])
def set_language():
    lang = normalize_lang(request.form.get("lang"), DEFAULT_LANG)
    next_url = request.form.get("next") or "/"
    resp = redirect(next_url)
    resp.set_cookie(LANG_COOKIE, lang, max_age=60 * 60 * 24 * 365, samesite="Lax")
    return resp


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/testing", methods=["GET"])
def testing():
    return render_template("testing.html", output_html=None)


@app.route("/submit_testing", methods=["POST"])
def submit_testing():
    lang = normalize_lang(request.form.get("language"), get_current_lang())
    t = get_texts(lang)
    input_text = (request.form.get("input_text") or "").strip()

    if not input_text:
        return _testing_response(
            output_html=(
                '<div class="small" style="color:#6b7280; font-weight:700;">'
                + html.escape(t["messages"]["no_text_submitted"])
                + "</div>"
            ),
            target=None
        )

    if len(input_text) > MAX_INPUT_CHARS:
        msg = t["messages"]["text_too_long"].format(max=MAX_INPUT_CHARS)
        return jsonify({"message": msg}), 413

    analysis = analyze_text_structured(input_text, lang)
    output_html = build_output_html(analysis, t["labels"])
    target = pick_feedback_target(analysis)

    return _testing_response(output_html=output_html, target=target)


@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    lang = normalize_lang(request.form.get("language"), get_current_lang())
    t = get_texts(lang)

    rewrite_correct = (request.form.get("rewrite_correct") or "").strip().lower()  # yes|no
    proposed_rewrite = (request.form.get("proposed_rewrite") or "").strip()
    classification_correct = (request.form.get("classification_correct") or "").strip().lower()  # yes|no

    input_text = (request.form.get("input_text") or "").strip()

    target_label = (request.form.get("target_label") or "").strip()
    target_score = (request.form.get("target_score") or "").strip()
    target_sentence = (request.form.get("target_sentence") or "").strip()
    system_rewrite = (request.form.get("system_rewrite") or "").strip()

    if not input_text:
        return jsonify({"ok": False, "message": t["messages"]["no_text_provided"]}), 400
    if len(input_text) > MAX_INPUT_CHARS:
        msg = t["messages"]["text_too_long"].format(max=MAX_INPUT_CHARS)
        return jsonify({"ok": False, "message": msg}), 413

    if rewrite_correct not in {"yes", "no"}:
        return jsonify({"ok": False, "message": t["messages"]["invalid_rewrite_correct"]}), 400

    if classification_correct not in {"yes", "no"}:
        return jsonify({"ok": False, "message": t["messages"]["invalid_classification_correct"]}), 400

    if not system_rewrite or not target_sentence:
        return jsonify({"ok": False, "message": t["messages"]["no_system_rewrite"]}), 400

    if rewrite_correct == "no" and not proposed_rewrite:
        return jsonify({"ok": False, "message": t["messages"]["proposed_required_server"]}), 400

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_text": input_text,
        "language": lang,
        "target": {
            "label": target_label,
            "score": float(target_score) if target_score else None,
            "sentence": target_sentence,
            "system_rewrite": system_rewrite,
        },
        "rewrite_correct": (rewrite_correct == "yes"),
        "proposed_rewrite": proposed_rewrite if rewrite_correct == "no" else "",
        "classification_correct": (classification_correct == "yes"),
    }

    try:
        append_feedback(payload)
        count = len(load_feedback_list())
    except Exception as e:
        msg = t["messages"]["failed_to_save_feedback"].format(error=str(e))
        return jsonify({"ok": False, "message": msg}), 500

    return jsonify({"ok": True, "count": count}), 200


def _testing_response(output_html: str, target):
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"output_HTML": output_html, "target": target}), 200
    return render_template("testing.html", output_html=output_html), 200


@app.route("/fairisle", methods=["GET", "POST"])
def fairisle_api():
    if request.method == "GET":
        return jsonify({"message": "Running...", "error_code": 0})

    data = request.get_json(silent=True)
    if data is None:
        if request.data:
            return jsonify({"message": "Invalid JSON", "error_code": 102}), 400
        data = {}

    text = (data.get("text") or "").strip()
    language = (data.get("language") or DEFAULT_LANG).lower()
    t = get_texts(language if language in SUPPORTED_LANGS else DEFAULT_LANG)

    if not text:
        return jsonify({"message": t["messages"]["no_text_provided"], "error_code": 100})

    if len(text) > MAX_INPUT_CHARS:
        msg = t["messages"]["text_too_long"].format(max=MAX_INPUT_CHARS)
        return jsonify({"message": msg, "error_code": 103}), 413

    if language not in SUPPORTED_LANGS:
        msg = t["messages"]["language_not_supported"].format(
            language=language,
            supported=", ".join(SUPPORTED_LANGS),
        )
        return jsonify({"message": msg, "error_code": 101})

    out = analyze_text_structured(text, language)
    return jsonify(out), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
