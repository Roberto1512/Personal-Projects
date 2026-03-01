from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import json
import os
import re
import torch
import transformers

# ------------------ Paths ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                 # .../Semantic/training_code_en
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))          # .../Semantic
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

SUPPORTED_LANGS = ("en", "it")
DEFAULT_LANG = "en"

MODEL_SPECS = {
    "en": {
        "clf_dir": "model_classifier_en",
        "s2s_dir": "model_inclusive_rewriter_en",
        "prefix": "Rewrite the sentence using inclusive language: ",
    },
    "it": {
        "clf_dir": "model_classifier_it",
        "s2s_dir": "model_inclusive_rewriter_it",
        # prefix come nella versione vecchia
        "prefix": (
            "[INCLUSIVO] "
            "Riformula la frase in modo inclusivo e rispettoso, mantenendo il tema. "
            "Non introdurre il tema del genere se non è presente. "
            "Output: una sola frase. Testo: "
        ),
    },
}

CONFIDENCE_THRESHOLD = 0.60
MAX_INPUT_CHARS = 50000

MODELS_CACHE = {}

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
    if not raw_label:
        return "unknown"
    lab = raw_label.strip()
    if lab.startswith("LABEL_") and id2label is not None:
        suffix = lab[len("LABEL_"):]
        if suffix.isdigit():
            return id2label.get(int(suffix), lab)
    return lab


# canonical label mapping )
LABEL_ALIASES = {
    # italian -> canonical
    "inclusiva": "inclusive",
    "non_inclusiva": "not_inclusive",
    "non inclusiva": "not_inclusive",
    "non-inclusiva": "not_inclusive",
    # english canonical
    "inclusive": "inclusive",
    "not_inclusive": "not_inclusive",
}


def canonicalize_label(label: str) -> str:
    if not label:
        return "unknown"
    lab = label.strip().lower()
    return LABEL_ALIASES.get(lab, lab)


def clean_rewrite_output(raw: str, original: str = "", lang: str = "en") -> str:
    if not raw:
        return ""

    s = raw.strip()

    # 1)  "Suggestion:" / "Suggerimento:" 
    s = re.sub(r"(?i)\b(suggestion|suggerimento)\s*:\s*", "", s).strip()

    # 2) se il modello ripete
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

    # 3) elimina frasi tipo "usando un linguaggio inclusivo" / "using inclusive language" se restano 
    s = re.sub(r"(?i)\busando\s+un\s+linguaggio\s+inclusivo\b\s*:?\s*", "", s).strip()
    s = re.sub(r"(?i)\busing\s+inclusive\s+language\b\s*:?\s*", "", s).strip()

    # 4) se ci sono più righe, tieni l'ultima significativa
    lines = [x.strip() for x in s.splitlines() if x.strip()]
    if lines:
        s = lines[-1].strip()

    # 5) se l'output ripete l'originale all'inizio
    if original:
        o = original.strip().lower()
        sl = s.strip().lower()
        if sl.startswith(o):
            s = s[len(original.strip()):].strip(" \t\n:-")

    # 6) normalizza spazi
    s = re.sub(r"\s+", " ", s).strip()

    return s


# ------------------ Models ------------------
DEVICE = 0 if torch.cuda.is_available() else -1


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

    clf_pipeline = transformers.pipeline(
        "text-classification",
        model=clf_dir,
        tokenizer=clf_dir,
        device=DEVICE,
        truncation=True,
        max_length=256,
    )

    s2s_pipeline = transformers.pipeline(
        "text2text-generation",
        model=s2s_dir,
        tokenizer=s2s_dir,
        device=DEVICE,
    )

    MODELS_CACHE[lang] = {
        "clf": clf_pipeline,
        "rewriter": s2s_pipeline,
        "id2label": id2label,
        "prefix": spec["prefix"],
    }
    return MODELS_CACHE[lang]


def split_sentences_simple(text: str):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]


def classify(text: str, model_bundle: dict):
    return model_bundle["clf"](text)[0]


def rewrite(text: str, model_bundle: dict, lang: str):
    raw = model_bundle["rewriter"](
        model_bundle["prefix"] + text,
        max_new_tokens=96,         
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        length_penalty=1.0,
        early_stopping=True,
    )[0]["generated_text"]

    return clean_rewrite_output(raw, original=text, lang=lang)


app = Flask(__name__)
api = Api(app)


class FairisleService(Resource):
    def get(self):
        return jsonify({"message": "Running...", "error_code": 0})

    def post(self):
        data = request.get_json(silent=True)
        if data is None:
            if request.data:
                return jsonify({"message": "Invalid JSON", "error_code": 102}), 400
            data = {}

        text = (data.get("text") or "").strip()
        language = (data.get("language") or DEFAULT_LANG).lower()

        if not text:
            return jsonify({"message": "No text provided", "error_code": 100})

        if len(text) > MAX_INPUT_CHARS:
            return jsonify({
                "message": f"Text too long (max {MAX_INPUT_CHARS} chars).",
                "error_code": 103,
            }), 413

        if language not in SUPPORTED_LANGS:
            return jsonify({
                "message": f"{language} not supported here. Use language='en' or 'it'.",
                "error_code": 101,
            })

        model_bundle = load_models(language)

        out = []
        for s in split_sentences_simple(text):
            pred = classify(s, model_bundle)
            raw_label = normalize_label(pred.get("label", "unknown"), model_bundle["id2label"])
            label = canonicalize_label(raw_label)
            score = float(pred.get("score", 0.0))

            rw = ""
            if score < CONFIDENCE_THRESHOLD:
                label = "not_pertinent"
            elif label == "not_inclusive":
                rw = rewrite(s, model_bundle, language)

            out.append({"sentence": s, "label": label, "score": score, "rewrite": rw})

        return jsonify(out)


api.add_resource(FairisleService, "/fairisle")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8082, use_reloader=False)
