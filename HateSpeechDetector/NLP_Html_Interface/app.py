from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from torch.nn.functional import softmax
import os

# === Inizializzazione Flask ===
app = Flask(__name__, static_folder='.', static_url_path='')

# === Percorsi e dispositivo ===
MODEL_DIR = "./best_toxic_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Caricamento modello di classificazione ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, local_files_only=True
).to(device)
model.eval()

def classify_toxic(text):
    enc = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding="max_length", max_length=256
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = softmax(logits, dim=1).cpu().numpy()[0]
    return float(probs[1])  # score tossicità

# === Homepage: serve index.html ===
@app.route("/")
def serve_index():
    return send_from_directory('.', 'index.html')

# === Endpoint API ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    score = classify_toxic(text)
    label = "Tossica" if score > 0.5 else "Non Tossica"
    explanation = "(Spiegazione non disponibile - modello Mistral disattivato)"
    return jsonify({
        "label": label,
        "score": score,
        "explanation": explanation
    })

# === Avvio ===
if __name__ == "__main__":
    app.run(debug=True)
