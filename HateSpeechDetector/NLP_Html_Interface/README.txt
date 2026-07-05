# NLP Toxicity Classifier Web App

Interfaccia Flask per classificare una conversazione come tossica o non tossica tramite un modello Transformer salvato localmente.

## Struttura

```text
NLP_Html_Interface/
|-- app.py
|-- index.html
|-- requirements.txt
|-- logo.png
|-- best_toxic_model/
`-- README.txt
```

La directory `best_toxic_model/` deve contenere configurazione, tokenizer e pesi compatibili con `AutoModelForSequenceClassification`.

## Installazione e avvio

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

L'applicazione è disponibile su `http://127.0.0.1:5000`. L'endpoint `POST /predict` accetta un oggetto JSON con il campo `text` e restituisce etichetta e probabilità di tossicità.
