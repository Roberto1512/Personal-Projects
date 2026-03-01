# 🧠 NLP Toxicity Classifier WebApp

## 📁 Struttura del progetto

```
NLP/
├── app.py                  # Backend Flask
├── index.html              # Interfaccia web (form HTML)
├── requirements.txt        # Pacchetti Python richiesti
├── README.md               # Questo file
└── best_toxic_model/       # Modello BERT fine-tuned salvato

## ⚙️ Setup ambiente virtuale in Visual Studio Code (VSC)

### 1. Apri la cartella `NLP/` in VSC

### 2. Da terminale su VSC svolgere gli altri punti 

          ## 📌 Dipendenze principali
          - Flask
          - torch
          - transformers
          - accelerate
          
          Installabili via:
          pip install -r requirements.txt
          
          ## 🚀 Avvio dell'applicazione
          
          1. Assicurati che la cartella `best_toxic_model/` contenga il modello salvato da HuggingFace.
          
          2. Runna app.py (va bene anche direttamente da vsc in alto a destra facendo run project):
              Assicurati di avere l'ambiente pronto (venv con percorso a py installato sul tuo pc
                  
                  python app.py

3. Apri il browser e visita:
http://127.0.0.1:5000

4. Scrivi una conversazione nella textarea e clicca su "Analizza"
   - Se tossica, verrà mostrata anche una **spiegazione generata da Mistral** 

