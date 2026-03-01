# Personal Projects Portfolio

Benvenuto nella repository dei miei **progetti personali e accademici**.

Questa raccolta documenta un percorso formativo che spazia dallo **Sviluppo Web e Mobile** all'**Intelligenza Artificiale** (Machine Learning, Deep Learning, NLP e Computer Vision), passando per la **Cybersecurity**, il **Project Management** e lo **UX/UI Design**.

Ogni sotto-cartella contiene un `README.md` con la descrizione dettagliata del progetto.

---

## Indice dei Progetti

### Machine Learning & Deep Learning

| Progetto | Descrizione | Tecnologie |
|---|---|---|
| [**BlastocystAnalyze**](./BlastocystAnalyze) | Pipeline di Deep Learning per la predizione precoce della vitalità embrionale da sequenze time-lapse. 3 task: classificazione binaria, regressione temporale (survival) e self-supervised contrastive learning (SimCLR). | PyTorch, ResNet18, LSTM, SimCLR |
| [**TennisMatchPrediction**](./TennisMatchPrediction) | Predizione probabilistica di match ATP. Feature engineering con Elo rating custom, confronto di 7 famiglie di modelli ML con grid search ed ablation study, ensemble finale tramite Stacked Generalization. | scikit-learn, pandas, matplotlib |
| [**Tesi Computer Vision + Machine Learning**](./Tesi%20Computer%20Vision%20%2B%20Machine%20Learning) | Tesi di Laurea su Computer Vision: web scraping per acquisizione dati, 3 dataset di immagini (armi, bandiere, soggetti), pipeline di classificazione multi-classe con reti neurali convoluzionali. | PyTorch, BeautifulSoup, Jupyter |

---

### NLP & MLOps

| Progetto | Descrizione | Tecnologie |
|---|---|---|
| [**Bugzilla**](./Bugzilla) | Architettura MLOps completa per la classificazione automatica di bug report Mozilla. API REST con 4 modelli (TF-IDF, GRU, LSTM, SetFit), pipeline DVC, monitoring Prometheus/Grafana, load testing Locust, CI/CD GitHub Actions. | FastAPI, PyTorch, Gradio, Docker, DVC, Prometheus |
| [**HateSpeech**](./HateSpeech) | **FAIR ISLE** — webapp per la detection e la riscrittura inclusiva di testi in italiano e inglese. Combina un classificatore Transformer e un rewriter Seq2Seq (FLAN-T5/IT5). Include modalità multilingue sperimentale (XLM-RoBERTa + mT5). | Flask, HuggingFace Transformers, BERTino, DistilBERT |

---

### The Verifiers — Ecosistema Progettuale

Cinque moduli interconnessi sviluppati nell'ambito di diversi corsi universitari per il progetto **"The Verifiers"**, una piattaforma di fact-checking digitale, e il progetto correlato **"FitMate"**:

| Progetto | Corso | Descrizione |
|---|---|---|
| [**Ingegneria del Software**](./Ingegneria%20del%20Software) | Ingegneria del Software | Webapp Flask/MySQL per fact-checking: verifica domini, analisi manipolazione immagini (PyTorch), gestione notizie/segnalazioni. 20+ template HTML, pannello admin completo. |
| [**Integrazione e Testing**](./Integrazione%20e%20Testing) | Integrazione e Testing del Software | Documentazione ITSS (piani di collaudo, test case, report), directory predisposte per test automatizzati. |
| [**Modelli e Metodi per la Qualità del Software**](./Modelli%20e%20Metodi%20per%20la%20Qualit%C3%A0%20del%20Software) | MMQS | Analisi qualità software: report finale, diario di bordo, 10 iterazioni di diagrammi di Gantt che documentano l'evoluzione della pianificazione. |
| [**Progettazione Interazione Utente**](./Progettazione%20Interazione%20Utente) | PIU | Ciclo completo di User-Centered Design: 7 interviste utenti, prototipi bassa/alta fedeltà, test di usabilità, presentazioni FitMate e The Verifiers. |
| [**Sviluppo Mobile Software**](./Sviluppo%20Mobile%20Software) | Sviluppo Mobile | App mobile **AsilApp**: documentazione tecnica, manuale utente, video demo, materiale marketing. |

---

### Cybersecurity

| Progetto | Descrizione | Tecnologie |
|---|---|---|
| [**Rans0mWar3 - CyberSecurity**](./Rans0mWar3%20-%20CyberSecurity) | Simulazione didattica dei meccanismi crittografici dei ransomware. Include un generatore di chiavi RSA-2048 e un decryptor ("Lazarus") per file cifrati con schema ibrido RSA + Fernet (AES). | Python, pycryptodome, cryptography |

---

## Tecnologie Principali

```
Python   •   Flask   •   FastAPI   •   PyTorch   •   scikit-learn   •   HuggingFace Transformers
MySQL    •   Docker   •   DVC   •   Prometheus   •   Gradio   •   Bootstrap 5   •   Jupyter
```
