# Tesi Computer Vision + Machine Learning

[Tesi_Davide_Poli (3).pdf](https://github.com/user-attachments/files/18218760/Tesi_Davide_Poli.3.pdf)

## Descrizione

Progetto di **Tesi di Laurea** incentrato su tecniche di **Computer Vision** e **Machine Learning** applicate al riconoscimento e alla classificazione automatica di immagini in ambito sicurezza/intelligence. Il sistema è in grado di identificare e categorizzare immagini relative ad armamenti, bandiere/loghi e soggetti d'interesse.

---

## Componenti del Progetto

### 1. Acquisizione Dati — Web Scraping (`WebScraping.py`)

Script Python per il download automatico di immagini da fonti web:

- **Librerie**: `BeautifulSoup`, `requests`, `urllib.parse`.
- **Fonte**: Pagine del National Counterterrorism Center (NCTC) — `https://www.dni.gov/nctc/`.
- **Funzionamento**: Parsing HTML → estrazione tag `<img>` → costruzione URL assoluti → download e salvataggio locale delle immagini.

### 2. Notebook di Addestramento

Le tre directory di lavoro contengono i notebook usati per addestrare e valutare i modelli YOLO:

| Cartella | Contenuto |
|---|---|
| `dataset_armi/Training_Weapons.ipynb` | Armi ed equipaggiamento |
| `dataset_bandiere/Training_Flags.ipynb` | Bandiere, loghi e simboli |
| `dataset_terroristi/Training_Terrorist.ipynb` | Persone e soggetti d'interesse |

### 3. Pipeline di Addestramento (`pipeline/`)

Contiene il notebook di inferenza, immagini di esempio e risultati di valutazione.

#### Notebook

- **`pipeline.ipynb`**: carica i tre modelli YOLO, combina e filtra i rilevamenti, classifica il contesto e produce immagini annotate e risultati CSV. I file dei modelli non sono inclusi nella versione corrente del repository.

#### Metriche di Valutazione

Per ciascun modello sono salvate le metriche di performance in sotto-cartelle dedicate, suddivise per split (train/validation e test):

| Cartella | Contenuto |
|---|---|
| `pipeline/terrorist_flags_metrics/` | Performance del modello bandiere (train_valid + test) |
| `pipeline/terrorist_people_metrics/` | Performance del modello soggetti (train_valid + test) |
| `pipeline/terrorist_weapons_metrics/` | Performance del modello armamenti (train_valid + test) |

---

## Struttura del Progetto

```text
.
├── WebScraping.py                          # Script per il download automatico di immagini dal web
├── dataset_armi/
│   └── Training_Weapons.ipynb
├── dataset_bandiere/
│   └── Training_Flags.ipynb
├── dataset_terroristi/
│   └── Training_Terrorist.ipynb
├── pipeline/
│   ├── pipeline.ipynb                      # Inferenza, fusione rilevamenti e classificazione
│   ├── images/                             # Immagini di supporto/visualizzazione
│   ├── terrorist_flags_metrics/            # Metriche classificazione bandiere
│   │   ├── train_valid/
│   │   └── test/
│   ├── terrorist_people_metrics/           # Metriche riconoscimento soggetti
│   │   ├── train_valid/
│   │   └── test/
│   └── terrorist_weapons_metrics/          # Metriche classificazione armamenti
│       ├── train_valid/
│       └── test/
└── README.md
```

---

## Tecnologie

- **Python 3**, **PyTorch**, **Ultralytics YOLO**
- **BeautifulSoup** + **requests** (web scraping)
- **Jupyter Notebook** (sperimentazione e pipeline)
