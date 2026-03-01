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

### 2. Dataset di Immagini

Le immagini raccolte (da web scraping e/o da altre fonti) sono state suddivise in **3 categorie** per l'addestramento multi-classe:

| Cartella | Contenuto |
|---|---|
| `dataset_armi/` | Immagini di armamenti e equipaggiamento militare |
| `dataset_bandiere/` | Immagini di bandiere, loghi e simboli di organizzazioni |
| `dataset_terroristi/` | Immagini di soggetti d'interesse (riconoscimento facciale/pattern) |

### 3. Pipeline di Addestramento (`pipeline/`)

Cuore del progetto: contiene il notebook e i modelli addestrati.

#### Notebook

- **`pipeline.ipynb`** (~14 KB): Notebook Jupyter che implementa la pipeline completa di training e valutazione. Include preprocessing, data augmentation, definizione dell'architettura della rete neurale, training loop e metriche di valutazione.

#### Modelli Addestrati (`pipeline/models/`)

Tre modelli **PyTorch** (`.pt`) pre-addestrati, uno per ciascuna categoria di classificazione:

| File | Dimensione | Task |
|---|---|---|
| `best_flags.pt` | ~19.6 MB | Classificazione bandiere/loghi |
| `best_people.pt` | ~19.6 MB | Riconoscimento soggetti |
| `best_weapons.pt` | ~19.6 MB | Classificazione armamenti |

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
├── dataset_armi/                           # Dataset immagini: armamenti
├── dataset_bandiere/                       # Dataset immagini: bandiere e loghi
├── dataset_terroristi/                     # Dataset immagini: soggetti d'interesse
├── pipeline/
│   ├── pipeline.ipynb                      # Notebook principale: training + valutazione
│   ├── images/                             # Immagini di supporto/visualizzazione
│   ├── models/                             # Modelli PyTorch addestrati
│   │   ├── best_flags.pt
│   │   ├── best_people.pt
│   │   └── best_weapons.pt
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

- **Python 3**, **PyTorch** (reti neurali convoluzionali)
- **BeautifulSoup** + **requests** (web scraping)
- **Jupyter Notebook** (sperimentazione e pipeline)
