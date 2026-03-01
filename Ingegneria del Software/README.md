# The Verifiers — Ingegneria del Software

## Descrizione

Applicazione web full-stack dedicata al **fact-checking** e alla verifica di contenuti digitali (notizie, domini, immagini). Sviluppata nell'ambito del corso di **Ingegneria del Software** dal team "The Verifiers".

La piattaforma espone due profili distinti — **Utente** e **Amministratore** — ciascuno con viste e funzionalità proprie.

---

## Tecnologie Utilizzate

| Componente | Tecnologia |
|---|---|
| **Backend** | Python 3 — Flask |
| **Database** | MySQL (`theverifiersdb`) tramite `flask_mysqldb` |
| **Frontend** | HTML5 / CSS3 / JavaScript — Bootstrap 5 |
| **Sicurezza** | Hashing password con SHA-256 (`hashlib`) |
| **Image Analysis** | Script esterno `Image-Manipulation-Detection/analyze.py` (invocato via `subprocess`) |
| **ML Model** | `model/model_c1.pth` — modello PyTorch serializzato (~57 MB) |

---

## Funzionalità nel Dettaglio

### Area Utente (14 viste)

| Route | Funzionalità |
|---|---|
| `/` | Home page utente (o redirect home admin se l'utente è admin) |
| `/registration` | Registrazione nuovo utente (nome, cognome, email, password, telefono) |
| `/log_in` | Login con email + password (SHA-256) |
| `/logout` | Logout e pulizia della sessione |
| `/recovery` | Pagina di recupero password |
| `/gestioneProfilo` | Visualizzazione dati del profilo personale |
| `/modificaProfilo` | Modifica password (verifica vecchia + impostazione nuova) |
| `/elimina_profilo` | Cancellazione definitiva dell'account |
| `/verifica` | Hub di verifica: dominio e immagine |
| `/verificaDominio` | Controllo attendibilità di un URL contro il database dei domini verificati |
| `/verificaImmagine` | Upload immagine → analisi manipolazioni tramite script Python esterno |
| `/notizieVerificate` | Consultazione elenco notizie verificate dall'admin |
| `/segnalazioneNotizie` | Segnalazione di un URL sospetto agli amministratori |
| `/subscription`, `/contatti`, `/faq` | Pagine informative e di contatto |

### Area Amministratore (6 viste)

| Route | Funzionalità |
|---|---|
| `/homeAdmin` | Dashboard amministratore |
| `/log_inAdmin` | Login admin con ID + password |
| `/gestioneNotizie` | Aggiunta manuale di notizie verificate (URL) |
| `/gestioneSegnalazioni` | Revisione segnalazioni utenti: possibilità di aggiungere alla blacklist o eliminare |
| `/gestioneDomini` | CRUD completo sui domini verificati: aggiunta, rimozione singola/multipla |
| `/gestioneForm` , `/senza_risposta` | Lettura e risposta ai form/messaggi inviati dagli utenti |

### Schema Database (`theverifiersdb`)

Tabelle principali dedotte dal codice:

- `credenzialiUtenti` — (nome, cognome, email, password, phone)
- `credenzialiamministratori` — (id, password)
- `dominiverificati` — (URL)
- `notizieaggiunte` — (URL)
- `notiziesegnalate` — (URL)
- `blacklist` — (URL)
- `form` — (id, nome, email, oggetto, messaggio, risposta)

---

## Struttura del Progetto

```text
.
├── app.py                         # Entry point Flask (tutte le route)
├── model/
│   └── model_c1.pth               # Modello PyTorch pre-addestrato (~57 MB)
├── templates/
│   ├── login.html                 # Pagina di login condivisa utente/admin
│   ├── viewsAdmin/                # 6 template HTML per l'area admin
│   │   ├── homeAdmin.html
│   │   ├── gestioneDomini.html
│   │   ├── gestioneForm.html
│   │   ├── gestioneNotizie.html
│   │   ├── gestioneSegnalazioni.html
│   │   └── rispondereAiForm.html
│   └── viewsUtente/               # 14 template HTML per l'area utente
│       ├── homeUtente.html
│       ├── registration.html
│       ├── gestioneProfilo.html
│       ├── modificaProfilo.html
│       ├── verifica.html
│       ├── dominiVerificati.html
│       ├── notizieVerificate.html
│       ├── segnalazioneNotizie.html
│       ├── filtro.html
│       ├── form.html
│       ├── subscription.html
│       ├── contatti.html
│       ├── faq.html
│       └── recovery.html
├── static/
│   ├── css/                       # Bootstrap 5 + CSS custom (main.css, style.css, popUp.css, ...)
│   ├── js/                        # Bootstrap bundle + JS custom (main.js, script.js, popUp.js, ...)
│   ├── fonts/                     # Font tipografici (~73 file)
│   ├── images/                    # Asset immagini
│   └── vendor/                    # Librerie di terze parti
├── temp/                          # Directory temporanea per upload immagini
└── pyvenv.cfg                     # Configurazione virtual environment
```

---

## Requisiti e Avvio

1. **MySQL**: Avviare un server MySQL locale e creare il database `theverifiersdb` con le tabelle sopra descritte.
2. **Ambiente Python**:
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   pip install flask flask_mysqldb
   ```
3. **Avvio**:
   ```bash
   python app.py
   ```
4. Accedere a **http://localhost:5000/**.
