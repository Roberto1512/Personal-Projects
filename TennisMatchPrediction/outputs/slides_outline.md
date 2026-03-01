# Slides: Tennis Match Prediction (Setting A)
*(Estimated time: 20-25 minutes)*

## 1. Titolo e Introduction (2 min)
- **Titolo:** Tennis Match Prediction: A Machine Learning framework for Pre-Match assessment.
- **Goal Principale:** Stimare la probabilità di vittoria, basandosi esclusivamente su features estratte *pre-match*.
- **The "Setting A" Constraint:** Divieto d'uso esplicito delle quote (odds) come feature. Obiettivo: quanto si riesce a estrarre informazione dal solo fattore sportivo contro fattori di mercato?
- **Leakage Policy:** Attenzione maniacale nel prevenire *future sight*. Esclusione dello Score dai fattori, e creazione statitiche rigorosamente "fino al match di ieri".

## 2. Il Dataset e L'Anatomia (4 min)
- **Dataset:** `atp_tennis.csv` ($2000$ - Oggi).
- **Problema originario:** Tanti nulls codificati col `-1` sulle stats originarie.
- **Risoluzione temporale:** Training shiftato per i modelli solo dal $2006$ in avanti permettendo il riempimento progressivo e il *burn-in* dello storico dei players.
- **Tecniche di Imputation:** Uso della mediana, statisticamente ancorata *esclusivamente* sul Train dataset per difendersi dal Leakage su Validation/Test Set emergenti dal tempo.

## 3. L'Ingegneria delle Feature (6 min)
- **Statiche (Immediatamente leggibili):** Differenze in Ranking ATP e Punti. Ordinalizzazione per pregio/tipologia Serie di eventi, Superficie in formato One-Hat.
- **Cumulativi Storici (Building Chronologically):**
  - *Current Form:* Rateo vittorie su N (ultimi 5 match).
  - *Streak Win:* Sequenze positive in atto.
  - *Head-to-Head:* Scontri diretti per carpire il fattore "Black-Beast" nei matchup testacoda.
- **Il Sistema Dinamico Elo:**
  - Aggiornamento della valutazione abilità Elo ad ogni iterazione, unito ad un parametro Learning (K) adattativo sui Grand Slam e Master.

## 4. Modeling & Experimental Protocol (4 min)
*Struttura allineata ad accademia temporale (Time Series Hold-out Split)*
- **Training:** 2006-2021
- **Validation:** 2022-2023 (Spazio esclusivo per Hyperparameter tuning ed euristiche)
- **Test:** 2024-2025 (Run finali one-shot del vincitore in solitaria)
- **Candidate Models:**
  - Rank Baseline (Euristic non-parametrica base).
  - Logistic Regression (Best-in-Class lineare e interpretatività log-odd).
  - Support Vector Machines, KNN, Random Forest, Decision Tree.
- **The Guiding Star Metric:** Uso marcato del Log-Loss per la Model Selection validazionale. Oltre alla stima accuracy bruta, quanto era decisa della predizione il modello? 

## 5. Risultati e Validation Selection (5 min)
- **Tuning Outcomes:** Comparazione tabellare dei modelli sul Log-loss e focus su AUC per inquadrare la threshold indipendence.
- **La stima puntuale sul TEST:** Output delle performance secche applicate sul futuro puro.
- **Reliability a testare:**
  - Plottaggio *Calibration curve* per determinare il confidence assessment sui prediction buckets (i.e Il modello sottostima gli sfavoriti?).
  - *Bootstrap CI intervals:* Estrapolazione e garanzia statistica con CI al 95% su 1000 iterazioni rimpiazzate, garantendo validità formale dell'Acc, log-loss, Brier metrics presentate.

## 6. Conclusion & Limitations (2-3 min)
- **Conclusioni Principali:** La prestazione puramente "Setting A" si dimostra una fortissima competenza pur restando un framework in solitaria e disarmato contro i super-informati mercati, dove subentrano componenti fisiologiche off-chart.
- **Limitazioni:** Fattori latenti non misurabili dal foglio (Variazione del tempo atmosferico improvviso, fatica latente da match recenti da 5 ore, infortuni o ritorni in forma imprevisti che alterano ranking ma non ELO rapido).
- **Future Works:** Espansioni che accolgono intra match analytics.
