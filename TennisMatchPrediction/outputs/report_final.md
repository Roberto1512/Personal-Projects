# Report: Tennis Match Prediction (Setting A)

**Author:** [Your Name/ID]  
**Date:** [Date]

## 1. Motivation & Problem Definition

L'obiettivo di questo progetto è predire l'esito di un match di tennis professionistico maschile (ATP) prima che esso abbia inizio. Il problema è formulato come un task di **classificazione binaria supervisionata**: stimare la probabilità che il Giocatore 1 (`Player_1`) vinca il match, ovvero calcolare $P(\text{Winner} == \text{Player\_1})$.

Come stabilito dal protocollo **Setting A**, è imposto un vincolo fondamentale: **le quote offerte dai bookmaker (`Odd_1`, `Odd_2`) NON sono state incluse come feature predittive**. Lo scopo è valutare quanto un modello possa essere accurato basandosi unicamente su informazioni sportive storiche, evitando il "signal leakage" intrinseco nelle quote di mercato.

Un'altra rigorosa *leakage policy* ha imposto l'esclusione dal dataset di ogni metrica *post-match* (incluso lo `Score`). Tutte le feature dinamiche sono state calcolate utilizzando esclusivamente lo storico degli eventi conclusi *prima* del match in esame.

## 2. Dataset & Preprocessing

Il dataset analizzato (`atp_tennis.csv`) contiene lo storico dei match ATP dal 2000 ad oggi. Per mitigare la massiccia assenza di dati sui punti ATP nei primi anni, i match pre-2006 sono stati usati come "burn-in" per inizializzare le memorie storiche dei giocatori (forma, head-to-head, rating). L'addestramento e la validazione dei modelli si concentrano quindi sui match disputati a partire dal **2006**.

### Gestione Dati Mancanti e Multicollinearità
Alcuni valori mancanti critici (es. ranking o punti codificati come `-1`) sono stati riconvertiti in `NaN` per evitare di introdurre del rumore sistematico nel training. 
L'imputazione finale è stata eseguita utilizzando la **mediana calcolata unicamente sul Training set**, garantendo l'assenza di data leakage verso il Validation e il Test set. Inoltre, per rimuovere la multicollinearità spaziale, le colonne assolute (Rank, Pts) sono state scartate in favore dei soli valori differenziali matematici pre-calcolati (`rank_diff`, `pts_diff`).

## 3. Methods: Feature Engineering & Models 

### Feature Engineering (Pre-Match Only)
- **Feature Statiche:** Differenza di ranking ATP (`rank_diff`), differenza di punti (`pts_diff`), superficie e tipologia della competizione codificati in formato ad-hoc e scalati ordinalmente o come vettori base OHE (e.g. `Surface`, `Series`), format del match (`Best of` 3 o 5 sets).
- **Feature Dinamiche Cumulative (Time-Series):** Scansionando i match in rigoroso ordine cronologico, sono state calcolate feature dinamiche per profilare lo stato fisico e competitivo del singolo tennista la mattina dell'incontro:
  - **Forma (Last 5):** il win-rate basato in percentuale sugli ultimi 5 match.
  - **Streak:** numero cumulativo di vittorie ininterrotte prima dell'evento.
  - **Head-to-Head (H2H):** lo score totale di scontri diretti storicamente vinti/persi dai due atleti interessati.
  - **Elo Rating:** un rating matematico autolibrato. L'aggiornamento (K-factor) viene amplificato qualora il match in chiusura fosse un Master o Grand Slam (ad esempio applicando pesi correttivi). Ad ogni step viene ricavato l' `elo_diff` dei due contendenti.

### Model Selection
In aderenza all'iter stabilito nel corso, sono state sviluppate 7 pipeline, dotate tutte di `StandardScaler` basato su statistiche di stima dal train, per testare famiglie di learner diverse:
1. **Rank Baseline:** un'euristica grezza dove vince sempre il giocatore con ranking migliore. 
2. **Logistic Regression:** solida baseline GLS in grado di fornire natural calibration di log-odds probabilistici e interpretabilità immediata dei coefficienti.
3. **K-Nearest Neighbors (k-NN)**
4. **Alberi e foreste:** `Decision Tree` e `Random Forest`.
5. **Support Vector Machines:** Testate con kernel `Linear` e non-lineare `RBF`.
6. **GaussianNB (Naive Bayes):** come approccio generativo basato sull'ipotesi Bayesiana condizionale.

Le topologie sono state rifinite tramite una ricerca degli iperparametri su griglia mirata (grid search), escludendo preventivamente configurazioni prone a overfit manifesto.

## 4. Experimental Setup

Garantire indipendenza temporale è imperativo. Lo split `shuffle=False` implementa uno split cronologico hold-out conforme:
- **Training Set:** 2006–2021 (circa 39,500 istanze per apprendere).
- **Validation Set:** 2022–2023 (impiegato per hyperparameter tuning rigido e selection tra baseline).
- **Test Set:** 2024–2025 (mantenuto incorrotto fino al test one-shot finale).

La Model Selection validazionale si basa sulla metrica regina della probabilità di classe: il **Log-Loss** (Cross-Entropy). Trarre benefici in Accuracy bruta senza calibrazione logaritmica solida indicherebbe un modello incerto, punito dalla loss. Sono state estratte ed osservate in background ulteriori metriche probabilistiche standardizzate e soglia-indipendenti: ROC-AUC e il Brier Score.

## 5. Results & Discussion

### Model Selection e Tuning
Durante l'analisi su Validation Set, la **Logistic Regression** ($C=0.1$) si è imposta sulle altre topologie minimizzando la threshold di errore del Log-Loss ($~0.612$) e superando le non-linearità parametriche per robustezza e stabilità su grandi numeriche. 

L'analisi di **Feature Ablation** conferma l'enorme apporto dell'ingegneria feature sviluppata: la base sola con Ranking produce AUC peggiori in fase logaritmica. L'aggiunta del gruppo H2H+Forma, seguito dall'innesto dell'**Elo Rating dinamico**, funge da spinta preponderante verso i minimi globali calcolati per il Log-Loss. Estraendo la diagnostica *Feature Importances* dalla Random Forest addestrata in parallelo su tutte le features, `elo_diff` si proietta nettamente come la variabile più discriminante dell'intero dataset (Gini ~ 0.28, distanziando `rank_diff` bloccato in terza posizione a 0.15).

### Valutazione Statistica (Test Set 2024-2025)
Sul test-set puro (one-shot selection run), la *Logistic Regression* attesta i seguenti misuratori probabilistici estesi:
- **Log-Loss:** $0.616$
- **AUC:** $0.714$
- **Accuracy:** $64.7\%$

Per dimostrare formale validità empirica secondo i canoni d'esame, i valori di cui sopra sono stati estratti unitamente a *Confidence Interval (CI)* al $95\%$ derivati da un solido loop resampler di **Bootstrap (2000 iterazioni)**, tracciando con esattezza lo spettro di variance atteso [AUC $95\%$ CI: $0.69 – 0.73$].

Uno step confermativo aggiuntivo si è avuto implementando il **Calibration Plot** multiclasse sul best model in tandem con gli avversari di griglia. La Log-Reg staziona incredibilmente fedele ed in aderenza stretta alla retta "Perfectly Calibrated" speculare indicando che al $P = 70\%$ predetto matematicamente da output per una classe, tale cluster riscontrava fedelmente tasso winner appoggiato di 7 contro 3 in campo nella realtà, dimostrando immensa bontà dell'assunto predittivo calibrato e la stabilità assunta dal Logistic framework in questo binario senza overfitting.

Il **Test di McNemar** evidenzia a corollario finale che la Logistic Regression differisce metricamente, con significatività statistica assoluta empirica ($p < 0.05$), dalla heuristica basica Rank Baseline testata alla pari su matrix confusion.

### Limiti e Conclusioni
Il progetto prova che un framework *puramente sportivo* pre-match (Setting A, blindato da inquinamenti di betting market) riesce con profitto a predire l'esito reale a frequenze competitive stabili, sfruttando la memoria a lungo termine creata artificialmente con profili storici ELO. Le discrepanze in Accuracy a massimali sub-$70\%$ non esibiscono un limite del learning, ma rappresentano fedelmente il tetto oggettivo (the ceiling) intrinseco della casualità fisica dello sport d'élite (e.g. malus fisici inattesi, reintro in circuito faticoso, forfait), irrisolvibile ed impalpabile dai dati anagrafici pre-start. 
Lavori e integrazioni futuri per varcare il limite non potranno sottrarsi all'inclusione forzata del "Mercato Scommesse Open" come predictor pesato, oppure inquadrare l'acquisizione stream e time-series delle giocate reali punto-su-punto live nel set di feature.
