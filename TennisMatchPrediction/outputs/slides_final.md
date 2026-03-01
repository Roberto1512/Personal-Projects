# Slides: Tennis Match Prediction (Setting A)
*(Estimated time: 20-25 minutes)*

## 1. Introduction (2 min)
- **Titolo:** Tennis Match Prediction: A Machine Learning framework for Pre-Match assessment.
- **Goal Principale:** Stimare preventivamente la probabilità di vittoria, basandosi esclusivamente su features raccolte *pre-match*.
- **The "Setting A" Constraint:** Divieto assoluto dell'uso delle quote di mercato (Odds).
  - *Perché?* Misura quanto si riesce ad estrarre informazione dal solo fattore sportivo contro i predittori di massa di mercato.
- **Leakage Policy:** Prevenire attivamente *future sight*.
  - *Score match* bandito.
  - Generazione di tutte le stats basata solo sui track storici di gioco fermati "al giorno prima dell'incontro in analisi".

## 2. Il Dataset e Anatomia (4 min)
- **Dataset:** `atp_tennis.csv` ($2000$ - Oggi).
- **Gestione missing temporali (-1):** Sostituzione dei Rank Point inclassificabili/assenti con `NaN` puri per scartare il rumore introdotto artificialmente ai modelli.
- **Burn-in storico:** I dati per l'addestramento sono traslati a partire dall'anno $2006$, per generare storie pregresse sature (Rank point popolati, memorie di tracking vittorie caricate).
- **Imputation strategy:** Uso imperativo della Statistica 'Mediana' basata *esclusivamente* sul Train dataset per schermare dal Data Leakage invisibile i Validation/Test Set spaccati temporaneamente.
- **Multicollinearità Curing:** Abbattimento delle features speculari pure (Rank_1, Rank_2 etc) usando esclusivamente Differenze Matematiche Relazionali (`rank_diff`).

## 3. L'Ingegneria delle Feature (6 min)
- **Statiche (Strutturali all'Evento):** Differenze calcolate di Punti ATP. Codifiche per tipologia Superficie e Ordine Serie Eventi (`ATP250 -> Grand Slam`), format del match (`Best of 3 vs 5`).
- **Cumulativi Storici (Time-Builders):**
  - *Current Form (5):* Rateo percentuale di win misurato sugli ultimi 5 match giocati dal tennista.
  - *Streak Win:* Contatore scalare ininterrotto di vittorie in corso.
  - *Head-to-Head (H2H):* Ratio scontri diretti passati risolti per cogliere le 'bestie-nere' del tabellone.
- **Il Cuore Differenziale: Sistema Elo Dinamico**
  - Base fissa autocalibrata.
  - K-factor Moltiplicativo: il fattore K di accrescimento rank Elo viene aumentato su tornei Elite (Grand Slam, Master1000) conferendo più inerzia e peso storico per il ranking interno ai giocatori d'alta fascia vera che over-performano nel lungo termine. Ottenimento `elo_diff`.

## 4. Modeling & Experimental Protocol (4 min)
*Strutturare uno Time Series Hold-out Split robusto.*
- **Training:** 2006-2021 (Apprendimento stabile profondo)
- **Validation:** 2022-2023 (Tuning HPs in Hyperparameter Grid Search e model selection finale)
- **Test Set:** 2024-2025 (Run asettica one-shot chiusa al futuro)
- **Candidate Models:**
  - *Euristiche e Baselines:* Rank-Difference Baseline
  - *Probability Kings:* Logistic Regression, GaussianNB (Generativo)
  - *Distanziometrici e Non Lineari:* SVM RBF/Linear, K-Nearest Neighbors (k-NN)
  - *Tree Builders:* Random Forest, Decision Tree
- **Criterio di Scelta Validazionale:** La minimizzazione della Cross Entropy Loss probabilistica (**Log-Loss**), garantendosi predizioni di trust probability.

## 5. Risultati e Statistiche (5 min)
- **Model Selection Validation Outcomes:** La *Logistic Regression ($C=0.1$)* incarna il best-model (Log Loss ~ $0.61$), imponendosi in robustezza su grandi test non overfittanti.
- **Ablation Study:** I grafici di *Random Forest Feature Importance* dimostrano la superiorità predittiva assoluta del feature core introdotto: `elo_diff` frantuma totalmente l'importanza nativa del `rank_diff`.
- **TEST SCORE - The Real Predictive Accuracy (2024-2025):** 
  - Predominanza One-Shot del Log Reg con **Accuracy: 64.7%**, **AUC: ~ 0.714**.
- **Robutness Assurance:**
  - *Bootstrap CI intervals (Test Set):* Ampiezza di varianza accertata via ricampionamento a 2000 iterazioni con CI 95% al ($0.63 - 0.66$ accuracy garantita e stabilizzata).
  - *Calibration Plot Review:* Plottaggio multiprovenzale della Logistic mostranteb curve in ricalco perfetto alla diagonale identità (`Perfectly Calibrated model` -> se stima P = 0.7 ci si aspetta e ritrova Win = 70% vero).
  - *Test Standard McNemar:* Provata indipendenza d'efficienza statistica contro classificazione basic (*P_value < 0.05*).

## 6. Conclusion & Limitations (2-3 min)
- **Conclusioni Principali:** Il modello puro pre-match Setting A dimostra formidabile validità competitiva (ceiling $\sim$ $65-68\%$ hard limit intrinseco mondiale per i soli fattori field sportivi disgiunti da market betting). L'ELO rating domina per qualità statistica sul target, surclassando i raw ranking ATP.
- **Limitazioni:** Il framework manca del fattore umano vitale dell'Open Market che incorpora variazioni micro (meteo, fatica, infortuni o ritorni improvvisi da pause lunghe dove ATP\ELO sfrecciano a zero di retention rate prediction prima che si assestino).
- **Future Works:** Espansioni verticali (reti Deep NLP integrate o LightGBM con hardware dedicato) e l'adozione delle Time Series inter-match (palla per palla, breakpoint recovery).
