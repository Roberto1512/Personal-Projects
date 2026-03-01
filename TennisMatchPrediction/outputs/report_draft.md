# Appendice C: Report Tennis Match Prediction (Setting A)

**Author:** [Your Name/ID]  
**Date:** [Date]

## 1. Motivation & Problem Definition
Il problema analizzato in questo elaborato consiste nella predizione dell'esito di un match di tennis professionistico maschile (ATP). Il task è formulato come classificazione binaria supervisionata: predire la probabilità che il giocatore 1 (`Player_1`) vinca il match, ovvero stimare $P(\text{Winner} == \text{Player\_1})$.

Come stabilito dal protocollo **Setting A**, è imposto un vincolo progettuale cruciale: **le quote offerte dai bookmaker (`Odd_1`, `Odd_2`) non sono state incluse come feature predittive**. Questo per misurare l'efficacia predittiva pura derivante esclusivamente dalle caratteristiche intrinseche e storiche dello sport, misurando quanto distante ci si possa posizionare senza il "signal leakage" che deriverebbe dall'informazione di mercato.

Un'altra severa strict policy riguardante il *data leakage* ha imposto di escludere dal set di feature ogni dato *post-match* (l'annotazione puntuale dello score). Ogni feature derivata è stata rigorosamente processata con informazioni aggiornate *fino al giorno precedente* il match in esame.

## 2. Dataset & Preprocessing
Il dataset di lavoro, `atp_tennis.csv`, contiene uno storico dal $2000$ ad oggi di match ATP.
Per via dell'assenza massiva di punti per il ranking pre-$2006$, i dati pregressi sono stati adoperati unicamente per l'inizializzazione e il warmup delle feature storiche, mentre set operativi (*split evaluation*) sono stati filtrati considerando i match dal 2006 in poi.

### Gestione Dati Mancanti e Imputazione
I missing values nei rank o points, originariamente codificati a `-1`, sono stati riallineati a `NaN` per evitare rumore.
L'imputazione dei dati mancanti è stata effettuata utilizzando la mediana delle distribuzioni.  
*Nota bene:* Coerentemente con le direttive del corso, la statistica mediana è calcolata esclusivamente sul **Train set** per evitare il data leakage verso Validation e Test.

## 3. Methods: Feature Engineering & Models 
### Feature Engineering
Le feature create sfruttano esclusivamente informazioni *pre-match* e storiche.
*   **Feature Statiche:** Differenza di ranking (`rank_diff`), differenza punti (`pts_diff`), superficie codificata ad one-hot ('Surface'), e tipologia di torneo tradotta in ordinal factor (`Series`, `Round`).
*   **Feature Dinamiche Cumulative:** Calcolate iterativamente scansionando i match cronologicamente:
    *   **Forma (Last 5):** Win-rate parziale del giocatore calcolato sugli ultimi 5 suoi match giocati.
    *   **Streak:** Numero cumulativo di vittorie consecutive.
    *   **Head-to-Head (H2H):** Bilancio pre-match delle vittorie dirette tra i due analizzati.
    *   **Elo Rating (Dinamic):** Parametro dinamico base (start $1500$) per i due player e l' `elo_diff`, con l'aggiunta di un learning rate `k` modulato dal prestigio della competizione in atto.

### Model Selection
Sono state esaminate un bacino di ipotesi rappresentative degli algoritmi trattati a lezione:
1.  **Baseline Non Parametrica (Rank-based):** Predice probabilità superiori per il player avvantaggiato dal differenziale nominale del ranking ATP (classificazione deterministica della classe maggiormente rankata come winner).
2.  **Logistic Regression:** Baseline ML per classificazione e stima log-odds calibrate naturalmente.
3.  **K-Nearest Neighbors (k-NN):** Approccio non standard-parametrico con esplorazione del differenziale k-density.
4.  **Tree-based e Ensemble:** Introverso l'utilizzo di `Decision Trees` e `Random Forest` (quest'ultimo con numero limitato di profondità per evitare l'estrazione di regole spuri).
5.  **Support Vector Machines (Linear & RBF):** Per tracciare confini a largo margine negli embeddings numerici scalati dell'ecosistema, calibrabili in esecuzione log-loss.

Tutte le pipeline che contengono l'ausilio di distance-metrics e gradient descent incorporano nativamente uno step di Standardizzazione (`StandardScaler`) calcolata sempre e solo sul Train set.

## 4. Experimental Setup
L'holdout method standardizzato non casuale (`shuffle=False`) è la condizione sine qua non dettata dalla dipendenza cronologica delle serie prodotte, e la formulazione suggerita e applicata è stata:
*   **Training Set:** Anni dal $2006$ al $2021$.
*   **Validation Set:** Anni dal $2022$ al $2023$ (Model e HPs selection).
*   **Test Set:** Anni dal $2024$ al $2025$ (Riservato esclusivamente alla singola one-shot finale).

Il criterio d'oro per la target metric di Model Selection definita e osservata sul Validation data è l'impiego del **Log Loss**, coerentemente supportando il goal di stimare calibrazioni di probabilità prima di class decision (Threshold-free metric).
A cui si accompagnano Brier Score, *ROC-AUC*, e la semplice, ma non bilanciata in soglia, *Accuracy*.

## 5. Results & Discussion
La comparazione in validation set mostra la propensione del framework lineare (Logistic Regression / SVM Lineare) ad approcciare minimi eccellenti di loss. 

All'applicazione del modello principe scelto (i.e Logistic Regression per miglior bilanciamento log-loss\interpretatività nel panorama *without-odds*) all'area temporale *one-shot di Test Set*, vengono estratti intervalli statistici di robustezza usando l'attestato protocollo per calcolo test CI **al 95% via tecnica di Bootstrap Resampling** su 1000 iteration resampled con reimmissione.

I risultati dimostrano che i modelli costruiti senza supporti alle *wisdom of crowd* di quote sfiorano accuracy comparativamente performanti rispetto alle predizioni a secco, riuscendosi ad estendere ed evadere parzialmente il basamento piatto standard imposto dalla cruda superiorità di ATP Ranking come da preteso dalle feature dinamiche cumulative come *L'Elo System* e *H2H*.
Limitazioni visibili derivano dalla volatilità della classe di varianza insita nello sport (assenza del coverage per assenze prolungate, non disponibilità di physical tracking, etc), ma è difendibile e inquadrato dai correttori di calibrazione e CI analizzati.
