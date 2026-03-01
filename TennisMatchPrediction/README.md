# Tennis Match Prediction

## Descrizione

Progetto di **Machine Learning** per la predizione probabilistica dell'esito di incontri di tennis professionistici (circuito ATP). Il sistema costruisce feature ingegnerizzate (Elo rating, head-to-head, form recente, statistiche di superficie) a partire da dati storici e allena una batteria di 7 classificatori, combinati infine in un **Stacked Generalization Ensemble**.

---

## Pipeline Completa

Il progetto è organizzato in **4 step** sequenziali:

### Step 1 — Feature Engineering & Baseline (`baseline.py`)

- Caricamento del dataset ATP (`data/atp_tennis.csv`, ~8.5 MB).
- **Sistema Elo custom**: rating dinamico per ogni giocatore con fattore K modulato dal prestigio del torneo (Grand Slam, Masters 1000, ecc.).
- Costruzione di **feature avanzate** (escluse le quote dei bookmaker — "Setting A"):
  - `rank_diff`, `rank_points_diff` — differenza di ranking e punti.
  - `elo_1`, `elo_2`, `elo_diff` — rating Elo calcolato in-house.
  - `h2h_p1_wins`, `h2h_p2_wins`, `h2h_diff` — storico head-to-head.
  - `form_diff`, `streak_diff` — forma recente e streak di vittorie/sconfitte.
  - One-hot encoding di `Surface` e `Court` (Indoor/Outdoor).
- **Split temporale**: Train (< 2015), Validation (2015), Test (2016+).
- Baseline con **Logistic Regression** + **Rank-only baseline**.
- **Metriche**: Accuracy, AUC-ROC, Log-Loss, Brier Score.
- **Output**: `outputs/step1_features.parquet`, `outputs/feature_cols.joblib`.

### Step 2 — Model Comparison & Ablation (`models_step2.py`)

Addestramento e confronto di **7 famiglie di modelli** con grid search sugli iperparametri:

| Modello | Iperparametri Testati |
|---|---|
| Logistic Regression | C ∈ {0.01, 0.1, 1.0, 10.0} |
| k-NN | n_neighbors ∈ {5, 15, 31, 51}, weights ∈ {uniform, distance} |
| Decision Tree | max_depth ∈ {3, 5, 8, 12, None} |
| Random Forest | n_estimators ∈ {100, 300}, max_depth ∈ {8, 15, None} |
| SVM (Linear) | C ∈ {0.01, 0.1, 1.0, 10.0} |
| SVM (RBF) | C ∈ {0.1, 1.0, 10.0}, gamma ∈ {scale, 0.01} |
| Naive Bayes | var_smoothing ∈ {1e-9, 1e-7, 1e-5} |

- **Ablation study** su 4 gruppi di feature (A: Rank+Surface, B: +H2H/Form, C: +Elo, D: All).
- **Test di McNemar** per significatività statistica tra coppie di modelli.
- **Output**: `outputs/models/best_*.joblib` (7 modelli), `outputs/step2_*.csv`.

### Step 3 — Evaluation & Visualization (`evaluation.py`)

Script completo di valutazione del miglior modello sul test set:

- **Bootstrap Confidence Intervals** (2000 iterazioni, 95% CI) su tutte le metriche.
- **Calibration Plot**: curva di calibrazione per tutti i modelli.
- **ROC Curve**: curve ROC sovrapposte con AUC per ogni modello.
- **Feature Importance**: bar chart delle top-15 feature per i modelli tree-based (Decision Tree, Random Forest).
- **Output**: grafici PNG (`calibration_plot.png`, `roc_curve.png`, `feature_importance_*.png`), `step3_test_metrics_with_ci.txt`.

### Step 4 — Stacked Generalization Ensemble (`stacking.py`)

Contributo algoritmico originale:

1. Caricamento di tutti i **7 modelli base** addestrati nello Step 2.
2. Generazione di **meta-feature**: la probabilità P(Player_1 vince) predetta da ciascun modello base.
3. Addestramento di un **meta-learner** (Logistic Regression con StandardScaler) sulle meta-feature del validation set, con tuning del parametro C ∈ {0.01, 0.1, 1.0, 10.0, 100.0}.
4. Valutazione dell'ensemble stacking vs. il miglior modello singolo.
5. Analisi dei **coefficienti del meta-learner** per interpretare il contributo di ogni modello base.
- **Output**: `outputs/stacked_model.joblib`, `outputs/step4_stacking_results.csv`, `outputs/step4_meta_coefficients.csv`.

---

## Struttura del Progetto

```text
.
├── data/
│   └── atp_tennis.csv              # Dataset storico ATP (~8.5 MB, ~65k match)
├── src/
│   ├── baseline.py                 # Step 1: Feature engineering + Elo + baseline
│   ├── models_step2.py             # Step 2: 7 modelli + grid search + ablation
│   ├── evaluation.py               # Step 3: Bootstrap CI + grafici calibrazione/ROC
│   └── stacking.py                 # Step 4: Stacked Generalization Ensemble
├── outputs/
│   ├── step1_features.parquet      # Feature matrix completa
│   ├── feature_cols.joblib         # Lista colonne feature
│   ├── best_model.joblib           # Miglior modello singolo
│   ├── stacked_model.joblib        # Modello ensemble stacking
│   ├── models/                     # 7 modelli base (best_*.joblib)
│   ├── step2_*.csv                 # Risultati grid search, ablation, test
│   ├── step3_test_metrics_with_ci.txt  # Metriche con intervalli di confidenza
│   ├── step4_*.csv                 # Risultati stacking + coefficienti meta-learner
│   ├── calibration_plot.png        # Grafico di calibrazione
│   ├── roc_curve.png               # Curve ROC
│   ├── feature_importance_*.png    # Feature importance (tree-based)
│   ├── report_draft.md             # Bozza report
│   ├── report_final.md             # Report finale
│   ├── slides_outline.md           # Outline presentazione
│   └── slides_final.md             # Presentazione finale
└── requirements.txt                # Dipendenze Python
```

---

## Dipendenze

```
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
joblib
pyarrow
```

## Esecuzione della Pipeline

```bash
# Step 1: Feature engineering + baseline
python src/baseline.py --csv data/atp_tennis.csv --out_dir outputs

# Step 2: Model comparison + ablation
python src/models_step2.py --features outputs/step1_features.parquet --out_dir outputs

# Step 3: Evaluation + visualizations
python src/evaluation.py --features outputs/step1_features.parquet --out_dir outputs

# Step 4: Stacked ensemble
python src/stacking.py --features outputs/step1_features.parquet --out_dir outputs
```
