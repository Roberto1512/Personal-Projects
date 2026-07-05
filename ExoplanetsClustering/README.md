# Exoplanet ML Project

Versione del progetto di Machine Learning sugli esopianeti.

## Struttura attuale

- `input/`
  - dataset raw originale: `nasa_exoplanet_intelligence.csv`
- `notebook_final/`
  - pipeline;
  - notebook 01-06;
  - output processati, tabelle e figure finali;
  - documentazione tecnica in `README_notebook_final.md`.
- `REPORT_FINALE_ESAME.md`
  - report finale.
- `requirements.txt`
  - dipendenze Python 3.11 dirette e versionate per la riproducibilita'.
- `.venv/`
  - ambiente Python locale usato per eseguire i notebook; non e' necessario includerlo nella consegna.


## Come rieseguire la pipeline

Creare l'ambiente e installare le dipendenze dalla root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Quindi eseguire i notebook in sequenza:

```powershell
cd C:\Users\rober\Desktop\UNI\ML\ProgettoML\notebook_final
$env:MPLBACKEND='Agg'
$nbs = @(
  '01_Audit_LogTransform_Preprocessing.ipynb',
  '02_EDA_and_Feature_Space.ipynb',
  '03_Clustering_Core.ipynb',
  '04_Cluster_Interpretation.ipynb',
  '05_Supervised_Leakage_Aware_Classification.ipynb',
  '06_Report_Tables_and_Figures.ipynb'
)
foreach ($nb in $nbs) {
  ..\.venv\Scripts\python.exe -m nbconvert --to notebook --execute $nb --inplace --ExecutePreprocessor.timeout=1200
}
```

## Output principali

- `notebook_final/reports/final_tables/`
- `notebook_final/reports/final_figures/`

La versione finale contiene 16 tabelle CSV, un riepilogo JSON e 12 figure consolidate, incluse il confronto esterno post-hoc tramite Adjusted Rand Index, l'audit/sensitivity analysis del cluster estremo e le analisi su intervalli di confidenza Macro-F1, repeated cross-validation e corrected resampled t-test per il Setup B supervisionato.
