# Exoplanets Clustering

Progetto di machine learning per l'analisi di un dataset NASA sugli esopianeti. La pipeline combina audit dei dati, preprocessing, clustering non supervisionato, interpretazione dei cluster e classificazione supervisionata.

## Struttura

```text
ExoplanetsClustering/
|-- input/
|   `-- nasa_exoplanet_intelligence.csv
|-- notebook/
|   |-- 01_Audit_LogTransform_Preprocessing.ipynb
|   |-- 02_EDA_and_Feature_Space.ipynb
|   |-- 03_Clustering_Core.ipynb
|   |-- 04_Cluster_Interpretation.ipynb
|   |-- 05_Supervised_Leakage_Aware_Classification.ipynb
|   |-- 06_Report_Tables_and_Figures.ipynb
|   |-- data/processed/
|   `-- reports/
|       |-- figures/
|       |-- final_figures/
|       |-- tables/
|       `-- final_tables/
|-- Report_Selvaggi_Roberto_Pio.pdf
|-- Selvaggi_Roberto_Pio_ML.pptx
|-- requirements.txt
`-- README.md
```

## Notebook

1. Audit, trasformazioni logaritmiche, imputazione e standardizzazione.
2. Analisi esplorativa e definizione degli spazi di feature.
3. Confronto degli algoritmi di clustering e selezione della configurazione.
4. Profilazione e interpretazione dei cluster mediante variabili esterne.
5. Classificazione supervisionata con split e valutazioni progettati per limitare il leakage.
6. Consolidamento delle tabelle e delle figure usate nel report.

Gli output elaborati sono salvati in `notebook/data/processed/`; tabelle e figure sono raccolte in `notebook/reports/`.

## Esecuzione

Da PowerShell, nella directory del progetto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:MPLBACKEND = "Agg"

<<<<<<< HEAD
Get-ChildItem .\notebook\*.ipynb | Sort-Object Name | ForEach-Object {
    .\.venv\Scripts\python.exe -m jupyter nbconvert `
        --to notebook --execute $_.FullName --inplace `
        --ExecutePreprocessor.timeout=1200
=======
Quindi eseguire i notebook in sequenza:

```powershell
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
>>>>>>> de03577a200018dc2bbf26d8ce96562e7625ebc6
}
```

I notebook devono essere eseguiti nell'ordine numerico indicato dai nomi dei file.
