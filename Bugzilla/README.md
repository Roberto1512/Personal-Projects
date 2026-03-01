# Naplace
## Issue Report Classification w/Mozilla

FastAPI service for **Mozilla Bugzilla issue report classification** (component prediction), with:

-  **Multiple predictors** exposed via REST (`/predict/tfidf`, `/predict/gru`, `/predict/lstm`, `/predict/setfit`)

-  **Prometheus metrics** at `/metrics`

-  **Gradio UI** mounted at `/label` (inside the FastAPI app)

  ---

title: Naplace Bug Component Classifier API

sdk: docker

app_port: 7860

---

<a  target="_blank"  href="https://cookiecutter-data-science.drivendata.org/">

<img  src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter"  />

</a>

  

## What this repository contains
This repository is structured as a SE4AI MLOps project (Milestones M1 → M6):

-  **M1**: project inception + organized repository structure

-  **M2**: reproducibility (pipelines, tracking)

-  **M3**: QA (lint, tests, data validation, behavioral tests)

-  **M4**: API serving (FastAPI + schema + tests)

-  **M5**: deployment (Docker + Compose + CI)

-  **M6**: monitoring & load testing  (Prometheus + Grafana + Locust + drift utilities)
 
The grading requirement is satisfied by listing, for each milestone:

1) what was implemented

2) where the reviewer can find evidence in the repository.

---
## Key files/folders used
-  `naplace/` → package Sorgente

-  `api/` → FastAPI app + inference routing

-  `cli/` → dataset pipeline (prepare/convert/split/check)

-  `modeling/` → training & evaluation logic

-  `observability/` → Prometheus metrics primitives

-  `ui/` → Gradio UI mounted in the API

-  `scripts/` → train / eval / data validation utilities

-  `tests/` → unit + integration + smoke + behavioral tests

-  `reports/` → exported evidence (Deepchecks HTML, metrics JSON, etc.)

-  `docker-compose.yml` → API deployment (host `8000` → container `7860`)

-  `docker-compose.monitoring.yml` + `monitoring/` → monitoring stack (Prometheus/Grafana/Locust)
-  `models/` → trained models/artifacts
-  `.github/workflows/ci.yml` → CI pipeline (lint/tests)

Note on monitoring folders:
- It is normal to have `monitoring/grafana/` empty if Grafana uses a Docker volume (`grafana_data`) for persistence.
- `monitoring/prometheus/prometheus.yml` is expected to exist because Prometheus loads config from file.


## Project Organization

```
├── .github/workflows/ci.yml          <- GitHub Actions CI (lint/test/build)
├── .dvc/                             <- DVC metadata (remote, cache config)
├── data/                             <- Data folder (see data/README.md for details)
├── docs/                             <- MkDocs project (documentation site sources)
│   ├── mkdocs.yml                    <- MkDocs configuration
│   └── docs/                         <- Documentation pages (index, getting-started, etc.)
├── loadtest/                         <- Load testing with Locust
│   └── locustfile.py                 <- Locust сценарии per stress/load test
├── models/                           <- Trained models and artifacts tracked via DVC
│   └── best.h5.dvc                   <- DVC pointer to best serialized model
├── monitoring/                       <- Observability stack configuration
│   └── prometheus/                   <- Prometheus configuration
│       └── prometheus.yml            <- Prometheus scrape config
├── naplace/                          <- Python package (project source code)
│   ├── __init__.py                   <- Makes naplace a Python module
│   ├── config.py                     <- Centralized config (paths, constants, settings)
│   ├── dataset.py                    <- Dataset utilities (loading/handling)
│   ├── labeling.py                   <- Labeling utilities (for UI / workflows)
│   ├── api/                          <- FastAPI service
│   │   ├── main.py                   <- API entrypoint (routers, app wiring)
│   │   ├── inference.py              <- Inference logic used by the API
│   │   └── models.py                 <- Pydantic/ML model schemas
│   ├── cli/                          <- CLI utilities (dataset prep, checks, splits)
│   │   ├── convert_bugbug.py         <- Convert BugBug format into project format
│   │   ├── prepare.py                <- Prepare datasets for training/eval
│   │   ├── split.py                  <- Train/val/test splitting utilities
│   │   └── check_dataset.py          <- Dataset sanity checks
│   ├── modeling/                     <- Modeling code (train/eval/predict)
│   │   ├── baseline_tfidf.py         <- TF-IDF baseline implementation
│   │   ├── train_lstm.py             <- LSTM training pipeline
│   │   ├── train_gru.py              <- GRU training pipeline
│   │   ├── setfit_model.py           <- SetFit model definition/training helpers
│   │   ├── eval_setfit.py            <- SetFit evaluation
│   │   ├── eval_seq.py               <- Sequence models evaluation (LSTM/GRU)
│   │   └── predict.py                <- Inference/prediction utilities
│   ├── observability/                <- Metrics & monitoring helpers
│   │   └── metrics.py                <- Prometheus metrics integration
│   └── ui/                           <- UI layer
│       └── gradio.py                 <- Gradio UI mounted under /label
├── reports/                           <- Generated reports/artifacts (metrics, validations)
│   ├── metrics_*.json                <- Metrics snapshots for models/experiments
│   ├── deepchecks_*.html             <- Data validation reports (Deepchecks)
│   └── figures/                      <- Figures used in reports
├── scripts/                           <- Project scripts (training, evaluation, validation)
│   ├── train_*.py                    <- Training entrypoints (baseline/setfit/tfidf_sgd)
│   ├── eval_*.py                     <- Evaluation scripts (API, SetFit, etc.)
│   ├── validate_data_*.py            <- Data validation (GE/Deepchecks)
│   ├── select_best.py                <- Select best model/artifact
│   └── alibi_detect_drift.py         <- Drift detection utilities
├── tests/                             <- Pytest test suite (API, CLI, smoke, imports)
├── docker-compose.yml                 <- Local stack (API + UI, etc.)
├── docker-compose.monitoring.yml      <- Monitoring stack (Prometheus, etc.)
├── Dockerfile                         <- Container build for the service
├── dvc.yaml                           <- DVC pipeline definition
├── dvc.lock                           <- Locked versions of DVC stages/artifacts
├── Makefile                           <- Convenience commands (dev, data, train, etc.)
├── pyproject.toml                     <- Project metadata + tool configuration (ruff, etc.)
├── requirements.txt                   <- Python dependencies (local/dev)
├── requirements.docker.txt            <- Python dependencies for Docker image
└── README.md                          <- This file


```

  

--------

# How to run

  

## Option A — Local (no Docker)

  

### Install
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    uvicorn naplace.api.main:app --host 0.0.0.0 --port 7860

## Option B — Docker (API Only)
### Install

    python -m venv .venv
    
    .venv\Scripts\Activate.ps1
    
    pip install -r requirements.txt
    
    docker compose up --build

Then open:

    http://localhost:8000/docs
    http://localhost:8000/health
    http://localhost:8000/metrics
    http://localhost:8000/label

  

## Option C — Monitoring stack 

  

### Install

    python -m venv .venv
    
    .venv\Scripts\Activate.ps1
    
    docker compose -f docker-compose.monitoring.yml up --build


This uses docker-compose.monitoring.yml and starts:

    api (host 8000)
    prometheus (host 9090)
    grafana (host 3000)
    locust (host 8089)

Open:

  
    api (host 8000)
    Prometheus: http://localhost:9090
    Grafana: http://localhost:3000
    Locust: http://localhost:8089
    
Drift detection (Alibi Detect): 

    docker compose -f docker-compose.monitoring.yml exec -T api python scripts/alibi_detect_drift.py

Prometheus scrapes the API metrics endpoint /metrics.

Locust targets the API internally at http://api:7860

--------

  

# Useful endpoints

    GET /docs → Swagger UI
    
    GET /health → service healthcheck
    
    GET /metrics → Prometheus metrics
    
    GET /label → Gradio UI
    
    POST /predict/{tfidf|gru|lstm|setfit} → predictions

  

# Data pipeline

The dataset workflow is implemented as CLI modules in naplace/cli/.
Typical sequence:

    dvc pull // to obtain dataset stored remotely
    python -m naplace.cli.prepare // for preprocessing
    python -m naplace.cli.convert_bugbug // to convert it into a usable formato
    python -m naplace.cli.split // to split in train / test
    python -m naplace.cli.check_dataset

Outputs are written under data/ (raw/interim), and train/test JSONL are generated by the split step.

  

-----------

# Training & evaluation
Some tests and inference endpoints expect model artifacts under models/.

If you are running on a fresh clone, train at least one model first (or ensure models/ is populated).

## Baselines
### TF-IDF baseline:

    python scripts/train_baseline_tfidf.py

### TF-IDF + SGD:

    python scripts/train_tfidf_sgd.py --train-path data/interim/train.jsonl --test-path data/interim/test.jsonl --model-out models/tfidf_sgd.joblib --metrics-out reports/metrics_tfidf_sgd.json

### SetFit
Training:

    python scripts/train_setfit.py --train-path data/interim/train.jsonl --val-path data/interim/test.jsonl --output-dir models/setfit_component --metrics-path reports/metrics_setfit_component.json

Evaluation:

    python scripts/eval_setfit.py

  

### Sequence models (GRU / LSTM)

Train:

    python -m naplace.modeling.train_gru
    python -m naplace.modeling.train_lstm

Evaluate:

    python -m naplace.modeling.eval_seq --model models/gru.h5 --tok models/gru_tokenizer.pkl --classes models/gru_label_classes.npy --data data/interim/test.jsonl --out reports/metrics_gru.json --run_name eval_gru

  

-----------

  

# QA (lint, tests, data validation)

 ## Lint (Ruff)

    ruff check .

  

## Tests (Pytest)

    pytest -q

  

The test suite includes:

- API tests for /health, /metrics, and prediction endpoints (tests/test_api.py, tests/test_smoke_api.py)

- CLI unit tests (tests/test_cli_*)

- Behavioral model tests (tests/test_behavioral_model.py) with invariance/minimum functionality checks (and some xfail “desired” tests)

### Tests Results
- 33 tests collected: 28 passed, 5 xfailed.
- All xfails are in tests/test_behavioral_model.py and are intentional “directional” checks that encode desired behavior the current LSTM model doesn’t reliably meet yet.

Why these are xfailed (expected failures)
- test_directional_ui_vs_network: expects different labels for UI vs. network bugs, but the model often collapses both to “General”.
- test_directional_crash_vs_layout: expects different labels for crash vs. layout issues; current model tends to map both to “General”.
- test_directional_stacktrace_detection: expects stacktrace-heavy crash reports to differ from layout bugs; current model collapses them.
- test_directional_product_component_consistency: expects Thunderbird/Mail issues to differ from Firefox UI; model often returns “General” for both.
- test_directional_history_length: expects complex/long-history bugs to differ from trivial UI glitches; model tends to give the same label.

Why the other tests pass
- Invariance tests (whitespace/case, small rewording, spam/noise, multilingual) pass because the model is stable under superficial or equivalent changes.
- Minimum functionality tests pass because the model always returns a non-empty label for basic inputs.
- API tests pass because /health, /metrics, and /predict endpoints return the expected status codes and response shapes.
- CLI/config/import tests pass because the utilities behave as expected and modules import cleanly.

---  

# Data validation

  

## Great Expectations:

  

    python scripts/validate_data_gx.py

  
  

Summarize GX results:

  

    python scripts/summarize_gx_results.py

  
  

Deepchecks (exports HTML under reports/):

  

    python scripts/validate_data_deepchecks.py

  

---------

# Milestones Recap :

This section is explicitly written for reviewers.

  

## M1 — Inception

  

Evidence:

  

- repository structure (naplace/, scripts/, tests/, reports/)

  

- Cookiecutter Data Science reference (badge)

  

## M2 — Reproducibility

  

Evidence:

  

- pipeline scripts in naplace/cli/ (prepare/convert/split)

  

- experiment tracking in scripts/train_setfit.py

  

- exported metrics in reports/metrics_*.json

  

- dvc.yaml / dvc.lock for pipeline tracking

  

## M3 — QA

  

Evidence:

  

- pyproject.toml and ruff usage

  

- pytest suite under tests/

  
- data validation scripts in scripts/

  

- Deepchecks HTML reports under reports/

  

## M4 — API

  

Evidence:

  

- FastAPI app in naplace/api/main.py

  

- request/response models in naplace/api/models.py

  

- inference integration in naplace/api/inference.py

  

- API tests in tests/test_api.py and tests/test_smoke_api.py

  

## M5 — Deployment

  

Evidence:

  
- Dockerfile

  

- docker-compose.yml

  

- .github/workflows/ci.yml

  

## M6 — Monitoring

  

Evidence:

- naplace/observability/metrics.py + /metrics endpoint- docker-compose.monitoring.yml (Prometheus + Grafana + Locust)
- monitoring/prometheus/prometheus.yml
- loadtest/locustfile.py
- scripts/alibi_detect_drift.py
- evidence in reports/

---








