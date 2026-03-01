# scripts/validate_data_deepchecks.py

from io import StringIO
import itertools
from pathlib import Path

import pandas as pd
from sklearn.metrics import get_scorer_names
from sklearn.metrics._scorer import _SCORERS

# Compatibility patch: Deepchecks expects a "max_error" scorer, removed in sklearn>=1.8.
if "max_error" not in get_scorer_names() and "neg_max_error" in get_scorer_names():
    _SCORERS["max_error"] = _SCORERS["neg_max_error"]

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation

from naplace.config import INTERIM, REPORTS

TRAIN_PATH = INTERIM / "train.jsonl"
TEST_PATH = INTERIM / "test.jsonl"


def load_jsonl_sample(path: Path, max_lines: int = 10000) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"❌ File not found: {path}. Run naplace/cli/split.py first.")

    with path.open("r", encoding="utf-8") as f:
        lines = list(itertools.islice(f, max_lines))

    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        raise SystemExit(f"❌ No valid lines found in {path}")

    buffer = StringIO("".join(lines))
    df = pd.read_json(buffer, lines=True)
    return df


def make_dc_dataset(df: pd.DataFrame) -> Dataset:
    # Usiamo solo le colonne pulite e rilevanti
    candidate_cols = ["id", "text", "component"]
    cols = [c for c in candidate_cols if c in df.columns]

    if "component" not in cols:
        raise SystemExit("❌ Column 'component' not found in dataframe for Deepchecks Dataset.")

    df_small = df[cols].copy()

    # Creiamo il Dataset:
    # - label = "component"
    # - le feature saranno tutte le altre colonne (id, text)
    # - niente cat_features esplicite (Deepchecks se la cava da solo o non è critico qui)
    return Dataset(
        df_small,
        label="component",
    )


def run_data_integrity(train_ds: Dataset):
    suite = data_integrity()
    result = suite.run(train_ds)

    REPORTS.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS / "deepchecks_data_integrity_train.html"
    result.save_as_html(str(html_path))

    print(f"✅ Deepchecks Data Integrity completata. Report: {html_path}")


def run_train_test_validation(train_ds: Dataset, test_ds: Dataset):
    suite = train_test_validation()
    result = suite.run(train_ds, test_ds)

    REPORTS.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS / "deepchecks_train_test_validation.html"
    result.save_as_html(str(html_path))

    print(f"✅ Deepchecks Train-Test Validation completata. Report: {html_path}")


def main():
    print("[Naplace] Carico sample di train/test per Deepchecks...")
    train_df = load_jsonl_sample(TRAIN_PATH, max_lines=10000)
    test_df = load_jsonl_sample(TEST_PATH, max_lines=10000)

    train_ds = make_dc_dataset(train_df)
    test_ds = make_dc_dataset(test_df)

    print("[Naplace] Eseguo Data Integrity Suite su train...")
    run_data_integrity(train_ds)

    print("[Naplace] Eseguo Train-Test Validation Suite su train vs test...")
    run_train_test_validation(train_ds, test_ds)


if __name__ == "__main__":
    main()
