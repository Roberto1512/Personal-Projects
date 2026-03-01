# scripts/validate_data_gx.py

from io import StringIO
import itertools
from pathlib import Path

import great_expectations as gx
import pandas as pd

from naplace.config import INTERIM
from naplace.labeling import BUGBUG_PRODUCTS

TRAIN_PATH = INTERIM / "train.jsonl"


def load_train_df(path: Path = TRAIN_PATH, max_lines: int = 10000) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"❌ Train file not found: {path}. Run naplace/cli/split.py first.")

    # Leggiamo solo le prime `max_lines` righe non vuote
    with path.open("r", encoding="utf-8") as f:
        lines = list(itertools.islice(f, max_lines))

    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        raise SystemExit(f"❌ No valid lines found in {path}")

    buffer = StringIO("".join(lines))
    df = pd.read_json(buffer, lines=True)
    return df


def build_context_and_batch(df: pd.DataFrame):
    # Step 1: Data Context
    context = gx.get_context()

    # Step 2: Datasource (Pandas)
    data_source = context.data_sources.add_pandas(
        name="naplace_train_datasource",
    )

    # Step 3: Data Asset
    data_asset = data_source.add_dataframe_asset(
        name="naplace_train_asset",
    )

    # Step 4: Batch Definition + Batch
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "naplace_train_batch_definition"
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    return context, batch_definition, batch


def build_expectation_suite(context, batch_definition):
    # Creiamo una Expectation Suite
    suite = context.suites.add(
        gx.core.expectation_suite.ExpectationSuite(name="naplace_train_expectations")
    )

    # === 1. Colonne "core" non nulle ===
    for col in ["id", "summary", "text", "component", "product", "macro_component"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=col,
            )
        )

    # id deve essere unico
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(
            column="id",
        )
    )

    # === 2. Qualità testi ===
    # summary: accettiamo anche summary corti tipo "test", ma evitiamo quelli proprio ridicoli
    suite.add_expectation(
        gx.expectations.ExpectColumnValueLengthsToBeBetween(
            column="summary",
            min_value=4,  # prima 10
            max_value=300,
        )
    )

    # text: facciamo una soglia minima più bassa, perché nel dump esistono bug molto brevi
    suite.add_expectation(
        gx.expectations.ExpectColumnValueLengthsToBeBetween(
            column="text",
            min_value=10,  # prima 50
            max_value=20000,
            mostly=0.99,
        )
    )

    # === 3. Domini categoriali (product / macro_component) ===

    product_values = sorted(BUGBUG_PRODUCTS)
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="product",
            value_set=product_values,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="macro_component",
            value_set=product_values,
        )
    )

    # === 4. Varieta di component (soft check) ===

    suite.add_expectation(
        gx.expectations.ExpectColumnUniqueValueCountToBeBetween(
            column="component",
            min_value=5,
        )
    )

    # Validation Definition
    validation_definition = context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name="naplace_train_validation_definition",
            data=batch_definition,
            suite=suite,
        )
    )

    return suite, validation_definition


def run_checkpoint(context, validation_definition, df: pd.DataFrame):
    # Step 4: Checkpoint
    checkpoint = context.checkpoints.add(
        gx.checkpoint.checkpoint.Checkpoint(
            name="naplace_train_checkpoint",
            validation_definitions=[validation_definition],
        )
    )

    # Eseguiamo il checkpoint passando il batch
    checkpoint_result = checkpoint.run(batch_parameters={"dataframe": df})

    # Stampa un riassunto in console
    print(checkpoint_result.describe())


def main():
    df = load_train_df()
    context, batch_definition, batch = build_context_and_batch(df)
    _, validation_definition = build_expectation_suite(context, batch_definition)
    run_checkpoint(context, validation_definition, df)


if __name__ == "__main__":
    main()
