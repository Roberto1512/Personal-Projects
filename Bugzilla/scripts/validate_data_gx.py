

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
        raise SystemExit(f"ERROR Train file not found: {path}. Run naplace/cli/split.py first.")


    with path.open("r", encoding="utf-8") as f:
        lines = list(itertools.islice(f, max_lines))

    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        raise SystemExit(f"ERROR No valid lines found in {path}")

    buffer = StringIO("".join(lines))
    df = pd.read_json(buffer, lines=True)
    return df


def build_context_and_batch(df: pd.DataFrame):

    context = gx.get_context()


    data_source = context.data_sources.add_pandas(
        name="naplace_train_datasource",
    )


    data_asset = data_source.add_dataframe_asset(
        name="naplace_train_asset",
    )


    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "naplace_train_batch_definition"
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    return context, batch_definition, batch


def build_expectation_suite(context, batch_definition):

    suite = context.suites.add(
        gx.core.expectation_suite.ExpectationSuite(name="naplace_train_expectations")
    )


    for col in ["id", "summary", "text", "component", "product", "macro_component"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=col,
            )
        )


    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(
            column="id",
        )
    )


    suite.add_expectation(
        gx.expectations.ExpectColumnValueLengthsToBeBetween(
            column="summary",
            min_value=4,
            max_value=300,
        )
    )


    suite.add_expectation(
        gx.expectations.ExpectColumnValueLengthsToBeBetween(
            column="text",
            min_value=10,
            max_value=20000,
            mostly=0.99,
        )
    )


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


    suite.add_expectation(
        gx.expectations.ExpectColumnUniqueValueCountToBeBetween(
            column="component",
            min_value=5,
        )
    )


    validation_definition = context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name="naplace_train_validation_definition",
            data=batch_definition,
            suite=suite,
        )
    )

    return suite, validation_definition


def run_checkpoint(context, validation_definition, df: pd.DataFrame):

    checkpoint = context.checkpoints.add(
        gx.checkpoint.checkpoint.Checkpoint(
            name="naplace_train_checkpoint",
            validation_definitions=[validation_definition],
        )
    )


    checkpoint_result = checkpoint.run(batch_parameters={"dataframe": df})


    print(checkpoint_result.describe())


def main():
    df = load_train_df()
    context, batch_definition, batch = build_context_and_batch(df)
    _, validation_definition = build_expectation_suite(context, batch_definition)
    run_checkpoint(context, validation_definition, df)


if __name__ == "__main__":
    main()
