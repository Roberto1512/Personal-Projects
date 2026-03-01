def test_import_naplace_package():
    import naplace  # noqa: F401


def test_import_cli_prepare():
    from naplace.cli import prepare  # noqa: F401


def test_import_cli_split():
    from naplace.cli import split  # noqa: F401


def test_import_modeling_modules():
    from naplace.modeling import train_gru, train_lstm, eval_seq  # noqa: F401
