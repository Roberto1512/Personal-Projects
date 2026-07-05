from naplace.config import ROOT, DATA, RAW, INTERIM, PROCESSED, MODELS, REPORTS


def test_config_paths_structure():


    assert DATA == ROOT / "data"
    assert RAW == DATA / "raw"
    assert INTERIM == DATA / "interim"
    assert PROCESSED == DATA / "processed"
    assert MODELS == ROOT / "models"
    assert REPORTS == ROOT / "reports"
