import json

import pytest

from naplace.cli.split import (
    read_jsonl_lines,
    parse_record,
    extract_label,
    build_text,
    stable_bucket,
    HASH_SALT,
)


def test_parse_record_valid_and_invalid():
    valid = '{"id": 1, "summary": "Test"}'
    invalid = '{"id": 1, "summary": "Test"'  # manca la graffa finale

    assert parse_record(valid) == {"id": 1, "summary": "Test"}
    assert parse_record(invalid) is None


def test_extract_label_default_unknown():
    rec_no_component = {"id": 1, "summary": "Test"}
    rec_with_component = {"id": 2, "component": "Core"}

    assert extract_label(rec_no_component) == "Unknown"
    assert extract_label(rec_with_component) == "Core"


@pytest.mark.parametrize(
    "rec, expected_substring",
    [
        ({"summary": "Summary only"}, "Summary only"),
        (
            {
                "summary": "Summary",
                "comments": [{"text": "First comment"}, {"text": "Second"}],
            },
            "Summary First comment",
        ),
        (
            {
                "summary": "Summary",
                "description": "Fallback description",
            },
            "Summary Fallback description",
        ),
    ],
)
def test_build_text_behavior(rec, expected_substring):
    text = build_text(rec, max_desc_len=400)
    assert expected_substring in text
    assert len(text) > 0


def test_build_text_truncates_description():
    long_desc = "A" * 1000
    rec = {"summary": "S", "description": long_desc}

    text = build_text(rec, max_desc_len=100)
    # summary + spazio + desc tagliata
    # quindi lunghezza deve essere <= 1 (summary) + 1 + 100
    assert len(text) <= 102


def test_stable_bucket_is_deterministic():
    key = "12345"
    bucket1 = stable_bucket(key, ratio=0.2)
    bucket2 = stable_bucket(key, ratio=0.2)

    # con la stessa chiave e lo stesso sale deve sempre dare lo stesso bucket
    assert bucket1 == bucket2

    # sanity check: con ratio molto alto, è molto probabile che esca "test"
    bucket_high_ratio = stable_bucket(key, ratio=0.99)
    assert bucket_high_ratio in ("train", "test")  # almeno il tipo è corretto

    # giusto per sicurezza, controlliamo che HASH_SALT sia usato come stringa
    assert isinstance(HASH_SALT, str)
