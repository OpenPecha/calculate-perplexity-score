"""
Tests for calculate_perplexity.py

All test data is embedded directly in this file as string constants.
No external data directory is required.
"""

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import calculate_perplexity as cp

# ── Sample data taken from data/inference/20260317/2026-03-17T08-21-29.csv ──

SAMPLE_TRANSCRIPTION = (
    "ནོར་གྱིས་འབྱོར་ན་སྟོབས་ཀྱང་འཕེལ་། "
    "ནོར་མེད་པ་ལ་སྟོབས་ཀྱང་འབྲི་། "
    "བྱི་བས་དབྱིག་ལྡན་ནོར་ཕྲོགས་པས་། "
    "བརྐུ་བའི་ནུས་པ་ཉམས་ཞེས་ཐོས་།"
)

SAMPLE_TRANSCRIPTION_WITH_NEWLINES = (
    "ནོར་གྱིས་འབྱོར་ན་སྟོབས་ཀྱང་འཕེལ་།\n"
    "ནོར་མེད་པ་ལ་སྟོབས་ཀྱང་འབྲི་།\n"
    "\n- 69 -\n"
)

SAMPLE_FILE_NAME = "0001.png"


# ── prepare_raw ────────────────────────────────────────────────────────────────


def test_prepare_raw_strips_newlines():
    text = "\nནོར་གྱིས་འབྱོར་ན།\n"
    assert cp.prepare_raw(text) == "ནོར་གྱིས་འབྱོར་ན།"


def test_prepare_raw_strips_leading_trailing_whitespace():
    text = "  ནོར་གྱིས་འབྱོར་ན།  "
    assert cp.prepare_raw(text) == "ནོར་གྱིས་འབྱོར་ན།"


def test_prepare_raw_preserves_internal_content():
    result = cp.prepare_raw(SAMPLE_TRANSCRIPTION_WITH_NEWLINES)
    # Internal content (including embedded newlines) is preserved; only
    # leading/trailing newlines and whitespace are removed.
    assert "ནོར་གྱིས་འབྱོར་ན་སྟོབས་ཀྱང་འཕེལ་།" in result


def test_prepare_raw_empty_string():
    assert cp.prepare_raw("") == ""


def test_prepare_raw_only_whitespace():
    assert cp.prepare_raw("   \n\r  ") == ""


# ── compute_perplexity ─────────────────────────────────────────────────────────


def _make_mock_model(perplexity_value: float = 123.45) -> MagicMock:
    model = MagicMock()
    model.perplexity.return_value = perplexity_value
    return model


def test_compute_perplexity_returns_float():
    model = _make_mock_model(250.0)
    result = cp.compute_perplexity("ནོར་ གྱིས་", model)
    assert isinstance(result, float)
    assert result == 250.0


def test_compute_perplexity_rounds_to_4_decimals():
    model = _make_mock_model(123.456789)
    result = cp.compute_perplexity("ནོར་", model)
    assert result == round(123.456789, 4)


def test_compute_perplexity_empty_string_returns_none():
    model = _make_mock_model()
    assert cp.compute_perplexity("", model) is None


def test_compute_perplexity_whitespace_only_returns_none():
    model = _make_mock_model()
    assert cp.compute_perplexity("   ", model) is None


def test_compute_perplexity_model_exception_returns_none():
    model = MagicMock()
    model.perplexity.side_effect = RuntimeError("kenlm error")
    assert cp.compute_perplexity("ནོར་", model) is None


def test_compute_perplexity_model_returns_none_propagates():
    model = MagicMock()
    model.perplexity.return_value = None
    assert cp.compute_perplexity("ནོར་", model) is None


# ── tokenize_syllables ────────────────────────────────────────────────────────


def test_tokenize_syllables_empty_returns_empty():
    tokenizer = MagicMock()
    assert cp.tokenize_syllables("", tokenizer) == ""


def test_tokenize_syllables_whitespace_only_returns_empty():
    tokenizer = MagicMock()
    assert cp.tokenize_syllables("   ", tokenizer) == ""


def test_tokenize_syllables_joins_tokens_with_spaces():
    tokenizer = MagicMock()
    tokenizer.tokenize.return_value = ["ནོར་", "གྱིས་", "འབྱོར་"]
    result = cp.tokenize_syllables("ནོར་གྱིས་འབྱོར་", tokenizer)
    assert result == "ནོར་ གྱིས་ འབྱོར་"


def test_tokenize_syllables_calls_tokenizer_with_input():
    tokenizer = MagicMock()
    tokenizer.tokenize.return_value = []
    cp.tokenize_syllables(SAMPLE_TRANSCRIPTION, tokenizer)
    tokenizer.tokenize.assert_called_once_with(SAMPLE_TRANSCRIPTION)


# ── compute_run ───────────────────────────────────────────────────────────────


def _write_inference_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file_name", "transcription", "confidence_index"])
        writer.writeheader()
        writer.writerows(rows)


def _make_run_dirs(tmp_path: Path, benchmark_id: str, run_id: str, rows: list[dict]) -> tuple[Path, Path]:
    inference_dir = tmp_path / "inference" / benchmark_id
    inference_dir.mkdir(parents=True)
    analysis_dir = tmp_path / "analysis" / "perplexity"
    analysis_dir.mkdir(parents=True)
    _write_inference_csv(inference_dir / f"{run_id}.csv", rows)
    return inference_dir, analysis_dir


SAMPLE_ROWS = [
    {
        "file_name": "0001.png",
        "transcription": SAMPLE_TRANSCRIPTION,
        "confidence_index": "0.95",
    },
    {
        "file_name": "0002.jpg",
        "transcription": (
            "ལྗོངས་འདིར་སྤྱི་ཤིང་སྣ་ཚོགས་ཡར་སྐྱེད་དུ་འཛོམས་ཤིང་། "
            "བཀྲ་ཤིས་གཡང་དུ་ཆགས་པའི་རྩ་བོད་ཚུགས་བྱུང་།"
        ),
        "confidence_index": "0.88",
    },
    {
        "file_name": "0003.jpg",
        "transcription": "",
        "confidence_index": "0.0",
    },
]


def _make_mocks() -> tuple[MagicMock, MagicMock]:
    model = _make_mock_model(300.0)
    tokenizer = MagicMock()
    tokenizer.tokenize.side_effect = lambda text: text.split()
    return model, tokenizer


def test_compute_run_creates_output_csv(tmp_path):
    benchmark_id = "20260317"
    run_id = "2026-03-17T08-21-29"
    _, analysis_dir = _make_run_dirs(tmp_path, benchmark_id, run_id, SAMPLE_ROWS)
    model, tokenizer = _make_mocks()

    with patch.object(cp, "INFERENCE", tmp_path / "inference"), \
         patch.object(cp, "ANALYSIS", tmp_path / "analysis" / "perplexity"):
        out_path = cp.compute_run(benchmark_id, run_id, model, tokenizer)

    assert out_path.exists()
    with out_path.open(encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))

    # Row with empty transcription still produces an entry (with empty ppl fields)
    file_names = {r["file_name"] for r in records}
    assert "0001.png" in file_names
    assert "0002.jpg" in file_names
    assert "0003.jpg" in file_names


def test_compute_run_output_columns(tmp_path):
    benchmark_id = "20260317"
    run_id = "2026-03-17T08-21-29"
    _make_run_dirs(tmp_path, benchmark_id, run_id, SAMPLE_ROWS)
    model, tokenizer = _make_mocks()

    with patch.object(cp, "INFERENCE", tmp_path / "inference"), \
         patch.object(cp, "ANALYSIS", tmp_path / "analysis" / "perplexity"):
        out_path = cp.compute_run(benchmark_id, run_id, model, tokenizer)

    with out_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert set(reader.fieldnames) == {"file_name", "perplexity_raw", "perplexity_normalized"}


def test_compute_run_rows_sorted_by_file_name(tmp_path):
    benchmark_id = "20260317"
    run_id = "2026-03-17T08-21-29"
    # Intentionally provide rows out of order
    rows = list(reversed(SAMPLE_ROWS))
    _make_run_dirs(tmp_path, benchmark_id, run_id, rows)
    model, tokenizer = _make_mocks()

    with patch.object(cp, "INFERENCE", tmp_path / "inference"), \
         patch.object(cp, "ANALYSIS", tmp_path / "analysis" / "perplexity"):
        out_path = cp.compute_run(benchmark_id, run_id, model, tokenizer)

    with out_path.open(encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))

    file_names = [r["file_name"] for r in records]
    assert file_names == sorted(file_names)


def test_compute_run_empty_transcription_produces_empty_perplexity(tmp_path):
    benchmark_id = "20260317"
    run_id = "2026-03-17T08-21-29"
    rows = [{"file_name": "0003.jpg", "transcription": "", "confidence_index": "0.0"}]
    _make_run_dirs(tmp_path, benchmark_id, run_id, rows)
    model, tokenizer = _make_mocks()

    with patch.object(cp, "INFERENCE", tmp_path / "inference"), \
         patch.object(cp, "ANALYSIS", tmp_path / "analysis" / "perplexity"):
        out_path = cp.compute_run(benchmark_id, run_id, model, tokenizer)

    with out_path.open(encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))

    assert records[0]["perplexity_raw"] == ""
    assert records[0]["perplexity_normalized"] == ""


def test_compute_run_missing_file_raises(tmp_path):
    benchmark_id = "20260317"
    run_id = "nonexistent-run"
    (tmp_path / "inference" / benchmark_id).mkdir(parents=True)
    (tmp_path / "analysis" / "perplexity").mkdir(parents=True)
    model, tokenizer = _make_mocks()

    with patch.object(cp, "INFERENCE", tmp_path / "inference"), \
         patch.object(cp, "ANALYSIS", tmp_path / "analysis" / "perplexity"):
        with pytest.raises(FileNotFoundError):
            cp.compute_run(benchmark_id, run_id, model, tokenizer)


def test_compute_run_skips_rows_with_no_file_name(tmp_path):
    benchmark_id = "20260317"
    run_id = "2026-03-17T08-21-29"
    rows = [
        {"file_name": "", "transcription": "some text", "confidence_index": "0.9"},
        {"file_name": "0001.png", "transcription": SAMPLE_TRANSCRIPTION, "confidence_index": "0.95"},
    ]
    _make_run_dirs(tmp_path, benchmark_id, run_id, rows)
    model, tokenizer = _make_mocks()

    with patch.object(cp, "INFERENCE", tmp_path / "inference"), \
         patch.object(cp, "ANALYSIS", tmp_path / "analysis" / "perplexity"):
        out_path = cp.compute_run(benchmark_id, run_id, model, tokenizer)

    with out_path.open(encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))

    assert len(records) == 1
    assert records[0]["file_name"] == "0001.png"
