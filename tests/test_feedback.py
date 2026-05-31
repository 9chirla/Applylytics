"""Tests for export feedback JSONL logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from applylytics.ui.logic import feedback_file_path, record_export_feedback


@pytest.fixture
def feedback_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "data" / "feedback.jsonl"
    monkeypatch.setattr("applylytics.ui.logic.FEEDBACK_JSONL", path)
    return path


def test_feedback_file_created_if_absent(feedback_path: Path):
    assert not feedback_path.exists()
    record_export_feedback("optimised_export", 55, helpful=True)
    assert feedback_path.is_file()


def test_feedback_appends_valid_json_line(feedback_path: Path):
    record_export_feedback("optimised_export", 72, helpful=True)
    lines = feedback_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["phase"] == "optimised_export"
    assert row["ats_score"] == 72
    assert row["helpful"] is True
    assert "ts" in row


def test_feedback_multiple_lines_not_overwrite(feedback_path: Path):
    record_export_feedback("optimised_export", 40, helpful=False)
    record_export_feedback("optimised_export", 80, helpful=True)
    lines = feedback_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["helpful"] is False
    assert json.loads(lines[1])["helpful"] is True


def test_feedback_lines_parseable(feedback_path: Path):
    for score in (10, 50, 90):
        record_export_feedback("test_phase", score, helpful=score >= 50)
    for line in feedback_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)


def test_feedback_file_path_under_data_dir(feedback_path: Path):
    record_export_feedback("x", 0, helpful=True)
    assert feedback_file_path().parent.name == "data"
