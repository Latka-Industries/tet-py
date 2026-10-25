"""`tet.run_job` — thin wrapper over Rust `tetration::job`."""

from __future__ import annotations

from pathlib import Path

import pytest

import tet
from tet import TetError


def test_run_job_dry_run_plans_steps(tmp_path: Path) -> None:
    job = tmp_path / "job.toml"
    job.write_text(
        'schema_version = 1\n[[step]]\nop = "convert"\nfrom = "t.csv"\nto = "t.tet"\n',
        encoding="utf-8",
    )
    (tmp_path / "t.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    report = tet.run_job(job, dry_run=True)
    assert report["dry_run"] is True
    assert len(report["steps"]) == 1
    assert report["steps"][0]["op"] == "convert"
    assert report["steps"][0]["status"] == "planned"
    assert report["steps"][0]["data"] is None
    assert not (tmp_path / "t.tet").exists()


def test_run_job_executes_convert_and_verify(tmp_path: Path) -> None:
    job = tmp_path / "job.toml"
    job.write_text(
        "schema_version = 1\n"
        '[[step]]\nop = "convert"\nfrom = "t.csv"\nto = "t.tet"\n'
        '[[step]]\nop = "verify"\npath = "t.tet"\n',
        encoding="utf-8",
    )
    (tmp_path / "t.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    report = tet.run_job(job)
    assert report["dry_run"] is False
    assert len(report["steps"]) == 2
    assert report["steps"][0]["data"]["dataset_count"] == 2
    assert report["steps"][1]["data"]["ok"] is True
    assert (tmp_path / "t.tet").is_file()


def test_run_job_step_failure_raises(tmp_path: Path) -> None:
    job = tmp_path / "job.toml"
    job.write_text(
        'schema_version = 1\n[[step]]\nop = "verify"\npath = "missing.tet"\n',
        encoding="utf-8",
    )
    with pytest.raises(TetError, match=r"step 1 \(verify\) failed"):
        tet.run_job(job)


def test_versions_pin_082() -> None:
    assert tet.__version__ == "0.8.2"
    assert tet.core_version() == "0.8.2"
