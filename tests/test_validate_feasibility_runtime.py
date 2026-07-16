"""Fast CPU tests for feasibility-validator runtime failure semantics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from g1_rickshaw_lab.validation import (
    synchronize_runtime_randomization_events,
    write_json_atomic,
)

from scripts import validate_feasibility


def _pending_report() -> dict:
    return {
        "schema_version": 2,
        "tool": "validate_feasibility",
        "status": "failed",
        "task": "task-v0",
        "created_utc": "2026-07-13T00:00:00Z",
        "inputs": {},
        "metrics": {},
        "failures": ["feasibility scan has not completed"],
        "metadata": {"failure_phase": "pending"},
    }


def test_runtime_randomization_rebinds_startup_and_reset_events_atomically() -> None:
    stale = object()
    runtime = object()
    initialize = SimpleNamespace(params={"cfg": stale, "sentinel": "startup"})
    sample = SimpleNamespace(params={"cfg": stale, "sentinel": "reset"})
    cfg = SimpleNamespace(
        runtime_randomization=stale,
        events=SimpleNamespace(
            initialize_curriculum=initialize,
            sample_physics=sample,
        ),
    )

    synchronize_runtime_randomization_events(cfg, runtime)

    assert cfg.runtime_randomization is runtime
    assert initialize.params == {"cfg": runtime, "sentinel": "startup"}
    assert sample.params == {"cfg": runtime, "sentinel": "reset"}
    assert initialize.params["cfg"] is sample.params["cfg"]

    incomplete = SimpleNamespace(
        runtime_randomization=stale,
        events=SimpleNamespace(sample_physics=SimpleNamespace(params={"cfg": stale})),
    )
    with pytest.raises(ValueError, match="initialize_curriculum"):
        synchronize_runtime_randomization_events(incomplete, runtime)
    assert incomplete.runtime_randomization is stale
    assert incomplete.events.sample_physics.params["cfg"] is stale


def test_kit_exception_writes_failed_report_before_app_close(tmp_path: Path) -> None:
    output = tmp_path / "feasibility_report.json"
    template = _pending_report()
    write_json_atomic(output, template)
    args = SimpleNamespace(output=output)
    events: list[str] = []

    def scan_fn(_args, _app_args):
        events.append("scan")
        raise RuntimeError("physx scan exploded")

    class FakeApp:
        def close(self) -> None:
            report = json.loads(output.read_text(encoding="utf-8"))
            assert report["status"] == "failed"
            assert report["metadata"]["failure_phase"] == "kit_runtime"
            assert "physx scan exploded" in report["failures"][0]
            events.append("close")

    exit_code = validate_feasibility._run_scan_with_app(
        args,
        SimpleNamespace(),
        FakeApp(),
        template,
        scan_fn=scan_fn,
    )

    assert exit_code == 1
    assert events == ["scan", "close"]


@pytest.mark.parametrize(
    "scan_fn",
    (
        lambda _args, _app_args: 0,
        lambda _args, _app_args: (_ for _ in ()).throw(SystemExit(0)),
    ),
)
def test_scan_cannot_report_false_success(
    tmp_path: Path, scan_fn
) -> None:
    output = tmp_path / "feasibility_report.json"
    template = _pending_report()
    write_json_atomic(output, template)
    args = SimpleNamespace(output=output)
    closed = False

    class FakeApp:
        def close(self) -> None:
            nonlocal closed
            closed = True

    exit_code = validate_feasibility._run_scan_with_app(
        args,
        SimpleNamespace(),
        FakeApp(),
        template,
        scan_fn=scan_fn,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert closed
    assert report["status"] == "failed"


def test_app_close_exception_overwrites_passed_report_and_exits_one(
    tmp_path: Path,
) -> None:
    output = tmp_path / "feasibility_report.json"
    template = _pending_report()
    write_json_atomic(output, template)
    args = SimpleNamespace(output=output)

    def scan_fn(_args, _app_args):
        passed = {**template, "status": "passed", "failures": []}
        write_json_atomic(output, passed)
        return 0

    class FailingCloseApp:
        def close(self) -> None:
            raise RuntimeError("Kit close failed")

    exit_code = validate_feasibility._run_scan_with_app(
        args,
        SimpleNamespace(),
        FailingCloseApp(),
        template,
        scan_fn=scan_fn,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["metadata"]["failure_phase"] == "simulation_app_close"
    assert "Kit close failed" in report["failures"][0]
