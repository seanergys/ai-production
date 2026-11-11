"""A recorded eval result must survive the run that produced it.

A measurement that is not stored cannot be compared to the next one, so a regression is
invisible by construction — the suite can only ever report "today", never "worse than before".
"""

from __future__ import annotations

import os

from typer.testing import CliRunner


def _db(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATFORM_DB", str(tmp_path / "platform.db"))
    monkeypatch.setenv("EXAMLOPS_DB_BACKEND", "sqlite")
    monkeypatch.delenv("EXAMLOPS_POSTGRES_DSN", raising=False)


def test_a_recorded_run_can_be_read_back_as_a_trend(tmp_path, monkeypatch) -> None:
    _db(tmp_path, monkeypatch)
    from examlops.data.evaluation import get_eval_results, record_eval_result

    record_eval_result(
        suite="cli-coverage",
        model="skipper",
        scores={"pass_rate": 0.948, "ambiguity_rate": 0.052, "error_rate": 0.0},
        run_id="r1",
        sample_size=363,
    )
    rows = get_eval_results("skipper", suite="cli-coverage")
    assert {r["metric"] for r in rows} == {"pass_rate", "ambiguity_rate", "error_rate"}
    assert next(r for r in rows if r["metric"] == "pass_rate")["sample_size"] == 363


def test_eval_history_prints_what_was_recorded(tmp_path, monkeypatch) -> None:
    """A number the CLI can write but not read is still not a trend."""
    _db(tmp_path, monkeypatch)
    from examlops.cli.main import app
    from examlops.data.evaluation import record_eval_result

    record_eval_result(
        suite="cli-coverage",
        model="skipper",
        scores={"pass_rate": 0.948},
        run_id="r1",
        sample_size=363,
    )
    env = {**os.environ}
    res = CliRunner().invoke(app, ["eval", "history", "skipper"], env=env)
    assert res.exit_code == 0, res.output
    assert "cli-coverage" in res.output and "0.948" in res.output


def test_eval_history_says_so_when_there_is_nothing_recorded(tmp_path, monkeypatch) -> None:
    """An empty history and a broken query must not look alike."""
    _db(tmp_path, monkeypatch)
    from examlops.cli.main import app

    res = CliRunner().invoke(app, ["eval", "history", "nobody"], env={**os.environ})
    assert res.exit_code == 0
    assert "no" in res.output.lower()
