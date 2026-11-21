"""Integration tests for the command-line interface (CLI)."""

import pytest
from click.testing import CliRunner
from volatility_lab.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Volatility Forecasting Lab" in result.output


def test_cli_estimate():
    runner = CliRunner()
    result = runner.invoke(main, ["estimate", "--ticker", "SYNTHETIC_SIM"])
    assert result.exit_code == 0
    assert "Current Volatility Estimates" in result.output
    assert "Parkinson" in result.output


def test_cli_forecast():
    runner = CliRunner()
    result = runner.invoke(main, ["forecast", "--ticker", "SYNTHETIC_SIM", "--horizon", "1"])
    assert result.exit_code == 0
    assert "Volatility Forecasts" in result.output


def test_cli_risk():
    runner = CliRunner()
    result = runner.invoke(main, ["risk", "--ticker", "SYNTHETIC_SIM"])
    assert result.exit_code == 0
    assert "Quantitative Risk Summary" in result.output
    assert "Kupiec POF LR Test" in result.output


def test_cli_report(tmp_path):
    out_file = tmp_path / "test_report.md"
    runner = CliRunner()
    result = runner.invoke(main, ["report", "--ticker", "SYNTHETIC_SIM", "--output", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Volatility Forecasting Lab: Research Report" in content
