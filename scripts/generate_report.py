"""Script to generate automated quantitative research report."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from volatility_lab.cli import report
from click.testing import CliRunner


def main():
    runner = CliRunner()
    result = runner.invoke(report, ["--ticker", "^NSEI", "--output", "experiments/research_report.md"])
    print(result.output)


if __name__ == "__main__":
    main()
