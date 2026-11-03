"""Gate: every requirement due by the current sprint is named by a test (scripts/trace.py)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_due_requirement_has_a_test():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "trace.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
