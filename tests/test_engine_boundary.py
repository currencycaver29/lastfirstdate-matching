"""Rule: the engine never reads the hidden ground truth (FR-LM-11, FR-LM-03)."""

import ast
from pathlib import Path

import lastfirstdate

PKG = Path(lastfirstdate.__file__).parent
ENGINE = [
    "app",
    "arrays",
    "billing",
    "booking",
    "config",
    "encoders",
    "fairness",
    "filter",
    "forecast",
    "geo",
    "learning",
    "lists",
    "looks",
    "models",
    "mutual",
    "native",
    "personality",
    "refit",
    "scheduler",
    "trust",
]
TEST_ONLY = {
    "lastfirstdate.synthetic",
    "lastfirstdate.member_behavior",
    "lastfirstdate.month_runs",
    "lastfirstdate.looks_report",
    "lastfirstdate.mutual_report",
    "lastfirstdate.fairness_report",
    "lastfirstdate.learning_report",
    "lastfirstdate.forecast_fit",
    "lastfirstdate.phase1_report",
    "lastfirstdate.cycle_runs",
    "lastfirstdate.gate_report",
    "lastfirstdate.app_fill",
    "lastfirstdate.fixes_report",
    "lastfirstdate.models_report",
    "lastfirstdate.nyc_report",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_engine_modules_do_not_import_test_run_code():
    """FR-LM-11: engine stages cannot see GroundTruth because they never import the test harness."""
    for name in ENGINE:
        bad = _imports(PKG / f"{name}.py") & TEST_ONLY
        assert not bad, f"{name}.py imports {bad}"


def test_engine_modules_never_mention_ground_truth():
    """FR-LM-03: no engine stage refers to the hidden truth fields."""
    for name in ENGINE:
        text = (PKG / f"{name}.py").read_text()
        for word in ("GroundTruth", ".truth", "pickiness", "shared_taste"):
            assert word not in text, f"{name}.py mentions {word}"
