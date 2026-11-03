"""Requirement traceability check.

Fails if any requirement in docs/SPEC.md belonging to the current or an earlier sprint
is not named by at least one test, or if a test names a requirement id the spec does not
have. Run: python scripts/trace.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "SPEC.md"
TESTS = ROOT / "tests"
ID = re.compile(r"FR-LM-\d{2}")


def parse_spec(text: str) -> tuple[int, dict[str, int]]:
    current = re.search(r"^Current sprint:\s*(\d+)", text, re.M)
    if not current:
        raise SystemExit("docs/SPEC.md has no 'Current sprint: N' line")
    reqs: dict[str, int] = {}
    for m in re.finditer(r"^#{3,4} (FR-LM-\d{2}).*?\nSprint:\s*(\d+)", text, re.M | re.S):
        reqs[m.group(1)] = int(m.group(2))
    return int(current.group(1)), reqs


def main() -> int:
    current, reqs = parse_spec(SPEC.read_text())
    named: dict[str, set[str]] = {}
    for path in sorted(TESTS.rglob("test_*.py")):
        for rid in ID.findall(path.read_text()):
            named.setdefault(rid, set()).add(path.name)

    due = sorted(r for r, s in reqs.items() if s <= current)
    missing = [r for r in due if r not in named]
    unknown = sorted(r for r in named if r not in reqs)

    print(f"Current sprint {current}: {len(due)} requirements due, {len(reqs)} in spec")
    for r in due:
        files = ", ".join(sorted(named.get(r, []))) or "NO TEST"
        print(f"  {r}  {files}")
    if unknown:
        print(f"Tests name ids not in the spec: {', '.join(unknown)}")
    if missing:
        print(f"Requirements with no test: {', '.join(missing)}")
    return 1 if missing or unknown else 0


if __name__ == "__main__":
    sys.exit(main())
