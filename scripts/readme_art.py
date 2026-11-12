"""Draw the README's banner, figures, charts and icons as SVG: python scripts/readme_art.py

Needs `pip install -e ".[dev,docs]"` (fontTools and HarfBuzz, and pytest to count the tests).
Text is set in Inter (SIL Open Font License, docs/assets/fonts/) and turned into outlines,
because GitHub shows README images without web fonts. The NYC numbers come from
docs/assets/charts/nyc.json, written by `python -m lastfirstdate.nyc_report --json`; the stats
are counted from the repository; the others are quoted from the sprint notes named beside
them. Colours follow the validated default palette of the dataviz method.

Sized for phones. The README shows every figure but the banner at its drawn width (600 px) on a
computer and shrinks it to about 380 px on a phone, so text is drawn at 15 px or more and
reads at 9.5 px or more there. The banner fills the page on both, so its text is larger still.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
FONTS = ASSETS / "fonts"
CHARTS = ASSETS / "charts"
ICONS = ASSETS / "icons"

THEMES = {
    "light": {
        "surface": "#ffffff",
        "raised": "#f6f8fa",
        "border": "#d1d9e0",
        "ink": "#1f2328",
        "ink2": "#59636e",
        "muted": "#818b98",
        "grid": "#eaeef2",
        "base": "#c9d1d9",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "dim": "#c3c9d1",
        "on": ["#ffffff", "#ffffff", "#ffffff", "#0b0b0b"],
        "target": "#1f2328",
    },
    "dark": {
        "surface": "#0d1117",
        "raised": "#151b23",
        "border": "#3d444d",
        "ink": "#f0f6fc",
        "ink2": "#9198a1",
        "muted": "#7a828e",
        "grid": "#212830",
        "base": "#3d444d",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "dim": "#3d444d",
        "on": ["#ffffff", "#ffffff", "#ffffff", "#0b0b0b"],
        "target": "#f0f6fc",
    },
}
ICON_INK = "#3380dd"  # one mid blue for the icons, legible on GitHub's light and dark pages

W = 600  # the width of every figure but the banner
PAD = 24
TITLE, SUB, LABEL, TICK, LEGEND = 23, 16, 17.5, 15, 16
BAR_H, PITCH = 20, 64  # horizontal bars: thickness, and the height of one labelled row

# Quoted results (the NYC numbers are read from nyc.json instead)
RUST = [  # docs/sprints/rust-core.md: speed-up of each hot loop, Rust over Python (low, high)
    ("Looks taste (calibration fits)", 3.6, 3.6, "3.6×"),
    ("Fair daily lists", 10.0, 10.0, "10×"),
    ("Ratio gate", 28.0, 81.0, "28× to 81×"),
    ("Date windows (per call)", 2.0, 75.0, "2× to 75×"),  # rust-core.md, section 10
]
PROMPTS = [  # workspace/process/03-test-runs/results-sprint-13.md: MiniLM cosine similarity
    ("rock climbing · bouldering", 0.67),
    ("Italian cooking · carbonara", 0.52),
    ("settle down · long-term partner", 0.45),
    ("Italian cooking · marathon", 0.20),
    ("rock climbing · opera", 0.13),
]


# ---------------------------------------------------------------- text as outlines


class Font:
    def __init__(self, path: Path):
        tt = TTFont(path)
        tt.flavor = None
        buf = io.BytesIO()
        tt.save(buf)
        data = buf.getvalue()
        self.tt = TTFont(io.BytesIO(data))
        self.glyphs = self.tt.getGlyphSet()
        self.order = self.tt.getGlyphOrder()
        self.upem = self.tt["head"].unitsPerEm
        self.hb = hb.Font(hb.Face(data))

    def _shape(self, text: str, tnum: bool):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb, buf, {"kern": True, "liga": True, "tnum": tnum})
        return buf.glyph_infos, buf.glyph_positions

    def width(self, text: str, size: float, spacing: float = 0, tnum: bool = False) -> float:
        _, pos = self._shape(text, tnum)
        return sum(p.x_advance for p in pos) * size / self.upem + spacing * max(len(pos) - 1, 0)

    def path(
        self, text, x, y, size, fill, anchor="start", spacing=0.0, tnum=False, halo=None
    ) -> str:
        """Text as one outline path; `halo` draws a ring of that colour behind it, so the text
        stays legible where it crosses a line."""
        infos, pos = self._shape(text, tnum)
        s = size / self.upem
        w = self.width(text, size, spacing, tnum)
        x0 = x - (w / 2 if anchor == "middle" else w if anchor == "end" else 0)
        pen = SVGPathPen(self.glyphs, ntos=lambda v: f"{v:.1f}".rstrip("0").rstrip("."))
        for info, p in zip(infos, pos, strict=True):
            t = TransformPen(pen, (s, 0, 0, -s, x0 + p.x_offset * s, y - p.y_offset * s))
            self.glyphs[self.order[info.codepoint]].draw(t)
            x0 += p.x_advance * s + spacing
        d = pen.getCommands()
        ring = (
            f'<path fill="none" stroke="{halo}" stroke-width="4" stroke-linejoin="round" d="{d}"/>'
            if halo
            else ""
        )
        return f'{ring}<path fill="{fill}" d="{d}"/>'


REGULAR, SEMIBOLD, BOLD = (Font(FONTS / f"inter-latin-{w}-normal.woff") for w in (400, 600, 800))


def wrap(font: Font, text: str, size: float, width: float) -> list[str]:
    """Split text into lines no wider than `width`."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and font.width(trial, size) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return [*lines, line] if line else lines


# ---------------------------------------------------------------- shared pieces


def svg(width: float, height: float, label: str, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{label}">'
        + "".join(body)
        + "</svg>\n"
    )


def card(t: dict, width: float, height: float) -> str:
    return (
        f'<rect x="0.5" y="0.5" width="{width - 1:.0f}" height="{height - 1:.0f}" rx="12" '
        f'fill="{t["surface"]}" stroke="{t["border"]}"/>'
    )


def subtitle(text: str) -> list[str]:
    return wrap(REGULAR, text, SUB, W - 2 * PAD)


def frame(t: dict, height: float, title: str, sub: list[str]) -> list[str]:
    """The card, title and subtitle lines of a chart; the last subtitle line sits at
    `below(sub)`."""
    assert SEMIBOLD.width(title, TITLE) <= W - 2 * PAD, title
    body = [card(t, W, height), SEMIBOLD.path(title, PAD, 44, TITLE, t["ink"])]
    body += [REGULAR.path(line, PAD, 71 + 22 * k, SUB, t["ink2"]) for k, line in enumerate(sub)]
    return body


def below(sub: list[str]) -> float:
    return 71 + 22 * (len(sub) - 1)


def column(x, y_base, w, h, fill) -> str:
    """A column rising from the baseline: 4px rounded top, square at the baseline."""
    r = min(4, h / 2, w / 2)
    top = y_base - h
    return (
        f'<path fill="{fill}" d="M{x:.1f} {y_base:.1f}V{top + r:.1f}Q{x:.1f} {top:.1f} '
        f"{x + r:.1f} {top:.1f}H{x + w - r:.1f}Q{x + w:.1f} {top:.1f} {x + w:.1f} {top + r:.1f}"
        f'V{y_base:.1f}Z"/>'
    )


def bar(x_base, y, w, h, fill, opacity=1.0) -> str:
    """A bar growing right from the baseline: 4px rounded end, square at the baseline."""
    r = min(4, w / 2, h / 2)
    end = x_base + w
    alpha = f' fill-opacity="{opacity}"' if opacity < 1 else ""
    return (
        f'<path fill="{fill}"{alpha} d="M{x_base:.1f} {y:.1f}H{end - r:.1f}Q{end:.1f} {y:.1f} '
        f"{end:.1f} {y + r:.1f}V{y + h - r:.1f}Q{end:.1f} {y + h:.1f} {end - r:.1f} {y + h:.1f}"
        f'H{x_base:.1f}Z"/>'
    )


def legend(t: dict, x: float, y: float, names: list[str], cols: int = 0) -> list[str]:
    """Swatches and names in one row, or in a grid of `cols` columns."""
    out = []
    for k, name in enumerate(names):
        if cols:
            cx, cy = x + (W - 2 * PAD) / cols * (k % cols), y + 28 * (k // cols)
        else:
            cx, cy = x, y
            x += 20 + REGULAR.width(name, LEGEND) + 26
        out.append(f'<rect x="{cx:.1f}" y="{cy - 11.5:.1f}" width="12" height="12" rx="3" '
                   f'fill="{t["series"][k]}"/>')  # fmt: skip
        out.append(REGULAR.path(name, cx + 20, cy, LEGEND, t["ink2"]))
    return out


def bar_rows(t: dict, top: float, rows: list[tuple], vmax: float, room: float) -> list[str]:
    """Horizontal bars, each with its label above it, so the bars can use the whole width.
    Rows are (label, low, high, value text, colour); where low < high the bar is solid up to
    low and pale from there to high."""
    span = W - 2 * PAD - room
    body = []
    for k, (label, low, high, text, fill) in enumerate(rows):
        y = top + k * PITCH
        body.append(REGULAR.path(label, PAD, y + 17, LABEL, t["ink"]))
        by = y + 27
        w_high = max(span * high / vmax, 6)
        if low < high:
            w_low = max(span * low / vmax, 4)
            body.append(bar(PAD + w_low + 2, by, w_high - w_low - 2, BAR_H, fill, opacity=0.4))
            body.append(f'<rect x="{PAD}" y="{by:.1f}" width="{w_low:.1f}" height="{BAR_H}" '
                        f'fill="{fill}"/>')  # fmt: skip
        else:
            body.append(bar(PAD, by, w_high, BAR_H, fill))
        body.append(SEMIBOLD.path(text, PAD + w_high + 8, by + 15.5, LABEL, t["ink"], tnum=True))
    return body


# ---------------------------------------------------------------- charts


def promise_chart(t: dict, nyc: dict) -> str:
    """Men with 3 delivered dates, by billing cycle, in a 300- and a 500-woman zone."""
    zones = [("300 women · 100 men", nyc["300"]), ("500 women · 167 men", nyc["500"])]
    sub = subtitle("Men who got their 3 dates, per billing cycle · 5 test zones of each size, "
                   "3 women per paying man")  # fmt: skip
    legend_y = below(sub) + 36
    top = legend_y + 46  # the 100% line
    base = top + 190  # the 0% line
    height = base + 52
    left, right = PAD + 46, W - PAD - 54  # room for the axis and for the target's label
    body = frame(t, height, "Bigger zones keep the 3-date promise", sub)
    body += legend(t, PAD, legend_y, [z[0] for z in zones])
    span = base - top
    for v in (0, 25, 50, 75, 100):
        y = base - span * v / 100
        color = t["base"] if v == 0 else t["grid"]
        body.append(f'<line x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}" '
                    f'stroke="{color}" stroke-width="1"/>')  # fmt: skip
        body.append(REGULAR.path(f"{v}%", left - 10, y + 5, TICK, t["muted"], "end", tnum=True))
    y90 = base - span * 0.9
    body.append(f'<line x1="{left}" x2="{right}" y1="{y90:.1f}" y2="{y90:.1f}" '
                f'stroke="{t["ink2"]}" stroke-width="1.5" stroke-dasharray="6 4"/>')  # fmt: skip
    body.append(SEMIBOLD.path("90%", right + 8, y90 + 5, TICK, t["ink"], tnum=True))
    body.append(REGULAR.path("target", right + 8, y90 + 23, TICK, t["ink2"]))
    group = (right - left) / 3
    bw, gap = 24, 10
    for c in range(3):
        cx = left + group * (c + 0.5)
        for k, (_, z) in enumerate(zones):
            v = z["per_cycle"][c] * 100
            x = cx - bw - gap / 2 + k * (bw + gap)
            body.append(column(x, base, bw, span * v / 100, t["series"][k]))
            label_y = base - span * v / 100 - 8
            if label_y - 12 < y90 < label_y + 3:  # the target line would cross it: lift it above
                label_y = y90 - 6
            body.append(SEMIBOLD.path(f"{v:.0f}%", x + bw / 2, label_y, TICK, t["ink"], "middle",
                                      tnum=True, halo=t["surface"]))  # fmt: skip
        body.append(REGULAR.path(f"Cycle {c + 1}", cx, base + 30, LABEL, t["ink2"], "middle"))
    return svg(W, height, "Men who got their 3 dates per billing cycle, by zone size", body)


def ordering_chart(t: dict, nyc: dict) -> str:
    """Chance both say yes in the top 10, by what orders the list (300-woman zones)."""
    names = {
        "random order": "Random order",
        "personality": "Personality only",
        "looks only": "Looks only",
        "mutual": "Mutual (what the engine uses)",
        "one-sided": "Her taste only",
    }
    data = sorted(nyc["300"]["top10_both_yes"].items(), key=lambda kv: kv[1])
    rows = [
        (names[key], v, v, f"{v:.1%}", t["dim"] if key == "random order" else t["series"][0])
        for key, v in data
    ]
    sub = subtitle("Chance both people say yes, top 10 of each list · graded on the members' "
                   "hidden true tastes, 300-woman zones")  # fmt: skip
    top = below(sub) + 26
    height = top + PITCH * len(rows) + 6
    body = frame(t, height, "Personal taste does most of the work", sub)
    body += bar_rows(t, top, rows, 0.25, 70)
    return svg(W, height, "Chance both say yes in the top 10, by list ordering", body)


def outcomes_chart(t: dict, nyc: dict) -> str:
    """What happened to booked dates: one tall segmented bar (300-woman zones)."""
    order = [("completed", "Went ahead"), ("no_show", "Someone didn't show"),
             ("expired", "Payment hold ran out"), ("cancelled", "Cancelled")]  # fmt: skip
    shares = nyc["300"]["outcomes"]
    sub = subtitle("What happened to every booked date over three billing cycles · "
                   "300-woman zones")  # fmt: skip
    y, h = below(sub) + 28, 28
    legend_y = y + h + 38
    height = legend_y + 28 * ((len(order) + 1) // 2) + 8
    body = frame(t, height, "Most booked dates go ahead", sub)
    x = PAD
    total = W - 2 * PAD - 2 * (len(order) - 1)
    for k, (key, _) in enumerate(order):
        w = total * shares.get(key, 0.0)
        if w <= 0:
            continue
        body.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="4" '
                    f'fill="{t["series"][k]}"/>')  # fmt: skip
        label = f"{shares[key]:.0%}"
        if SEMIBOLD.width(label, TICK) + 14 < w:
            body.append(SEMIBOLD.path(label, x + 7, y + 19.5, TICK, t["on"][k], tnum=True))
        x += w + 2
    body += legend(t, PAD, legend_y, [name for _, name in order], cols=2)
    return svg(W, height, "What happened to booked dates", body)


def speed_chart(t: dict) -> str:
    rows = [(name, low, high, text, t["series"][0]) for name, low, high, text in RUST]
    sub = subtitle("Speed-up over the pure-Python twin, which gives identical results · the "
                   "pale part spans the range across cases")  # fmt: skip
    top = below(sub) + 26
    height = top + PITCH * len(rows) + 6
    body = frame(t, height, "The Rust core: same answers, faster", sub)
    body += bar_rows(t, top, rows, max(r[2] for r in RUST), 114)
    return svg(W, height, "Speed-up of the Rust core over Python", body)


def prompts_chart(t: dict) -> str:
    rows = [(name, v, v, f"{v:.2f}", t["series"][0]) for name, v in PROMPTS]
    sub = subtitle("How alike two made-up prompts are, from 0 to 1 · counting shared words "
                   "scores every pair here 0")  # fmt: skip
    top = below(sub) + 26
    height = top + PITCH * len(rows) + 6
    body = frame(t, height, "MiniLM reads meaning, not just words", sub)
    body += bar_rows(t, top, rows, 1.0, 60)
    return svg(W, height, "Similarity of made-up prompt pairs read by MiniLM", body)


# ---------------------------------------------------------------- figures


def counts() -> list[tuple[str, str]]:
    """The stats, counted from the repository: sprint notes, requirement ids in the
    specification, collected Python tests and Rust tests, and the Rust core's best speed-up."""
    sprints = len(list((ROOT / "docs" / "sprints").glob("sprint-*.md")))
    spec = (ROOT / "docs" / "SPEC.md").read_text()
    requirements = len(set(re.findall(r"FR-LM-\d+", spec)))
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    per_file = re.findall(r"^\S+\.py: (\d+)$", out, re.M)
    python = sum(map(int, per_file)) if per_file else out.count("::")
    rust = sum(p.read_text().count("#[test]") for p in (ROOT / "rust").rglob("*.rs"))
    speed = max(r[2] for r in RUST)
    return [
        (f"{sprints}", f"sprints done, 0 to {sprints - 1}"),
        (f"{requirements}", "requirements, each with a test"),
        (f"{python + rust}", f"tests: {python} Python, {rust} Rust"),
        (f"{speed:.0f}×", "faster hot loops, Rust at its best"),
    ]


def stats_grid(t: dict, stats: list[tuple[str, str]]) -> str:
    """Four stat tiles in a 2 x 2 grid: wide tiles keep the labels on one line, large enough
    to read on a phone."""
    tile_w, tile_h = W / 2, 96
    height = 2 * tile_h
    body = [
        card(t, W, height),
        f'<line x1="{W / 2:.0f}" x2="{W / 2:.0f}" y1="20" y2="{height - 20:.0f}" '
        f'stroke="{t["grid"]}" stroke-width="1"/>',
        f'<line x1="20" x2="{W - 20}" y1="{tile_h:.0f}" y2="{tile_h:.0f}" '
        f'stroke="{t["grid"]}" stroke-width="1"/>',
    ]
    for k, (value, label) in enumerate(stats):
        cx, top = tile_w * (k % 2 + 0.5), tile_h * (k // 2)
        assert REGULAR.width(label, 17) <= tile_w - 24, label
        body.append(SEMIBOLD.path(value, cx, top + 54, 40, t["ink"], "middle"))
        body.append(REGULAR.path(label, cx, top + 80, 17, t["ink2"], "middle"))
    return svg(W, height, "; ".join(f"{v} {label}" for v, label in stats), body)


def pipeline(t: dict) -> str:
    """How it works: the eight stages and the loop after every date, top to bottom."""
    x0, x1 = PAD, W - PAD - 32  # stage cards; the lane on their right carries the loop back
    lane = W - PAD - 14
    mid = (x0 + x1) / 2
    stage, learn, promise = t["series"][0], t["series"][2], t["series"][1]
    arrow_ink = t["muted"]
    body: list[str] = []

    def box(x, top, w, title, detail, color, h=None) -> float:
        lines = wrap(REGULAR, detail, 16, w - 36) if isinstance(detail, str) else detail
        h = h or 42 + 20 * len(lines)
        body.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{w:.1f}" height="{h:.1f}" rx="10" '
                    f'fill="{t["raised"]}" stroke="{t["border"]}"/>')  # fmt: skip
        body.append(f'<rect x="{x + 0.5:.1f}" y="{top + 11:.1f}" width="4" height="{h - 22:.1f}" '
                    f'rx="2" fill="{color}"/>')  # fmt: skip
        body.append(SEMIBOLD.path(title, x + 18, top + 27, 18, t["ink"]))
        for k, line in enumerate(lines):
            assert REGULAR.width(line, 16) <= w - 30, line
            body.append(REGULAR.path(line, x + 18, top + 49 + 20 * k, 16, t["ink2"]))
        return top + h

    def arrow(x, a, b, label=None) -> float:
        body.append(f'<line x1="{x:.1f}" y1="{a:.1f}" x2="{x:.1f}" y2="{b - 7:.1f}" '
                    f'stroke="{arrow_ink}" stroke-width="1.5"/>')  # fmt: skip
        body.append(f'<path d="M{x - 5:.1f} {b - 8:.1f}L{x:.1f} {b:.1f}L{x + 5:.1f} {b - 8:.1f}Z" '
                    f'fill="{arrow_ink}"/>')  # fmt: skip
        if label:
            body.append(REGULAR.path(label, x + 12, (a + b) / 2 + 5, TICK, t["ink2"]))
        return b

    y = 22.0
    name = "Member profiles"
    pw = SEMIBOLD.width(name, LEGEND) + 40
    body.append(f'<rect x="{mid - pw / 2:.1f}" y="{y}" width="{pw:.1f}" height="34" rx="17" '
                f'fill="{t["raised"]}" stroke="{t["border"]}"/>')  # fmt: skip
    body.append(SEMIBOLD.path(name, mid, y + 22.5, LEGEND, t["ink"], "middle"))
    y = arrow(mid, y + 34, y + 58)
    y = box(x0, y, x1 - x0, "1 · Filter", "Orientation both ways, age, distance, "
            "dealbreakers, verification", stage)  # fmt: skip

    gap = 14
    cw = (x1 - x0 - gap) / 2
    lc, rc = x0 + cw / 2, x0 + cw + gap + cw / 2
    arrow(lc, y, y + 24)
    y = arrow(rc, y, y + 24)
    split = y
    y = box(x0, y, cw, "2 · Calibration", "About 30 photos to rate", stage)
    y = arrow(lc, y, y + 20)
    y = box(x0, y, cw, "3 · Looks taste", "A model for each member", stage)
    box(x0 + cw + gap, split, cw, "4 · Personality", "Prompts, intent answers and interests",
        stage, h=y - split)  # fmt: skip
    arrow(lc, y, y + 24)
    y = arrow(rc, y, y + 24)

    y = box(x0, y, x1 - x0, "5 · Mutual score", "50% looks, 50% personality, scored both ways",
            stage)  # fmt: skip
    y = arrow(mid, y, y + 24)
    lists_top = y
    y = box(x0, y, x1 - x0, "6 · Daily lists", "Up to 30 a day, attention shared fairly", stage)
    lists_mid = (lists_top + y) / 2
    y = arrow(mid, y, y + 36, "she picks, he accepts")
    y = box(x0, y, x1 - x0, "7 · Booking", "A time both can make, a 15-minute payment hold, "
            "a fair venue", stage)  # fmt: skip
    y = arrow(mid, y, y + 24)

    group_top = y
    body.append(SEMIBOLD.path("After every date: shapes the next day's lists", x0 + 14,
                              group_top + 25, TICK, t["ink2"]))  # fmt: skip
    inner, gap3 = 12, 10
    cw3 = (x1 - x0 - 2 * inner - 2 * gap3) / 3
    xs = [x0 + inner + k * (cw3 + gap3) for k in range(3)]
    top3 = group_top + 38
    ends = [
        box(xs[0], top3, cw3, "8 · Learning", ["sharpens each", "member's taste"], learn),
        box(xs[1], top3, cw3, "Trust", ["from behaviour,", "never looks"], learn),
        box(xs[2], top3, cw3, "Guarantee", ["keeps the", "3-date promise"], promise),
    ]
    group_bottom = max(ends) + inner
    body.append(f'<rect x="{x0}" y="{group_top:.1f}" width="{x1 - x0}" '
                f'height="{group_bottom - group_top:.1f}" rx="12" fill="none" '
                f'stroke="{t["muted"]}" stroke-dasharray="5 4"/>')  # fmt: skip
    gy = (group_top + group_bottom) / 2
    body.append(f'<path d="M{x1} {gy:.1f}H{lane - 8}Q{lane} {gy:.1f} {lane} {gy - 8:.1f}'
                f'V{lists_mid + 8:.1f}Q{lane} {lists_mid:.1f} {lane - 8} {lists_mid:.1f}'
                f'H{x1 + 8}" fill="none" stroke="{arrow_ink}" stroke-width="1.5"/>')  # fmt: skip
    body.append(f'<path d="M{x1 + 8} {lists_mid - 5:.1f}L{x1} {lists_mid:.1f}'
                f'L{x1 + 8} {lists_mid + 5:.1f}Z" fill="{arrow_ink}"/>')  # fmt: skip
    height = group_bottom + 24
    body.insert(0, card(t, W, height))
    label = (
        "How the engine works: member profiles pass the filter, then calibration and the looks "
        "taste, and personality, then the mutual score, daily lists and booking; after every "
        "date, learning, trust and the guarantee shape the next day's lists"
    )
    return svg(W, height, label, body)


# ---------------------------------------------------------------- banner and icons


def banner() -> str:
    """Fills the page on a computer and a phone alike, so its text is drawn large: the
    tagline reads at about 10 px on a phone."""
    width, height = 720, 250
    body = [
        '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#0b1626"/><stop offset="1" stop-color="#1a2b4c"/>'
        '</linearGradient><linearGradient id="her" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#ff7a8a"/><stop offset="1" stop-color="#e2456b"/>'
        '</linearGradient><linearGradient id="him" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#5aa9ff"/><stop offset="1" stop-color="#2f6fd6"/>'
        '</linearGradient><radialGradient id="glow">'
        '<stop offset="0" stop-color="#4f86e8" stop-opacity="0.35"/>'
        '<stop offset="1" stop-color="#4f86e8" stop-opacity="0"/></radialGradient></defs>',
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="20" fill="url(#bg)" '
        'stroke="#24395f" stroke-width="2"/>',
        '<circle cx="590" cy="125" r="130" fill="url(#glow)"/>',
        '<g transform="translate(590 125)">'
        '<circle cx="-32" cy="0" r="62" fill="url(#her)" fill-opacity="0.88"/>'
        '<circle cx="32" cy="0" r="62" fill="url(#him)" fill-opacity="0.88"/>'
        '<path d="M0,-52.9 A62,62 0 0 1 0,52.9 A62,62 0 0 1 0,-52.9 Z" fill="#ffffff" '
        'fill-opacity="0.93"/></g>',
        BOLD.path("match", 590, 130, 14, "#1a2b4c", "middle"),
        SEMIBOLD.path("AI MATCHMAKING ENGINE", 44, 62, 16, "#8fb3e8", spacing=2.4),
        BOLD.path("lastfirstdate", 44, 120, 54, "#ffffff", spacing=-1),
        BOLD.path("matching", 44, 174, 54, "#8fb3e8", spacing=-1),
        REGULAR.path("Mutual, fair, and built around real dates.", 44, 214, 20, "#c7d5ea"),
    ]
    label = "lastfirstdate-matching: the AI matchmaking engine behind lastfirstdate"
    return svg(width, height, label, body)


ICON_SHAPES = {  # 24-unit line icons for the README's feature cards
    "taste": '<path d="M2.5 12C5 7.6 8.3 5.5 12 5.5S19 7.6 21.5 12C19 16.4 15.7 18.5 12 18.5'
    'S5 16.4 2.5 12Z"/><circle cx="12" cy="12" r="3.2"/>',
    "mutual": '<circle cx="9" cy="12" r="6.2"/><circle cx="15" cy="12" r="6.2"/>',
    "fair": '<path d="M12 4.5V19.5M8 19.5H16M5 7.5H19M5 7.5 2.6 13.2H7.4ZM19 7.5 16.6 13.2'
    'H21.4Z"/><circle cx="12" cy="4.5" r="0.6"/>',
    "dates": '<rect x="3.5" y="5" width="17" height="15.5" rx="2.5"/><path d="M3.5 9.8H20.5'
    'M8 3V6.5M16 3V6.5"/><path d="M12 18.1 9.6 15.8A1.55 1.55 0 1 1 12 13.9 1.55 1.55 0 1 1 '
    '14.4 15.8Z"/>',
    "promise": '<path d="M12 3 19.5 6V11.6C19.5 16.1 16.4 19.6 12 21 7.6 19.6 4.5 16.1 4.5 11.6'
    'V6Z"/><path d="M8.6 12.1 11 14.5 15.6 9.7"/>',
    "fast": '<path d="M13.6 2.8 5.2 13.6H11.4L10.4 21.2 18.8 10.4H12.6Z"/>',
}


def icon(shape: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" '
        f'fill="none" stroke="{ICON_INK}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{shape}</svg>\n'
    )


def main() -> None:
    nyc = json.loads((CHARTS / "nyc.json").read_text())["summary"]
    stats = counts()
    (ASSETS / "banner.svg").write_text(banner())
    for mode, t in THEMES.items():
        (ASSETS / f"pipeline-{mode}.svg").write_text(pipeline(t))
        (ASSETS / f"stats-{mode}.svg").write_text(stats_grid(t, stats))
        for name, chart in (
            ("promise", promise_chart(t, nyc)),
            ("ordering", ordering_chart(t, nyc)),
            ("outcomes", outcomes_chart(t, nyc)),
            ("speed", speed_chart(t)),
            ("prompts", prompts_chart(t)),
        ):
            (CHARTS / f"{name}-{mode}.svg").write_text(chart)
    ICONS.mkdir(exist_ok=True)
    for name, shape in ICON_SHAPES.items():
        (ICONS / f"{name}.svg").write_text(icon(shape))
    print(f"wrote the banner, 2 figures and 5 charts in light and dark, and "
          f"{len(ICON_SHAPES)} icons, under {ASSETS}")  # fmt: skip


if __name__ == "__main__":
    main()
