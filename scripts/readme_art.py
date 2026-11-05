"""Draw the README's banner and charts as SVG, in light and dark: python scripts/readme_art.py

Needs `pip install -e ".[docs]"` (fontTools and HarfBuzz). Text is set in Inter (SIL Open
Font License, docs/assets/fonts/) and turned into outlines, because GitHub shows README
images without web fonts. The NYC numbers come from docs/assets/charts/nyc.json, written by
`python -m lastfirstdate.nyc_report --json`; the others are quoted from the sprint notes named
beside them. Colours follow the validated default palette of the dataviz method.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
FONTS = ASSETS / "fonts"
CHARTS = ASSETS / "charts"

THEMES = {
    "light": {
        "surface": "#ffffff",
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

# Quoted results (the NYC numbers are read from nyc.json instead)
RUST = [  # docs/sprints/rust-core.md: speed-up of each hot loop, Rust over Python
    ("Looks taste (calibration fits)", 3.6, "3.6×"),
    ("Fair daily lists", 10.0, "10×"),
    ("Ratio gate", 81.0, "28× to 81×"),
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


def svg(width: int, height: int, label: str, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{label}">'
        + "".join(body)
        + "</svg>\n"
    )


def card(t: dict, width: int, height: int) -> str:
    return (
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="12" '
        f'fill="{t["surface"]}" stroke="{t["border"]}"/>'
    )


def heading(t: dict, title: str, subtitle: str) -> list[str]:
    return [
        SEMIBOLD.path(title, 28, 44, 19, t["ink"]),
        REGULAR.path(subtitle, 28, 68, 13.5, t["ink2"]),
    ]


def column(x, y_base, w, h, fill) -> str:
    """A column rising from the baseline: 4px rounded top, square at the baseline."""
    r = min(4, h / 2, w / 2)
    top = y_base - h
    return (
        f'<path fill="{fill}" d="M{x:.1f} {y_base:.1f}V{top + r:.1f}Q{x:.1f} {top:.1f} '
        f"{x + r:.1f} {top:.1f}H{x + w - r:.1f}Q{x + w:.1f} {top:.1f} {x + w:.1f} {top + r:.1f}"
        f'V{y_base:.1f}Z"/>'
    )


def bar(x_base, y, w, h, fill) -> str:
    """A bar growing right from the baseline: 4px rounded end, square at the baseline."""
    r = min(4, w / 2, h / 2)
    end = x_base + w
    return (
        f'<path fill="{fill}" d="M{x_base:.1f} {y:.1f}H{end - r:.1f}Q{end:.1f} {y:.1f} '
        f"{end:.1f} {y + r:.1f}V{y + h - r:.1f}Q{end:.1f} {y + h:.1f} {end - r:.1f} {y + h:.1f}"
        f'H{x_base:.1f}Z"/>'
    )


def legend(t: dict, x: float, y: float, names: list[str]) -> list[str]:
    out = []
    for k, name in enumerate(names):
        out.append(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" '
                   f'fill="{t["series"][k]}"/>')  # fmt: skip
        out.append(REGULAR.path(name, x + 16, y, 13, t["ink2"]))
        x += 16 + REGULAR.width(name, 13) + 22
    return out


# ---------------------------------------------------------------- charts


def promise_chart(t: dict, nyc: dict) -> str:
    """Men with 3 delivered dates, by billing cycle, in a 300- and a 500-woman zone."""
    W, H = 680, 380
    zones = [("300 women · 100 men", nyc["300"]), ("500 women · 167 men", nyc["500"])]
    left, right, top, base = 76, W - 28, 112, H - 58
    body = [card(t, W, H)]
    body += heading(t, "The three-date promise needs a big enough zone",
                    "Men who got their 3 dates, per billing cycle · 5 test zones each, "
                    "3 women per paying man")  # fmt: skip
    body += legend(t, left, 96, [z[0] for z in zones])
    span = base - top
    for v in (0, 25, 50, 75, 100):
        y = base - span * v / 100
        color = t["base"] if v == 0 else t["grid"]
        body.append(f'<line x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}" '
                    f'stroke="{color}" stroke-width="1"/>')  # fmt: skip
        body.append(REGULAR.path(f"{v}%", left - 10, y + 4.5, 12, t["muted"], "end", tnum=True))
    y90 = base - span * 0.9
    body.append(f'<line x1="{left}" x2="{right}" y1="{y90:.1f}" y2="{y90:.1f}" '
                f'stroke="{t["target"]}" stroke-width="1.5" stroke-dasharray="6 4"/>')  # fmt: skip
    body.append(SEMIBOLD.path("90% target", right, y90 - 7, 12, t["ink"], "end", halo=t["surface"]))
    group = (right - left) / 3
    bw = 24
    for c in range(3):
        cx = left + group * (c + 0.5)
        for k, (_, z) in enumerate(zones):
            v = z["per_cycle"][c] * 100
            x = cx - bw - 1 + k * (bw + 2)
            body.append(column(x, base, bw, span * v / 100, t["series"][k]))
            label_y = base - span * v / 100 - 7
            body.append(
                SEMIBOLD.path(
                    f"{v:.0f}%",
                    x + bw / 2,
                    label_y,
                    12,
                    t["ink"],
                    "middle",
                    tnum=True,
                    halo=t["surface"],
                )
            )
        body.append(REGULAR.path(f"Cycle {c + 1}", cx, base + 24, 13, t["ink2"], "middle"))
    return svg(W, H, "Men who got their 3 dates per billing cycle, by zone size", body)


def ordering_chart(t: dict, nyc: dict) -> str:
    """Chance both say yes in the top 10, by what orders the list (300-woman zones)."""
    names = {
        "random order": "Random order",
        "personality": "Personality only",
        "looks only": "Looks only",
        "mutual": "Mutual (what the engine uses)",
        "one-sided": "Her taste only",
    }
    rows = sorted(nyc["300"]["top10_both_yes"].items(), key=lambda kv: kv[1])
    W = 680
    left, top, row = 236, 106, 40
    H = top + row * len(rows) + 30
    right = W - 90
    body = [card(t, W, H)]
    body += heading(t, "Personal taste does most of the work",
                    "Chance both people say yes, top 10 of each list · from the members' hidden "
                    "true tastes")  # fmt: skip
    vmax = 0.25
    for k, (key, v) in enumerate(rows):
        y = top + k * row
        fill = t["dim"] if key == "random order" else t["series"][0]
        body.append(REGULAR.path(names[key], left - 14, y + 17, 13.5, t["ink"], "end"))
        body.append(bar(left, y + 1, (right - left) * v / vmax, 22, fill))
        body.append(SEMIBOLD.path(f"{v:.1%}", left + (right - left) * v / vmax + 8, y + 17,
                                  13, t["ink"], tnum=True))  # fmt: skip
    body.append(f'<line x1="{left}" x2="{left}" y1="{top - 6}" y2="{top + row * len(rows) - 12}" '
                f'stroke="{t["base"]}"/>')  # fmt: skip
    return svg(W, H, "Chance both say yes in the top 10, by list ordering", body)


def outcomes_chart(t: dict, nyc: dict) -> str:
    """What happened to booked dates: one tall segmented bar (300-woman zones)."""
    order = [("completed", "Went ahead"), ("no_show", "Someone didn't show"),
             ("expired", "Payment hold ran out"), ("cancelled", "Cancelled")]  # fmt: skip
    shares = nyc["300"]["outcomes"]
    W, H = 680, 196
    left, right, y, h = 28, W - 28, 100, 28
    body = [card(t, W, H)]
    body += heading(t, "Most booked dates go ahead",
                    "What happened to every booked date over three billing cycles · "
                    "300-woman zones")  # fmt: skip
    x = left
    total = right - left - 2 * (len(order) - 1)
    for k, (key, _) in enumerate(order):
        w = total * shares.get(key, 0.0)
        if w <= 0:
            continue
        body.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="4" '
                    f'fill="{t["series"][k]}"/>')  # fmt: skip
        label = f"{shares[key]:.0%}"
        if SEMIBOLD.width(label, 13) + 16 < w:
            body.append(SEMIBOLD.path(label, x + 8, y + 19, 13, t["on"][k], tnum=True))
        x += w + 2
    body += legend(t, left, y + h + 34, [name for _, name in order])
    return svg(W, H, "What happened to booked dates", body)


def speed_chart(t: dict) -> str:
    W = 680
    left, top, row = 236, 106, 40
    H = top + row * len(RUST) + 30
    right = W - 120
    body = [card(t, W, H)]
    body += heading(
        t,
        "The Rust core: same answers, faster",
        "Speed-up over the pure-Python twin, which gives identical results",
    )
    vmax = 81.0
    for k, (name, v, label) in enumerate(RUST):
        y = top + k * row
        body.append(REGULAR.path(name, left - 14, y + 17, 13.5, t["ink"], "end"))
        w = max((right - left) * v / vmax, 6)
        body.append(bar(left, y + 1, w, 22, t["series"][0]))
        body.append(SEMIBOLD.path(label, left + w + 8, y + 17, 13, t["ink"], tnum=True))
    body.append(f'<line x1="{left}" x2="{left}" y1="{top - 6}" y2="{top + row * len(RUST) - 12}" '
                f'stroke="{t["base"]}"/>')  # fmt: skip
    return svg(W, H, "Speed-up of the Rust core over Python", body)


def prompts_chart(t: dict) -> str:
    W = 680
    left, top, row = 264, 106, 40
    H = top + row * len(PROMPTS) + 30
    right = W - 80
    body = [card(t, W, H)]
    body += heading(t, "MiniLM reads meaning, not just words",
                    "How alike two made-up prompts are (0 to 1) · plain word counts score "
                    "every pair here 0")  # fmt: skip
    for k, (name, v) in enumerate(PROMPTS):
        y = top + k * row
        body.append(REGULAR.path(name, left - 14, y + 17, 13.5, t["ink"], "end"))
        w = (right - left) * v / 0.8
        body.append(bar(left, y + 1, w, 22, t["series"][0]))
        body.append(SEMIBOLD.path(f"{v:.2f}", left + w + 8, y + 17, 13, t["ink"], tnum=True))
    body.append(f'<line x1="{left}" x2="{left}" y1="{top - 6}" '
                f'y2="{top + row * len(PROMPTS) - 12}" stroke="{t["base"]}"/>')  # fmt: skip
    return svg(W, H, "Similarity of made-up prompt pairs read by MiniLM", body)


# ---------------------------------------------------------------- banner


def banner() -> str:
    W, H = 960, 260
    chips = ["50% looks · 50% personality", "Mutual by design", "Fair daily lists",
             "Real dates, booked"]  # fmt: skip
    body = [
        '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#0b1626"/><stop offset="1" stop-color="#1a2b4c"/>'
        '</linearGradient><linearGradient id="her" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#ff7a8a"/><stop offset="1" stop-color="#e2456b"/>'
        '</linearGradient><linearGradient id="him" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#5aa9ff"/><stop offset="1" stop-color="#2f6fd6"/>'
        "</linearGradient></defs>",
        f'<rect x="1" y="1" width="{W - 2}" height="{H - 2}" rx="22" fill="url(#bg)" '
        'stroke="#24395f" stroke-width="2"/>',
        '<g transform="translate(812 130)">'
        '<circle cx="-38" cy="0" r="72" fill="url(#her)" fill-opacity="0.85"/>'
        '<circle cx="38" cy="0" r="72" fill="url(#him)" fill-opacity="0.85"/>'
        '<path d="M0,-61 A72,72 0 0 1 0,61 A72,72 0 0 1 0,-61 Z" fill="#ffffff" '
        'fill-opacity="0.92"/></g>',
        BOLD.path("match", 812, 135, 15, "#1a2b4c", "middle"),
        SEMIBOLD.path("AI MATCHMAKING ENGINE", 48, 78, 13.5, "#8fb3e8", spacing=2.6),
        BOLD.path("lastfirstdate-matching", 48, 126, 44, "#ffffff", spacing=-0.6),
        REGULAR.path("Mutual, fair, and built around real dates.", 48, 160, 18, "#c7d5ea"),
    ]
    x = 48
    for chip in chips:
        w = SEMIBOLD.width(chip, 13) + 28
        body.append(f'<rect x="{x:.1f}" y="186" width="{w:.1f}" height="30" rx="15" '
                    'fill="#ffffff" fill-opacity="0.08" stroke="#3a5584"/>')  # fmt: skip
        body.append(SEMIBOLD.path(chip, x + w / 2, 205.5, 13, "#e6eefb", "middle"))
        x += w + 8
    label = "lastfirstdate-matching: the AI matchmaking engine behind lastfirstdate"
    return svg(W, H, label, body)


def main() -> None:
    nyc = json.loads((CHARTS / "nyc.json").read_text())["summary"]
    (ASSETS / "banner.svg").write_text(banner())
    for mode, t in THEMES.items():
        for name, chart in (
            ("promise", promise_chart(t, nyc)),
            ("ordering", ordering_chart(t, nyc)),
            ("outcomes", outcomes_chart(t, nyc)),
            ("speed", speed_chart(t)),
            ("prompts", prompts_chart(t)),
        ):
            (CHARTS / f"{name}-{mode}.svg").write_text(chart)
    print(f"wrote {ASSETS / 'banner.svg'} and {len(THEMES) * 5} charts in {CHARTS}")


if __name__ == "__main__":
    main()
