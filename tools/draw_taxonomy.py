"""Regenerate images/feature-ranking-tree.svg from the catalogue in README.md."""
from collections import OrderedDict
from pathlib import Path
import re
from textwrap import wrap
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "Marginal filters": ("#925125", "Evaluate one column at a time. These scores are myopic to other predictors."),
    "Neighborhoods": ("#126b60", "Build neighbors across features, then measure each column’s local contribution."),
    "Conditional information": ("#68529a", "Update candidate scores as the selected feature set grows."),
    "Predictive models": ("#255c8a", "Use a fitted predictor, a change in loss, or a comparison to shuffled features."),
    "Aggregation and controls": ("#785366", "Combine target-specific rankings or establish a chance baseline."),
}


def read_catalogue():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("<!-- algorithm-catalog -->")[1].split("<!-- /algorithm-catalog -->")[0]
    groups = OrderedDict()
    for row in table.splitlines():
        methods = re.findall(r"`([a-z][a-z0-9-]*)`", row)
        if methods:
            family, rule, _ = (cell.strip() for cell in row.strip("|").split("|"))
            groups.setdefault(family, []).append((rule, methods))
    return groups


def draw(groups):
    content = []
    add = content.append

    def label(x, y, value, size, color="#1a302f", **attrs):
        options = " ".join('%s="%s"' % (k.replace("_", "-"), v) for k, v in attrs.items())
        add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" {options}>{escape(value)}</text>')

    def line(x1, y1, x2, y2, color):
        add(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{color}" fill="none"/>')

    total = sum(len(ids) for rows in groups.values() for _, ids in rows)
    label(66, 59, "RELIEFE   /   METHOD ATLAS", 17, "#126b60", letter_spacing="3", font_weight="bold")
    label(64, 121, "How features earn their rank", 54, font_family="Georgia, serif")
    label(66, 162, f"{total} public method IDs grouped by the evidence each ranker uses.", 23)
    line(65, 186, 1535, 186, "#84918d")
    for x, heading in [(83, "SCORING FAMILY"), (366, "SCORING RULE"), (735, "METHOD IDS")]:
        label(x, 222, heading, 15, "#455b57", font_weight="bold", letter_spacing="2")
    y = 241
    anchors = []
    for family, rows in groups.items():
        color, description = FAMILIES[family]
        positions = []
        top = y
        for rule, methods in rows:
            lines = wrap("   ".join(methods), width=73, break_long_words=False, break_on_hyphens=False)
            positions.append((y + 30, rule, lines))
            y += 27 * len(lines) + 21
        y = max(y, top + 148)
        add(f'<rect x="54" y="{top}" width="1488" height="{y-top}" fill="{color}" opacity=".045"/>')
        anchor = top + 30
        anchors.append(anchor - 7)
        line(45, anchor - 7, 75, anchor - 7, color)
        add(f'<circle cx="75" cy="{anchor-7}" r="4" fill="{color}"/>')
        for i, part in enumerate(wrap(family, 20)):
            label(87, anchor + 26*i, part, 24, color, font_weight="bold")
        desc_y = anchor + len(wrap(family, 20))*26 + 2
        for i, part in enumerate(wrap(description, 28)):
            label(87, desc_y + i*19, part, 16, "#415652")
        line(341, positions[0][0]-6, 341, positions[-1][0]-6, color)
        line(334, anchor-6, 341, anchor-6, color)
        for pos, rule, lines in positions:
            line(341, pos-6, 356, pos-6, color)
            label(366, pos, rule, 18, font_weight="bold")
            line(688, pos-6, 722, pos-6, color)
            for i, ids in enumerate(lines):
                label(735, pos+27*i, ids, 17, font_family="DejaVu Sans Mono, monospace")
        y += 9
    line(45, anchors[0], 45, anchors[-1], "#81938f")
    for anchor in anchors:
        add(f'<circle cx="45" cy="{anchor}" r="4" fill="#47615c"/>')
    label(66, y+24, "Each ID appears at its primary scoring rule. Several algorithms have both native and reference implementations.", 17)
    height = y + 52
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 {height}" role="img" '
            'aria-labelledby="title description">\n<title id="title">How features earn their rank</title>\n'
            f'<desc id="description">Tree of {total} public ranking methods. Read from scoring family to rule '
            'to method ID. The README contains the same catalogue as text.</desc>\n'
            '<rect width="100%" height="100%" fill="#fcfbf7"/>\n'
            '<g font-family="DejaVu Sans, sans-serif" xml:space="preserve">\n' +
            "\n".join(content) + '\n</g>\n</svg>\n')


if __name__ == "__main__":
    (ROOT / "images/feature-ranking-tree.svg").write_text(draw(read_catalogue()), encoding="utf-8")
