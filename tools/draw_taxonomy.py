"""Regenerate images/feature-ranking-tree.svg from the catalogue in README.md."""
from collections import OrderedDict
from pathlib import Path
import re
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]


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
    lines = []
    y = 36

    def label(x, value, size, bold=False, ids=()):
        weight = ' font-weight="bold"' if bold else ''
        metadata = ' data-methods="' + ' '.join(ids) + '"' if ids else ''
        lines.append(f'<text x="{x}" y="{y}" font-size="{size}"{weight}{metadata}>{escape(value)}</text>')

    def branches(x, start, children):
        lines.append(f'<path d="M{x} {start} V{children[-1]}" stroke="black" fill="none"/>')
        for at in children:
            lines.append(f'<path d="M{x} {at} h12" stroke="black"/>')

    label(18, "Feature ranking", 26, bold=True)
    families = []
    for family, rules in groups.items():
        y += 36
        families.append(y - 6)
        label(44, family, 20, bold=True)
        start = y + 7
        children = []
        for rule, methods in rules:
            y += 28
            children.append(y - 5)
            label(72, rule, 16, bold=True)
            # Show one name per algorithm in a rule. Keep its public IDs on the leaf.
            names = {}
            for method in methods:
                name = method.split("-", 1)[1] if method.startswith(("numba-", "rebate-")) else method
                names.setdefault(name if name in methods else method, []).append(method)
            row, ids = [], []
            for name, implementations in names.items():
                if row and len(", ".join(row + [name])) > 76:
                    label(336, ", ".join(row), 15, ids=ids)
                    y += 21
                    row, ids = [], []
                row.append(name)
                ids.extend(implementations)
            label(336, ", ".join(row), 15, ids=ids)
        branches(54, start, children)
    branches(26, 45, families)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1060 {y+24}" role="img" '
            'aria-labelledby="title desc">\n<title id="title">Feature ranking</title>\n'
            '<desc id="desc">Scoring families branch into rules and their algorithms. '
            'The README lists each public implementation.</desc>\n'
            '<rect width="100%" height="100%" fill="white"/>\n'
            '<g fill="black" font-family="Arial, Helvetica, sans-serif">\n' +
            "\n".join(lines) + '\n</g>\n</svg>\n')


if __name__ == "__main__":
    (ROOT / "images/feature-ranking-tree.svg").write_text(draw(read_catalogue()), encoding="utf-8")
