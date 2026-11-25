"""Regenerate the README tree from its expandable method lists."""

from pathlib import Path
import re
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]


def read_catalogue():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    groups = {}
    for family, body in re.findall(
        r"<summary>(.*?)</summary>(.*?)</details>", readme, re.S
    ):
        groups[family] = [
            (rule, re.findall(r"`([a-z0-9-]+)`", ids))
            for rule, ids in re.findall(r"^- \*\*(.*?)\*\* — (.+)$", body, re.M)
        ]
    return groups


def draw(groups):
    # Each family sits at the midpoint of its leaves, as in a cladogram.
    rows = []
    centers = []
    y = 18

    def path(d):
        rows.append(f'<path d="{d}" stroke="black" fill="none"/>')

    for family, rules in groups.items():
        first = y
        last = first + 21 * (len(rules) - 1)
        center = (first + last) // 2
        centers.append(center)
        path(f"M8 {center} H28")
        rows.append(f'<text x="34" y="{center + 5}">{escape(family)}</text>')
        path(f"M219 {center} H234 M234 {first} V{last}")
        for rule, methods in rules:
            path(f"M234 {y} H248")
            ids = " ".join(methods)
            rows.append(
                f'<text x="254" y="{y + 5}" data-methods="{ids}">'
                f"<title>{escape(', '.join(methods))}</title>{escape(rule)}</text>"
            )
            y += 21
        y += 14

    path(f"M8 {centers[0]} V{centers[-1]}")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 {y - 13}" '
        'role="img" aria-labelledby="title desc">\n<title id="title">Feature ranking</title>\n'
        '<desc id="desc">Branches group ranking methods by their scoring rule. '
        "Expand the corresponding family in the README for all method IDs.</desc>\n"
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<g fill="black" font-family="Georgia, Times New Roman, serif" font-size="16">\n'
        + "\n".join(rows)
        + "\n</g>\n</svg>\n"
    )


if __name__ == "__main__":
    (ROOT / "images/feature-ranking-tree.svg").write_text(
        draw(read_catalogue()), encoding="utf-8"
    )
