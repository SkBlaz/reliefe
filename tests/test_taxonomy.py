import runpy
import unittest
from pathlib import Path
from xml.etree import ElementTree

from reliefe import list_methods


class TaxonomyTests(unittest.TestCase):
    def test_catalogue_and_published_tree_match_registry(self):
        root = Path(__file__).resolve().parents[1]
        generator = runpy.run_path(str(root / "tools/draw_taxonomy.py"))
        catalogue = generator["read_catalogue"]()
        ids = [name for rules in catalogue.values() for _, methods in rules for name in methods]
        self.assertCountEqual(ids, list(list_methods()))
        published = (root / "images/feature-ranking-tree.svg").read_text(encoding="utf-8")
        self.assertEqual(published, generator["draw"](catalogue))
        svg = ElementTree.fromstring(published)
        leaves = svg.findall(".//*[@data-methods]")
        self.assertCountEqual([key for leaf in leaves for key in leaf.attrib["data-methods"].split()], ids)
        colors = {value for node in svg.iter() for key, value in node.attrib.items()
                  if key in ("fill", "stroke")}
        self.assertEqual(colors, {"black", "white", "none"})
