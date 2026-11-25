from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from reliefe.utils import feature_ranking_wrapper, load_arff


class FeatureIdentityTests(unittest.TestCase):
    def test_noncontiguous_reordered_nominal_source_columns(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "columns.arff"
            path.write_text("@relation demo\n@attribute first numeric\n@attribute y {n,y}\n"
                            "@attribute skip numeric\n@attribute category {a,b,c}\n"
                            "@data\n1,n,10,a\n2,y,20,b\n3,n,30,c\n")
            x, _, ranges = load_arff(path, [3, 0], 1)
            self.assertEqual(ranges, {0: (0, 3), 1: (3, 4)})
            self.assertEqual(x.shape, (3, 4))
            scores = feature_ranking_wrapper(path, [3, 0], 1, lambda x, y: [1, 2, 3, 10])
            self.assertEqual(scores, {3: 6, 0: 10})
            _, _, original = load_arff(path, None, 1, source_index_ranges=True)
            self.assertEqual(original, {0: (0, 1), 2: (1, 2), 3: (2, 5)})
