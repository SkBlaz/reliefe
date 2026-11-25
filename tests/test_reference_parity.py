"""Frozen outputs of the original implementations. No foreign runtime is used."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from reliefe import FeatureRanker, list_methods
from reliefe.corelearn import CORE_METHODS
from reliefe.legacy_relief import OPTIONS


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(
            (Path(__file__).parent / "fixtures/reference_scores.json").read_text()
        )

    def compare(self, method, x, y, expected, **params):
        with self.subTest(method=method, options=params), patch(
            "subprocess.Popen", side_effect=AssertionError
        ):
            result = FeatureRanker(
                method, task=list_methods()[method].tasks[0], method_params=params
            ).fit(x, y)
            values = result.scores_ if result.scores_ is not None else result.ranks_
            np.testing.assert_allclose(values, expected, rtol=1e-9, atol=1e-10)

    def test_all_reference_criteria(self):
        for case in ("continuous", "True", "False"):
            if case == "continuous":
                x = np.random.RandomState(903).uniform(size=(30, 4))
                y = np.tile(np.arange(3), 10)
                response = x[:, 0] + x[:, 1] ** 2
                expected = self.reference["core_continuous"]
            else:
                x = np.random.RandomState(319).randint(0, 4, (36, 4)).astype(float)
                y = (x[:, 0] + x[:, 1]).astype(int) % 3
                response = x[:, 0] + 0.7 * x[:, 3]
                expected = self.reference["core_integer"][case]
            self.assertEqual(set(expected), set(CORE_METHODS))
            for method, values in expected.items():
                target = (
                    y
                    if list_methods()[method].tasks[0] == "classification"
                    else response
                )
                self.compare(
                    method,
                    x,
                    target,
                    values,
                    categorical_features=list(range(4)) if case == "True" else [],
                )

    def test_cost_sensitive_reference(self):
        x = np.random.RandomState(299).uniform(size=(53, 4))
        y = np.repeat([0, 1, 2], [11, 18, 24])
        for m, b in self.reference["core_cost"].items():
            self.compare(m, x, y, b, cost_matrix=[[0, 2, 3], [1, 0, 7], [5, 3, 0]])

    def test_historical_reference_weights(self):
        x = np.random.RandomState(903).uniform(size=(30, 4))
        y = np.tile(np.arange(3), 10)
        self.assertEqual(set(self.reference["relief_continuous"]), set(OPTIONS))
        for m, b in self.reference["relief_continuous"].items():
            self.compare(
                m, x, y, b, **({"k": 3} if m in ("reliefmss", "ec-relieff") else {})
            )

    def test_regression_models_and_split_sampling(self):
        rng = np.random.RandomState(129)
        x = rng.uniform(size=(74, 4))
        x[:, 0] = rng.randint(0, 9, 74)
        y = x[:, 1] ** 2 + 0.5 * x[:, 3] - 0.02 * x[:, 0]
        for key, expected in self.reference["core_model_options"].items():
            self.compare(
                key[:-1],
                x,
                y,
                expected,
                modelTypeReg=int(key[-1]),
                categorical_features=[0],
            )

    def test_seeded_subsets_and_discrete_updates(self):
        rng = np.random.RandomState(41)
        y = np.repeat([0, 1, 2], [12, 16, 20])
        options = {
            "irelief": {"k_width": 1.3, "max_iter": 6, "initial_w_div": 4.0},
            "reliefmss": {"k": 2, "m": 10},
            "boostedsurf": {"m": 12, "phi": 1},
            "iterative-relief": {"m": 8, "max_iter": 4, "min_incl": 2},
            "ec-relieff": {"m": 10, "k": 2},
        }
        for case in ("numeric", "discrete"):
            x = (
                rng.uniform(0.05, 1, (48, 5))
                if case == "numeric"
                else rng.randint(0, 5, (48, 5)).astype(float)
            )
            for name, expected in self.reference["relief_sampled_" + case].items():
                self.compare(
                    name,
                    x,
                    y,
                    expected,
                    **options[name],
                    f_type="discrete" if case == "discrete" else "continuous",
                )

    def test_boostedsurf_initializes_distances_before_first_query(self):
        # Relief.jl 0.2.0, rand(MersenneTwister(18), 48, 5), Random.seed!(3).
        # phi=1 initializes the upstream distance buffer before every query.
        # Its refresh always assigns unit weights, so later refreshes are identical.
        x = np.loadtxt(
            Path(__file__).parent / "fixtures/boostedsurf_seed3.csv", delimiter=","
        )
        y = np.tile([1, 2, 3], 16)
        expected = [
            0.1247273758087881,
            0.10534782123787005,
            0.11492547315103457,
            0.1385367735923368,
            0.07985779799274981,
        ]

        def fit():
            return FeatureRanker("boostedsurf", random_state=3).fit(x, y).scores_

        scores = fit()
        np.testing.assert_array_equal(scores, fit())
        np.testing.assert_allclose(scores, expected, atol=1e-10, rtol=1e-9)

    def test_validations_and_degenerate_inputs(self):
        from reliefe.legacy_relief import _Sampler

        for k, expected in [
            (1, [62]),
            (2, [62, 44]),
            (3, [62, 45, 47]),
            (12, [62, 45, 48, 71, 68, 20, 63, 69, 50, 36, 7, 67]),
        ]:
            np.testing.assert_array_equal(_Sampler(42).sample(100, k), expected)
        sampler = _Sampler(42)
        sampler.state = [0, 1, 0, 0]
        self.assertIn(sampler.index(3), range(3))
        y = np.repeat([0, 1], 8)
        x = np.c_[np.arange(16), np.ones(16), y]
        for m, options in [
            ("core-gini", {"categorical_features": [3]}),
            ("core-dkm", {"cost_matrix": [1]}),
            ("core-relieffexprank", {"kNearestEqual": 0}),
            ("irelief", {"other": 3}),
            ("irelief", {"max_iter": 0}),
            ("boostedsurf", {"m": 19}),
        ]:
            with self.assertRaises(ValueError):
                FeatureRanker(m, method_params=options).fit(x, y)
        with self.assertRaisesRegex(ValueError, "two rows"):
            FeatureRanker("reliefmss").fit(x, np.arange(len(y)))
        with patch(
            "reliefe.legacy_relief._nb_accuracy", return_value=0
        ), self.assertRaisesRegex(ValueError, "candidate"):
            FeatureRanker("ec-relieff").fit(x + 0.2, y)
        scores = (
            FeatureRanker("iterative-relief", random_state=None)
            .fit(np.zeros_like(x), y)
            .scores_
        )
        np.testing.assert_array_equal(scores, np.ones(x.shape[1]))
        for mode in (2, 3, 4):
            scores = (
                FeatureRanker(
                    "core-impurityhellinger",
                    random_state=None,
                    method_params={"multiclassEvaluation": mode},
                )
                .fit(x, y)
                .scores_
            )
            self.assertGreater(scores[2], scores[1])
        result = FeatureRanker(
            "core-relief", method_params={"ReliefIterations": 5}
        ).fit(x, y)
        self.assertEqual(result.ranks_.shape, (3,))
