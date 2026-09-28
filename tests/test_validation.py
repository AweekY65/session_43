"""非法参数与空数据的明确报错。"""

import numpy as np
import pytest

from localdp import DPError, InvalidParameterError, LocalDPEngine
from localdp.mechanisms import gaussian_sigma, laplace_scale


class TestInvalidEpsilon:
    @pytest.mark.parametrize("eps", [0.0, -1.0, float("inf"), float("nan")])
    def test_count_rejects_bad_epsilon(self, eps):
        eng = LocalDPEngine(total_epsilon=10.0, seed=0)
        with pytest.raises(InvalidParameterError):
            eng.count(["u1"], epsilon=eps)

    def test_engine_rejects_bad_total_epsilon(self):
        with pytest.raises(InvalidParameterError):
            LocalDPEngine(total_epsilon=0.0)


class TestInvalidDelta:
    @pytest.mark.parametrize("delta", [0.0, -1e-5, 1.0, 1.5, float("nan")])
    def test_gaussian_rejects_bad_delta(self, delta):
        eng = LocalDPEngine(total_epsilon=0.9, total_delta=0.5, seed=0)
        with pytest.raises(InvalidParameterError):
            eng.count(["u1"], epsilon=0.5, mechanism="gaussian", delta=delta)

    def test_gaussian_requires_delta(self):
        eng = LocalDPEngine(total_epsilon=0.9, total_delta=0.5, seed=0)
        with pytest.raises(InvalidParameterError):
            eng.count(["u1"], epsilon=0.5, mechanism="gaussian")

    def test_laplace_rejects_delta(self):
        eng = LocalDPEngine(total_epsilon=10.0, seed=0)
        with pytest.raises(InvalidParameterError):
            eng.count(["u1"], epsilon=1.0, mechanism="laplace", delta=1e-5)


class TestInvalidBounds:
    @pytest.mark.parametrize("lo,hi", [(5.0, 5.0), (10.0, 1.0),
                                       (float("nan"), 1.0), (0.0, float("inf"))])
    def test_sum_rejects_bad_bounds(self, lo, hi):
        eng = LocalDPEngine(total_epsilon=10.0, seed=0)
        with pytest.raises(InvalidParameterError):
            eng.sum(["u1"], [1.0], lower=lo, upper=hi, epsilon=1.0)


class TestInvalidMechanismAndMisc:
    def test_unknown_mechanism(self):
        eng = LocalDPEngine(total_epsilon=10.0, seed=0)
        with pytest.raises(InvalidParameterError):
            eng.count(["u1"], epsilon=1.0, mechanism="not-a-mechanism")

    def test_gaussian_requires_epsilon_below_one(self):
        with pytest.raises(InvalidParameterError):
            gaussian_sigma(1.0, epsilon=1.5, delta=1e-5)

    def test_bad_max_contributions(self):
        with pytest.raises(InvalidParameterError):
            LocalDPEngine(total_epsilon=1.0, max_contributions=0)
        with pytest.raises(InvalidParameterError):
            LocalDPEngine(total_epsilon=1.0, max_contributions=1.5)

    def test_mismatched_input_lengths(self):
        eng = LocalDPEngine(total_epsilon=10.0, seed=0)
        with pytest.raises(InvalidParameterError):
            eng.sum(["u1", "u2"], [1.0], lower=0.0, upper=1.0, epsilon=1.0)

    def test_empty_bins_rejected(self):
        eng = LocalDPEngine(total_epsilon=10.0, seed=0)
        with pytest.raises(InvalidParameterError):
            eng.histogram(["u1"], ["a"], bins=[], epsilon=1.0)

    def test_duplicate_bins_rejected(self):
        eng = LocalDPEngine(total_epsilon=10.0, seed=0)
        with pytest.raises(InvalidParameterError):
            eng.histogram(["u1"], ["a"], bins=["a", "a"], epsilon=1.0)


class TestEmptyData:
    def test_count_on_empty_data(self):
        eng = LocalDPEngine(total_epsilon=10.0, seed=0)
        res = eng.count([], epsilon=1.0)
        assert abs(res.value - 0.0) < 20  # 纯噪声

    def test_sum_on_empty_data(self):
        eng = LocalDPEngine(total_epsilon=10.0, seed=0)
        res = eng.sum([], [], lower=0.0, upper=1.0, epsilon=1.0)
        assert abs(res.value - 0.0) < 20

    def test_histogram_on_empty_data(self):
        eng = LocalDPEngine(total_epsilon=10.0, seed=0)
        res = eng.histogram([], [], bins=["a", "b"], epsilon=1.0)
        assert res.value.shape == (2,)
        assert all(abs(v) < 20 for v in res.value)

    def test_mean_on_empty_data_raises(self):
        eng = LocalDPEngine(total_epsilon=10.0, seed=0)
        with pytest.raises(DPError, match="空数据"):
            eng.mean([], [], lower=0.0, upper=1.0, epsilon=1.0)
