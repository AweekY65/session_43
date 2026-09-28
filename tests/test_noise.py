"""噪声分布的基本性质与输出安全性。"""

import math

import numpy as np
import pytest

from localdp import LocalDPEngine
from localdp.mechanisms import gaussian_sigma, laplace_scale


def test_laplace_noise_distribution():
    # Laplace(0, b)：均值 0，方差 2b^2
    b = laplace_scale(sensitivity=2.0, epsilon=0.5)
    assert b == pytest.approx(4.0)
    rng = np.random.default_rng(0)
    samples = rng.laplace(0.0, b, size=200_000)
    assert abs(samples.mean()) < 0.1
    assert samples.var() == pytest.approx(2 * b * b, rel=0.05)


def test_gaussian_noise_distribution():
    sigma = gaussian_sigma(sensitivity=1.0, epsilon=0.5, delta=1e-5)
    expected = math.sqrt(2 * math.log(1.25 / 1e-5)) / 0.5
    assert sigma == pytest.approx(expected)
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, sigma, size=200_000)
    assert abs(samples.mean()) < 0.1
    assert samples.std() == pytest.approx(sigma, rel=0.02)


def test_result_metadata_complete_and_safe():
    eng = LocalDPEngine(total_epsilon=10.0, seed=0)
    res = eng.sum(["u1", "u2"], [3.0, 4.0], lower=0.0, upper=10.0, epsilon=1.0)
    d = res.to_dict()
    # 必须包含的元信息
    for key in ("value", "mechanism", "epsilon", "delta", "sensitivity",
                "noise_parameter", "params"):
        assert key in d
    assert d["mechanism"] == "laplace"
    assert d["epsilon"] == 1.0
    # 不得包含可还原单条原始记录的字段
    forbidden = ("user_ids", "values", "records", "data", "raw")
    for key in forbidden:
        assert key not in d
        assert key not in d["params"]
