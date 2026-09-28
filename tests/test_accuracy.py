"""统计正确性：固定种子下带噪结果应接近真实值，大样本下误差收敛。"""

import numpy as np
import pytest

from localdp import LocalDPEngine


def test_count_close_to_true_value():
    users = [f"u{i}" for i in range(2000)]
    eng = LocalDPEngine(total_epsilon=10.0, seed=42)
    res = eng.count(users, epsilon=1.0)
    # Laplace scale = 1/1 = 1，误差应以极高概率远小于 50
    assert abs(res.value - 2000) < 50
    assert res.mechanism == "laplace"
    assert res.sensitivity == 1.0


def test_sum_close_to_true_value():
    users = [f"u{i}" for i in range(1000)]
    values = np.linspace(0.0, 10.0, 1000)
    eng = LocalDPEngine(total_epsilon=10.0, seed=7)
    res = eng.sum(users, values, lower=0.0, upper=10.0, epsilon=1.0)
    true_sum = float(values.sum())
    assert abs(res.value - true_sum) < 100  # scale = 10，噪声标准差 ~14
    assert res.sensitivity == 10.0


def test_mean_close_to_true_value():
    rng = np.random.default_rng(0)
    users = [f"u{i}" for i in range(5000)]
    values = rng.uniform(2.0, 8.0, size=5000)
    eng = LocalDPEngine(total_epsilon=10.0, seed=123)
    res = eng.mean(users, values, lower=0.0, upper=10.0, epsilon=2.0)
    assert res.value is not None
    assert abs(res.value - float(values.mean())) < 0.5


def test_histogram_close_to_true_counts():
    users = [f"u{i}" for i in range(3000)]
    categories = ["a", "b", "c"] * 1000
    eng = LocalDPEngine(total_epsilon=10.0, seed=5)
    res = eng.histogram(users, categories, bins=["a", "b", "c"], epsilon=1.0)
    assert isinstance(res.value, np.ndarray)
    assert res.value.shape == (3,)
    for v in res.value:
        assert abs(v - 1000) < 50


def test_histogram_ignores_unknown_categories():
    eng = LocalDPEngine(total_epsilon=10.0, seed=5)
    res = eng.histogram(["u1", "u2", "u3"], ["x", "y", "zzz"], bins=["x", "y"], epsilon=1.0)
    # 真实计数为 [1, 1]，"zzz" 被忽略；噪声 scale=1，容差宽松
    assert all(abs(v - 1.0) < 10 for v in res.value)


def test_gaussian_sum_close_to_true_value():
    users = [f"u{i}" for i in range(1000)]
    values = np.ones(1000)
    eng = LocalDPEngine(total_epsilon=0.9, total_delta=1e-4, seed=11)
    res = eng.sum(users, values, lower=0.0, upper=1.0, epsilon=0.5,
                  mechanism="gaussian", delta=1e-5)
    assert res.mechanism == "gaussian"
    assert res.delta == 1e-5
    assert abs(res.value - 1000.0) < 100
