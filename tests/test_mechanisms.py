"""Laplace / Gaussian 机制：标定公式、可重复性与噪声分布性质。"""

import numpy as np
import pytest
from scipy import stats

from ldp_stats import (
    MECHANISM_GAUSSIAN,
    MECHANISM_LAPLACE,
    RandomnessProvider,
    gaussian_sigma,
    laplace_scale,
)
from ldp_stats.errors import InvalidParameterError
from ldp_stats.mechanisms import gaussian_mechanism, laplace_mechanism


# ---------- 参数标定 -----------------------------------------------------
def test_laplace_scale():
    assert laplace_scale(3.0, 0.5) == pytest.approx(6.0)
    assert laplace_scale(0.0, 1.0) == 0.0


def test_gaussian_sigma_classic_formula():
    # σ = (Δ/ε)*sqrt(2 ln(1.25/δ))
    eps, delta, s = 0.5, 1e-5, 2.0
    expected = (s / eps) * np.sqrt(2.0 * np.log(1.25 / delta))
    assert gaussian_sigma(s, eps, delta, method="classic") == pytest.approx(expected)


def test_gaussian_analytic_tighter_than_classic():
    # 解析 Gaussian 标定给出的 σ 不应超过经典闭式界
    eps, delta, s = 0.5, 1e-5, 2.0
    sigma_a = gaussian_sigma(s, eps, delta, method="analytic")
    sigma_c = gaussian_sigma(s, eps, delta, method="classic")
    assert sigma_a <= sigma_c + 1e-9


def test_gaussian_analytic_allows_epsilon_ge_one():
    # classic 拒绝 ε>=1，analytic 正常返回正数
    with pytest.raises(InvalidParameterError):
        gaussian_sigma(1.0, 1.0, 1e-5, method="classic")
    assert gaussian_sigma(1.0, 1.0, 1e-5, method="analytic") > 0


@pytest.mark.parametrize("epsilon", [0, -1.0, float("nan"), float("inf")])
def test_invalid_epsilon(epsilon):
    with pytest.raises(InvalidParameterError):
        laplace_scale(1.0, epsilon)
    with pytest.raises(InvalidParameterError):
        gaussian_sigma(1.0, epsilon, 1e-5)


@pytest.mark.parametrize("delta", [0, 1.0, -0.1, 2.0, float("nan")])
def test_invalid_delta(delta):
    with pytest.raises(InvalidParameterError):
        gaussian_sigma(1.0, 0.5, delta)


def test_unknown_method_and_sensitivity():
    with pytest.raises(InvalidParameterError):
        gaussian_sigma(1.0, 0.5, 1e-5, method="bogus")
    with pytest.raises(InvalidParameterError):
        gaussian_sigma(-1.0, 0.5, 1e-5)
    with pytest.raises(InvalidParameterError):
        laplace_scale(float("inf"), 1.0)


# ---------- 可重复性与随机性 ---------------------------------------------
def test_fixed_seed_reproducible():
    r1 = RandomnessProvider(12345)
    r2 = RandomnessProvider(12345)
    r3 = RandomnessProvider(999)
    assert np.allclose(r1.laplace(1.0, size=1000), r2.laplace(1.0, size=1000))
    assert not np.allclose(r1.normal(1.0, size=1000), r3.normal(1.0, size=1000))


def test_normal_mode_is_non_fixed_and_independent():
    r1 = RandomnessProvider()
    r2 = RandomnessProvider()
    assert r1.is_test is False and r2.is_test is False
    assert r1.seed is None and r2.seed is None
    # 两个独立随机源产生完全相同大向量的概率为零
    assert not np.allclose(r1.normal(1.0, size=1000), r2.normal(1.0, size=1000))


def test_does_not_touch_numpy_global_state():
    # 固定全局种子后，正常模式（seed=None）的结果仍不应由全局种子决定
    np.random.seed(0)
    a = RandomnessProvider().laplace(1.0, size=50)
    np.random.seed(0)
    b = RandomnessProvider().laplace(1.0, size=50)
    assert not np.allclose(a, b)


def test_invalid_seed():
    with pytest.raises(InvalidParameterError):
        RandomnessProvider(seed="not-an-int")


# ---------- 噪声分布基本性质 ---------------------------------------------
def test_laplace_noise_distribution():
    rng = RandomnessProvider(2024)
    scale = 2.0
    noise = rng.laplace(scale, size=60000)
    # 均值≈0、方差≈2b²；KS 检验与 Laplace(0,b) 分布一致
    assert abs(noise.mean()) < 0.05
    assert noise.var() == pytest.approx(2 * scale ** 2, rel=0.05)
    _, p = stats.kstest(noise, "laplace", args=(0.0, scale))
    assert p > 0.001


def test_gaussian_noise_distribution():
    rng = RandomnessProvider(2024)
    sigma = 1.5
    noise = rng.normal(sigma, size=60000)
    assert abs(noise.mean()) < 0.05
    assert noise.std() == pytest.approx(sigma, rel=0.05)
    _, p = stats.kstest(noise, "norm", args=(0.0, sigma))
    assert p > 0.001


def test_mechanism_noise_zero_centered_on_zero_input():
    vals = [
        laplace_mechanism(0.0, 1.0, 1.0, RandomnessProvider(i))[0]
        for i in range(3000)
    ]
    assert abs(np.mean(vals)) < 0.1


def test_gaussian_vector_noise_shape():
    rng = RandomnessProvider(1)
    v, sigma = gaussian_mechanism(np.zeros(5), 2.0, 0.5, 1e-5, rng)
    assert isinstance(v, np.ndarray) and v.shape == (5,)
    assert sigma > 0
