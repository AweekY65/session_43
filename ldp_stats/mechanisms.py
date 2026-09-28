"""随机化机制：Laplace 机制与 Gaussian 机制。

- Laplace 机制：对敏感度 Δ 的数值（或向量，按 L1 敏感度逐维加噪），
  噪声尺度 b = Δ / ε，满足纯 (ε, 0)-DP。
- Gaussian 机制：对 L2 敏感度 Δ 的数值（或向量），加 N(0, σ²) 噪声，
  满足 (ε, δ)-DP。σ 的标定支持两种方法：
    * ``"classic"``（默认）：Dwork & Roth (2014) Theorem A.1，
      σ = (Δ / ε) · sqrt(2 ln(1.25 / δ))，要求 0 < ε < 1；
    * ``"analytic"``：Balle & Wang (2018) 解析 Gaussian 机制，
      通过二分搜索满足 ΦΔ(ε,σ) ≤ δ 的最小 σ，界更紧，支持 ε>0。

随机状态全部由 :class:`RandomnessProvider` 在本地管理：
- 测试模式（显式传入 seed）：可重复；
- 正常模式（seed=None）：每次用操作系统熵全新播种，绝不使用固定种子，
  也不触碰 NumPy 的全局随机状态（``np.random.*``）。
"""

import numpy as np
from scipy import stats

from .errors import InvalidParameterError

MECHANISM_LAPLACE = "laplace"
MECHANISM_GAUSSIAN = "gaussian"
_GAUSSIAN_METHODS = ("classic", "analytic")


class RandomnessProvider:
    """本地随机状态提供者。

    Parameters
    ----------
    seed : int or None
        显式整数 -> 测试模式，结果可复现；
        None（默认）-> 正常模式，使用系统熵构造新的 Generator，
        不同实例之间相互独立，且不改变 NumPy 全局随机种子。
    """

    def __init__(self, seed=None):
        if seed is None:
            self._is_test = False
            self._seed = None
        else:
            if not isinstance(seed, (int, np.integer)):
                raise InvalidParameterError("seed 必须是整数或 None，收到 %r" % (seed,))
            self._is_test = True
            self._seed = int(seed)
        self._rng = np.random.default_rng(seed)

    @property
    def is_test(self):
        return self._is_test

    @property
    def seed(self):
        return self._seed

    def laplace(self, scale, size=None):
        return self._rng.laplace(loc=0.0, scale=scale, size=size)

    def normal(self, sigma, size=None):
        return self._rng.normal(loc=0.0, scale=sigma, size=size)


def _check_epsilon(epsilon):
    try:
        eps = float(epsilon)
    except (TypeError, ValueError):
        raise InvalidParameterError("epsilon 必须是正实数，收到 %r" % (epsilon,))
    if not np.isfinite(eps) or eps <= 0:
        raise InvalidParameterError("epsilon 必须是正的有限值，收到 %s" % eps)
    return eps


def _check_delta(delta):
    try:
        d = float(delta)
    except (TypeError, ValueError):
        raise InvalidParameterError("delta 必须是 (0,1) 内的实数，收到 %r" % (delta,))
    if not np.isfinite(d) or not (0.0 < d < 1.0):
        raise InvalidParameterError("Gaussian 机制要求 0 < delta < 1，收到 %s" % d)
    return d


def _check_sensitivity(sensitivity):
    try:
        s = float(sensitivity)
    except (TypeError, ValueError):
        raise InvalidParameterError("sensitivity 必须是非负有限数，收到 %r" % (sensitivity,))
    if not np.isfinite(s) or s < 0:
        raise InvalidParameterError("sensitivity 必须是非负有限数，收到 %s" % s)
    return s


def laplace_scale(sensitivity, epsilon):
    """Laplace 噪声尺度 b = Δ / ε。"""
    s = _check_sensitivity(sensitivity)
    eps = _check_epsilon(epsilon)
    return s / eps


def gaussian_sigma(sensitivity, epsilon, delta, method="classic"):
    """Gaussian 噪声标准差 σ。

    Parameters
    ----------
    sensitivity : float
        L2 敏感度 Δ（≥0）。
    epsilon : float
        隐私参数 ε。``classic`` 要求 0 < ε < 1；``analytic`` 允许 ε > 0。
    delta : float
        隐私参数 δ，要求 0 < δ < 1。
    method : {"classic", "analytic"}
        标定方法。
    """
    s = _check_sensitivity(sensitivity)
    eps = _check_epsilon(epsilon)
    d = _check_delta(delta)
    if method not in _GAUSSIAN_METHODS:
        raise InvalidParameterError(
            "method 必须是 %s 之一，收到 %r" % (_GAUSSIAN_METHODS, method)
        )
    if method == "classic":
        # Dwork–Roth (2014) 的经典闭式界只在 0 < ε < 1 下成立。
        if eps >= 1.0:
            raise InvalidParameterError(
                "classic Gaussian 标定要求 0 < epsilon < 1，收到 %s；"
                "可改用 method='analytic'" % eps
            )
        return (s / eps) * np.sqrt(2.0 * np.log(1.25 / d))
    # analytic: Balle–Wang (2018)，对 ΦΔ(ε, σ) ≤ δ 二分搜索最小 σ
    if s == 0.0:
        return 0.0
    return _analytic_sigma(s, eps, d)


def _analytic_sigma(sensitivity, epsilon, delta):
    """Balle–Wang 解析 Gaussian 标定（二分搜索）。

    令 Φ 为标准正态 CDF：
      ΦΔ(ε, σ) = Φ(Δ/(2σ) − εσ/Δ) − e^ε · Φ(−Δ/(2σ) − εσ/Δ)
    (ε,δ)-DP 成立当且仅当 ΦΔ(ε,σ) ≤ δ；ΦΔ 随 σ 单调递减，故可二分。
    """
    norm_cdf = stats.norm.cdf
    lo, hi = 0.0, sensitivity  # σ=0 时 ΦΔ=1（不满足），再指数扩张上界
    while _gaussian_tail(sensitivity, epsilon, hi, norm_cdf) > delta:
        lo, hi = hi, hi * 2.0
        if hi > 1e12:  # 数值保护：正常参数下远不会触及
            raise InvalidParameterError("Gaussian σ 标定失败，请检查 epsilon/delta")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _gaussian_tail(sensitivity, epsilon, mid, norm_cdf) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def _gaussian_tail(sensitivity, epsilon, sigma, norm_cdf):
    a = sensitivity / (2.0 * sigma)
    b = epsilon * sigma / sensitivity
    return float(norm_cdf(a - b) - np.exp(epsilon) * norm_cdf(-a - b))


def laplace_mechanism(value, sensitivity, epsilon, randomness):
    """对标量或向量加 Laplace 噪声（向量按 L1 敏感度逐维独立加噪）。"""
    scale = laplace_scale(sensitivity, epsilon)
    raw = np.asarray(value, dtype=np.float64)
    scalar_input = raw.ndim == 0
    noise = randomness.laplace(scale, size=raw.shape)
    result = raw + noise
    return (float(result) if scalar_input else result), scale


def gaussian_mechanism(value, sensitivity, epsilon, delta, randomness,
                       method="classic"):
    """对标量或向量加 Gaussian 噪声（L2 敏感度）。"""
    sigma = gaussian_sigma(sensitivity, epsilon, delta, method=method)
    raw = np.asarray(value, dtype=np.float64)
    scalar_input = raw.ndim == 0
    noise = randomness.normal(sigma, size=raw.shape)
    result = raw + noise
    return (float(result) if scalar_input else result), sigma
