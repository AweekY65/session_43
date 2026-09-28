"""参数校验与 Laplace / Gaussian 噪声机制。

仅使用本地随机源（numpy.random.Generator），不访问任何外部服务。
"""

from __future__ import annotations

import math

import numpy as np

from .exceptions import InvalidParameterError


def check_epsilon(epsilon: float) -> float:
    """校验 epsilon：必须为有限正数。"""
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0.0:
        raise InvalidParameterError(
            f"epsilon 必须为有限正数，收到 {epsilon!r}"
        )
    return eps


def check_delta(delta: float) -> float:
    """校验 delta：必须落在开区间 (0, 1)。"""
    d = float(delta)
    if not math.isfinite(d) or not 0.0 < d < 1.0:
        raise InvalidParameterError(
            f"delta 必须满足 0 < delta < 1，收到 {delta!r}"
        )
    return d


def check_bounds(lower: float, upper: float) -> tuple[float, float]:
    """校验数值上下界：必须有限且 lower < upper。"""
    lo, hi = float(lower), float(upper)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise InvalidParameterError(
            f"上下界必须为有限值，收到 lower={lower!r}, upper={upper!r}"
        )
    if not lo < hi:
        raise InvalidParameterError(
            f"要求 lower < upper，收到 lower={lo}, upper={hi}"
        )
    return lo, hi


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    """Laplace 机制的噪声尺度 b = sensitivity / epsilon（纯 epsilon-DP）。"""
    eps = check_epsilon(epsilon)
    if sensitivity <= 0 or not math.isfinite(sensitivity):
        raise InvalidParameterError(
            f"sensitivity 必须为有限正数，收到 {sensitivity!r}"
        )
    return sensitivity / eps


def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """经典 Gaussian 机制的噪声标准差（(epsilon, delta)-DP）。

    sigma = sqrt(2 * ln(1.25 / delta)) * sensitivity / epsilon
    该界在 epsilon < 1 时成立；epsilon >= 1 时拒绝以保证严格性。
    """
    eps = check_epsilon(epsilon)
    d = check_delta(delta)
    if sensitivity <= 0 or not math.isfinite(sensitivity):
        raise InvalidParameterError(
            f"sensitivity 必须为有限正数，收到 {sensitivity!r}"
        )
    if eps >= 1.0:
        raise InvalidParameterError(
            "经典 Gaussian 机制要求 epsilon < 1；请减小 epsilon 或改用 Laplace 机制"
        )
    return math.sqrt(2.0 * math.log(1.25 / d)) * sensitivity / eps


def add_laplace_noise(value, sensitivity: float, epsilon: float, rng: np.random.Generator):
    """对标量或数组添加 Laplace 噪声，返回加噪后的同形结果。"""
    scale = laplace_scale(sensitivity, epsilon)
    noise = rng.laplace(loc=0.0, scale=scale, size=np.shape(value) or None)
    return np.asarray(value, dtype=float) + noise, scale


def add_gaussian_noise(
    value, sensitivity: float, epsilon: float, delta: float, rng: np.random.Generator
):
    """对标量或数组添加 Gaussian 噪声，返回加噪后的同形结果。"""
    sigma = gaussian_sigma(sensitivity, epsilon, delta)
    noise = rng.normal(loc=0.0, scale=sigma, size=np.shape(value) or None)
    return np.asarray(value, dtype=float) + noise, sigma
