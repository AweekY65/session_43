"""边界裁剪与 contribution bounding。"""

import numpy as np
import pytest

from localdp import LocalDPEngine
from localdp.bounds import bound_contributions


def test_contribution_bounding_deterministic():
    users = ["a", "a", "a", "b", "b", "c"]
    keep1 = bound_contributions(users, 2)
    keep2 = bound_contributions(list(users), 2)
    assert list(keep1) == list(keep2) == [0, 1, 3, 4, 5]


def test_contribution_bounding_limits_single_user():
    # 单个用户贡献 100 条，限制为 1 后计数应约为 1 而非 100
    users = ["attacker"] * 100
    eng = LocalDPEngine(total_epsilon=10.0, max_contributions=1, seed=3)
    res = eng.count(users, epsilon=1.0)
    assert abs(res.value - 1.0) < 10


def test_contribution_bounding_caps_sensitivity():
    eng = LocalDPEngine(total_epsilon=10.0, max_contributions=3, seed=3)
    res = eng.count(["u1"] * 10, epsilon=1.0)
    assert res.sensitivity == 3.0
    assert abs(res.value - 3.0) < 30


def test_sum_clips_values_to_bounds():
    eng = LocalDPEngine(total_epsilon=10.0, seed=9)
    # 值 1000 超出上界 10，应被裁剪为 10；-5 超出下界 0，应被裁剪为 0
    res = eng.sum(["u1", "u2"], [1000.0, -5.0], lower=0.0, upper=10.0, epsilon=1.0)
    assert abs(res.value - 10.0) < 50  # 真实裁剪和 = 10 + 0


def test_sum_sensitivity_uses_bound_magnitude():
    eng = LocalDPEngine(total_epsilon=10.0, max_contributions=2, seed=9)
    res = eng.sum(["u1"], [3.0], lower=-5.0, upper=10.0, epsilon=1.0)
    # Δ = max_contributions * max(|lower|, |upper|) = 2 * 10
    assert res.sensitivity == 20.0
