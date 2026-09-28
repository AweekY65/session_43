"""隐私预算账户测试：累计、耗尽拒绝、组合方法。"""

import math

import pytest

from ldp_stats import BudgetAccountant
from ldp_stats.errors import BudgetExceededError, InvalidParameterError


# ---------- 基本组合 -----------------------------------------------------
def test_basic_composition_accumulates():
    acc = BudgetAccountant(epsilon=1.0)
    assert acc.check(0.4)
    acc.spend(0.4, label="q1")
    acc.spend(0.3, label="q2")
    assert acc.used_epsilon == pytest.approx(0.7)
    assert acc.remaining_epsilon == pytest.approx(0.3)
    assert acc.n_queries == 2


def test_basic_composition_rejects_over_budget_and_leaks_nothing():
    acc = BudgetAccountant(epsilon=1.0)
    acc.spend(0.8)
    assert acc.check(0.3) is False
    # 超预算查询必须抛异常
    with pytest.raises(BudgetExceededError):
        acc.spend(0.3)
    # 关键安全性质：拒绝后账户状态不变（未扣费），无法通过侧信道获知结果
    assert acc.used_epsilon == pytest.approx(0.8)
    assert acc.n_queries == 1
    assert acc.ledger()[0]["epsilon"] == 0.8


def test_exact_budget_boundary_allowed():
    acc = BudgetAccountant(epsilon=1.0)
    acc.spend(0.6)
    acc.spend(0.4)  # 恰好等于总预算，浮点容差内应允许
    assert acc.used_epsilon == pytest.approx(1.0)


def test_delta_accumulation_for_gaussian():
    acc = BudgetAccountant(epsilon=1.0, delta=1e-4)
    acc.spend(0.5, 1e-5)
    assert acc.used_delta == pytest.approx(1e-5)
    acc.spend(0.4, 5e-5)
    assert acc.used_delta == pytest.approx(6e-5)
    # δ 也会触发拒绝
    with pytest.raises(BudgetExceededError):
        acc.spend(0.05, 1e-3)


# ---------- 高级组合 -----------------------------------------------------
def test_advanced_composition_closed_form():
    # k 次 (ε0,0) 机制，总 ε = 2ε0 sqrt(2k ln(1/δ')) + 2kε0²
    eps0, k, delta_prime = 0.2, 5, 1e-5
    acc = BudgetAccountant(epsilon=10.0, delta=delta_prime,
                           composition="advanced")
    for _ in range(k):
        acc.spend(eps0, 0.0)
    expected = 2 * eps0 * math.sqrt(2 * k * math.log(1 / delta_prime)) \
        + 2 * k * eps0 ** 2
    assert acc.used_epsilon == pytest.approx(expected, rel=1e-9)


def test_advanced_rejects_epsilon_ge_one_per_query():
    acc = BudgetAccountant(epsilon=10.0, delta=1e-5, composition="advanced")
    with pytest.raises(InvalidParameterError):
        acc.spend(1.0)


def test_advanced_rejects_when_total_exceeded():
    # δ'=1e-5 时，单笔 ε0=0.2 的高级组合成本约 1.999，
    # 故总预算 2.5 允许 1 笔、第 2 笔（累计约 2.874）被拒绝
    acc = BudgetAccountant(epsilon=2.5, delta=1e-5, composition="advanced")
    acc.spend(0.2)
    with pytest.raises(BudgetExceededError):
        acc.spend(0.2)
    assert acc.n_queries == 1


def test_invalid_accountant_construction():
    with pytest.raises(InvalidParameterError):
        BudgetAccountant(epsilon=0)
    with pytest.raises(InvalidParameterError):
        BudgetAccountant(epsilon=1.0, delta=1.0)
    with pytest.raises(InvalidParameterError):
        BudgetAccountant(epsilon=1.0, composition="weird")


def test_check_does_not_mutate():
    acc = BudgetAccountant(epsilon=1.0)
    acc.check(0.9)
    acc.check(2.0)
    assert acc.n_queries == 0 and acc.used_epsilon == 0.0
