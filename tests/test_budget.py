"""预算账户：累计消耗、超预算拒绝且不泄露结果。"""

import pytest

from localdp import BudgetExhaustedError, LocalDPEngine, PrivacyAccountant


def test_budget_accumulates_across_queries():
    eng = LocalDPEngine(total_epsilon=2.0, seed=1)
    eng.count(["u1", "u2"], epsilon=0.5)
    eng.count(["u1", "u2"], epsilon=0.5)
    assert eng.accountant.spent_epsilon == pytest.approx(1.0)
    assert eng.accountant.remaining_epsilon == pytest.approx(1.0)


def test_budget_exhaustion_rejects_query():
    eng = LocalDPEngine(total_epsilon=1.0, seed=1)
    eng.count(["u1"], epsilon=1.0)
    with pytest.raises(BudgetExhaustedError):
        eng.count(["u1"], epsilon=0.1)


def test_rejected_query_does_not_spend_budget():
    eng = LocalDPEngine(total_epsilon=1.0, seed=1)
    eng.count(["u1"], epsilon=0.8)
    with pytest.raises(BudgetExhaustedError):
        eng.count(["u1"], epsilon=0.5)
    # 失败的查询不得扣减预算
    assert eng.accountant.spent_epsilon == pytest.approx(0.8)


def test_mean_query_budget_atomicity():
    # 预算只够一半拆分时，mean 必须整体拒绝且不产生半扣费
    eng = LocalDPEngine(total_epsilon=1.0, seed=1)
    eng.count(["u1"], epsilon=0.6)
    with pytest.raises(BudgetExhaustedError):
        eng.mean(["u1"], [5.0], lower=0.0, upper=10.0, epsilon=0.6)
    assert eng.accountant.spent_epsilon == pytest.approx(0.6)


def test_delta_budget_tracked_for_gaussian():
    eng = LocalDPEngine(total_epsilon=0.9, total_delta=2e-5, seed=1)
    eng.count(["u1"], epsilon=0.4, mechanism="gaussian", delta=1e-5)
    assert eng.accountant.spent_delta == pytest.approx(1e-5)
    with pytest.raises(BudgetExhaustedError):
        eng.count(["u1"], epsilon=0.4, mechanism="gaussian", delta=1.5e-5)


def test_accountant_standalone():
    acc = PrivacyAccountant(total_epsilon=1.0, total_delta=1e-4)
    acc.spend(0.3)
    acc.spend(0.3, 1e-5)
    assert acc.spent_epsilon == pytest.approx(0.6)
    assert acc.spent_delta == pytest.approx(1e-5)
    with pytest.raises(BudgetExhaustedError):
        acc.spend(0.5)
