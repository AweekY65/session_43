"""端到端查询测试：count / sum / mean / histogram 的统计正确性、
预算耗尽行为、种子可重复性、空数据与非法参数处理、输出安全性。"""

import json

import numpy as np
import pytest

from ldp_stats import (
    MECHANISM_GAUSSIAN,
    MECHANISM_LAPLACE,
    PrivacyEngine,
)
from ldp_stats.errors import (
    BudgetExceededError,
    EmptyDataError,
    InvalidParameterError,
)

SEED = 20260926


def make_engine(**kwargs):
    defaults = dict(epsilon=5.0, lower=0.0, upper=100.0,
                    max_contributions=1, seed=SEED)
    defaults.update(kwargs)
    return PrivacyEngine(**defaults)


# ---------- 统计正确性（无偏性，大样本蒙特卡洛） -------------------------
def test_count_is_unbiased_laplace():
    # 每次迭代使用不同种子，得到独立噪声样本
    estimates = [
        make_engine(seed=SEED + i).count(np.arange(100, dtype=float), epsilon=2.0).value
        for i in range(2000)
    ]
    res = make_engine(seed=SEED).count(np.arange(100, dtype=float), epsilon=2.0)
    assert np.mean(estimates) == pytest.approx(100, abs=0.5)
    # 噪声尺度 = Δ/ε = 1/2
    assert res.noise_scale == pytest.approx(0.5)
    assert res.sensitivity == 1.0


def test_sum_is_unbiased_laplace():
    data = np.full(100, 50.0)
    estimates = [
        make_engine(seed=SEED + i).sum(data, epsilon=2.0).value
        for i in range(2000)
    ]
    assert np.mean(estimates) == pytest.approx(5000, abs=20)
    # Δ = C*max(|L|,|U|) = 100, 尺度 = 100/2 = 50
    res = make_engine(seed=SEED).sum(data, epsilon=2.0)
    assert res.sensitivity == 100.0
    assert res.noise_scale == pytest.approx(50.0)


def test_mean_is_unbiased_laplace():
    rng = np.random.default_rng(1)
    data = rng.uniform(0, 100, size=1000)
    true_mean = data.mean()
    estimates = [
        make_engine(seed=SEED + i).mean(data, epsilon=2.0).value
        for i in range(1500)
    ]
    assert np.mean(estimates) == pytest.approx(true_mean, abs=1.0)


def test_mean_with_public_denominator_unbiased():
    data = np.full(1000, 30.0)
    estimates = [
        make_engine(seed=SEED + i).mean(data, epsilon=2.0, denominator=1000).value
        for i in range(1000)
    ]
    assert np.mean(estimates) == pytest.approx(30.0, abs=1.0)


def test_histogram_bin_counts_laplace():
    data = np.array([5.0] * 10 + [25.0] * 20 + [95.0] * 30)
    estimates = np.array([
        make_engine(seed=SEED + i).histogram(data, bins=10, epsilon=3.0).value
        for i in range(800)
    ])
    res = make_engine(seed=SEED).histogram(data, bins=10, epsilon=3.0)
    means = estimates.mean(axis=0)
    assert means[0] == pytest.approx(10, abs=1.0)
    assert means[2] == pytest.approx(20, abs=1.0)
    assert means[9] == pytest.approx(30, abs=1.0)
    # 直方图 L1 敏感度 = C = 1，每维尺度 1/ε
    assert res.sensitivity == 1.0
    assert res.noise_scale == pytest.approx(1 / 3.0)
    assert res.parameters["n_bins"] == 10


def test_histogram_explicit_edges():
    eng = make_engine(lower=0.0, upper=10.0)
    res = eng.histogram([0.0, 9.9, 10.0], epsilon=1.0,
                        edges=[0.0, 5.0, 10.0])
    assert res.value.shape == (2,)
    assert res.parameters["bin_edges"] == [0.0, 5.0, 10.0]


def test_multi_record_contribution_sensitivity():
    # 每用户最多 3 条：count 敏感度应为 3
    eng = make_engine(max_contributions=3)
    res = eng.count([1.0, 2.0], user_ids=["a", "b"], epsilon=1.0)
    assert res.sensitivity == 3.0


# ---------- Gaussian 端到端 ---------------------------------------------
def test_gaussian_count_end_to_end():
    estimates = []
    last = None
    for i in range(1500):
        eng = make_engine(epsilon=2.0, delta=1e-5, seed=SEED + i)
        last = eng.count(np.full(100, 1.0), epsilon=0.5,
                         mechanism=MECHANISM_GAUSSIAN)
        estimates.append(last.value)
    assert np.mean(estimates) == pytest.approx(100, abs=0.5)
    assert last.delta_spent == pytest.approx(1e-5)
    expected_sigma = (1.0 / 0.5) * np.sqrt(2 * np.log(1.25 / 1e-5))
    assert last.noise_scale == pytest.approx(expected_sigma)
    assert last.parameters["gaussian_method"] == "classic"


def test_gaussian_analytic_end_to_end():
    eng = make_engine(epsilon=2.0, delta=1e-5, gaussian_method="analytic")
    res = eng.count([1.0, 2.0], epsilon=0.8, mechanism=MECHANISM_GAUSSIAN)
    assert np.isfinite(res.value)
    assert res.noise_scale > 0


def test_gaussian_requires_delta():
    eng = make_engine()  # 总 delta=0
    with pytest.raises(InvalidParameterError, match="delta"):
        eng.count([1.0], epsilon=0.5, mechanism=MECHANISM_GAUSSIAN)


# ---------- 预算耗尽：拒绝且不泄露 ---------------------------------------
def test_budget_exhaustion_rejects_and_produces_no_result():
    eng = make_engine(epsilon=1.0)
    eng.count([1.0, 2.0], epsilon=0.6)
    with pytest.raises(BudgetExceededError):
        eng.count([1.0, 2.0], epsilon=0.5)
    assert eng.accountant.used_epsilon == pytest.approx(0.6)
    assert eng.accountant.n_queries == 1


def test_budget_checked_before_data_access():
    # 非法数据 + 超预算同时存在时，应先报预算不足（预算检查不碰数据）
    eng = make_engine(epsilon=0.5)
    eng.count([1.0], epsilon=0.5)
    with pytest.raises(BudgetExceededError):
        eng.count([float("nan")], epsilon=0.3)


def test_mixed_laplace_gaussian_budget():
    eng = make_engine(epsilon=1.0, delta=1e-4)
    eng.count([1.0], epsilon=0.4)  # Laplace: δ 不增加
    eng.count([1.0], epsilon=0.3, delta=5e-5, mechanism=MECHANISM_GAUSSIAN)
    assert eng.accountant.used_delta == pytest.approx(5e-5)
    with pytest.raises(BudgetExceededError):
        eng.count([1.0], epsilon=0.5)


# ---------- 贡献裁剪在引擎中生效 -----------------------------------------
def test_engine_truncates_heavy_user():
    # 单用户 1000 条记录，cap=1：count 结果应围绕 1 而不是 1000
    data = np.full(1000, 100.0)
    users = np.array(["u"] * 1000)
    estimates = [
        make_engine(seed=SEED + i).count(data, user_ids=users, epsilon=2.0).value
        for i in range(1000)
    ]
    assert np.mean(estimates) == pytest.approx(1.0, abs=0.2)

    # sum 敏感度受 cap 限制，围绕 100（裁剪后的真实贡献）
    sum_est = [
        make_engine(seed=SEED + i).sum(data, user_ids=users, epsilon=2.0).value
        for i in range(1000)
    ]
    assert np.mean(sum_est) == pytest.approx(100.0, abs=8.0)


# ---------- 随机种子 -----------------------------------------------------
def test_seeded_results_are_reproducible():
    data = [10.0, 20.0, 30.0]
    r1 = make_engine(seed=42).count(data, epsilon=1.0)
    r2 = make_engine(seed=42).count(data, epsilon=1.0)
    r3 = make_engine(seed=43).count(data, epsilon=1.0)
    assert r1.value == r2.value
    assert r1.value != r3.value
    assert r1.test_mode is True and r1.seed == 42


def test_normal_mode_is_not_seeded():
    eng = make_engine(seed=None)
    assert eng.randomness.is_test is False
    r = eng.count([1.0, 2.0], epsilon=1.0)
    assert r.test_mode is False and r.seed is None


# ---------- 空数据 -------------------------------------------------------
def test_empty_data_raises_for_count_sum_histogram():
    eng = make_engine()
    with pytest.raises(EmptyDataError):
        eng.count([], epsilon=1.0)
    with pytest.raises(EmptyDataError):
        eng.sum([], epsilon=1.0)
    with pytest.raises(EmptyDataError):
        eng.histogram([], epsilon=1.0)
    # 空数据失败不扣预算
    assert eng.accountant.n_queries == 0


def test_mean_empty_needs_public_denominator():
    eng = make_engine()
    with pytest.raises(EmptyDataError):
        eng.mean([], epsilon=1.0)
    # 显式公开分母时空数据也可以发布（结果只反映下界 + 噪声）
    res = eng.mean([], epsilon=1.0, denominator=100)
    assert np.isfinite(res.value)
    assert eng.accountant.n_queries == 1


# ---------- 非法参数 -----------------------------------------------------
def test_invalid_engine_construction():
    with pytest.raises(InvalidParameterError):
        PrivacyEngine(epsilon=1.0, lower=10, upper=0, max_contributions=1)
    with pytest.raises(InvalidParameterError):
        PrivacyEngine(epsilon=-1.0, lower=0, upper=1)
    with pytest.raises(InvalidParameterError):
        PrivacyEngine(epsilon=1.0, lower=0, upper=1, max_contributions=0)
    with pytest.raises(InvalidParameterError):
        PrivacyEngine(epsilon=1.0, lower=0, upper=1, delta=1.0)


def test_invalid_query_params():
    eng = make_engine()
    with pytest.raises(InvalidParameterError):
        eng.count([1.0], epsilon=0)
    with pytest.raises(InvalidParameterError):
        eng.count([1.0], epsilon=-0.5)
    with pytest.raises(InvalidParameterError):
        eng.count([1.0], epsilon=1.0, mechanism="unknown")
    with pytest.raises(InvalidParameterError):
        eng.count([1.0], epsilon=0.5, delta=1e-5)  # Laplace 不接受 delta>0
    # Gaussian classic 且 ε >= 1 -> 明确报错
    eng2 = make_engine(epsilon=5.0, delta=1e-5)
    with pytest.raises(InvalidParameterError):
        eng2.count([1.0], epsilon=1.0, mechanism=MECHANISM_GAUSSIAN)


def test_invalid_histogram_edges():
    eng = make_engine()
    with pytest.raises(InvalidParameterError):
        eng.histogram([1.0], epsilon=1.0, edges=np.array([0.0, 10.0, 5.0]))
    with pytest.raises(InvalidParameterError):
        eng.histogram([1.0], epsilon=1.0, edges=np.array([10.0, 90.0]))
    with pytest.raises(InvalidParameterError):
        eng.histogram([1.0], epsilon=1.0, bins=0)


def test_non_finite_and_mismatched_records():
    eng = make_engine()
    with pytest.raises(InvalidParameterError):
        eng.count([float("nan")], epsilon=1.0)
    with pytest.raises(InvalidParameterError):
        eng.count([1.0, 2.0], user_ids=["a"], epsilon=1.0)


# ---------- 输出安全性 ---------------------------------------------------
SECRET = 909090909.0


def test_output_contains_required_metadata_no_raw_records():
    eng = make_engine()
    res = eng.count([1.0, 2.0, SECRET], epsilon=1.0)
    d = res.to_dict()
    # 必需字段
    assert d["mechanism"] == MECHANISM_LAPLACE
    assert "budget" in d and d["budget"]["epsilon_spent"] == 1.0
    assert d["sensitivity"] == 1.0
    assert "noise_scale" in d
    assert d["parameters"]["lower"] == 0.0
    # 可 JSON 序列化
    json.dumps(d)
    # 不得附带可还原单条原始记录的信息
    blob = json.dumps(d)
    assert str(int(SECRET)) not in blob
    assert "truncated_users" not in blob
    assert "raw" not in blob and "true_value" not in blob


def test_histogram_output_serializable():
    eng = make_engine()
    res = eng.histogram([1.0, 99.0], epsilon=1.0, bins=4)
    d = res.to_dict()
    assert isinstance(d["value"], list) and len(d["value"]) == 4
    json.dumps(d)


def test_rejection_message_carries_no_data():
    eng = make_engine(epsilon=0.1)
    try:
        eng.count([SECRET, SECRET], epsilon=0.5)
        pytest.fail("应当拒绝")
    except BudgetExceededError as exc:
        assert str(int(SECRET)) not in str(exc)
