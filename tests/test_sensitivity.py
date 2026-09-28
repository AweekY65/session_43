"""敏感度公式的显式测试。"""

import numpy as np
import pytest

from ldp_stats import (
    count_sensitivity,
    histogram_sensitivity,
    mean_sensitivity,
    sum_sensitivity,
)
from ldp_stats.errors import InvalidParameterError


def test_count_sensitivity():
    assert count_sensitivity(1) == 1.0
    assert count_sensitivity(5) == 5.0


def test_sum_sensitivity_asymmetric_bounds():
    # [0, 10]：最大绝对单条影响是 10
    assert sum_sensitivity(0, 10, 1) == pytest.approx(10.0)
    # [-100, 10]：M = max(|L|, |U|) = 100
    assert sum_sensitivity(-100, 10, 1) == pytest.approx(100.0)
    # 每用户 3 条
    assert sum_sensitivity(-5, 5, 3) == pytest.approx(15.0)


def test_mean_sensitivity_formula():
    # (U-L)*C/n_pub = 10*2/50 = 0.4
    assert mean_sensitivity(0, 10, 2, 50) == pytest.approx(0.4)
    # 非零下界时敏感度只依赖宽度
    assert mean_sensitivity(40, 50, 2, 50) == pytest.approx(0.4)


def test_histogram_sensitivity_l1_l2():
    assert histogram_sensitivity(3) == (3.0, 3.0)


@pytest.mark.parametrize("kwargs", [
    dict(lower=1, upper=0, max_contributions=1),   # lower > upper
    dict(lower=0, upper=0, max_contributions=1),   # 退化区间
    dict(lower=float("-inf"), upper=1, max_contributions=1),
    dict(lower=0, upper=1, max_contributions=0),   # 非正贡献上限
    dict(lower=0, upper=1, max_contributions=-2),
])
def test_sensitivity_invalid_params(kwargs):
    with pytest.raises(InvalidParameterError):
        sum_sensitivity(**kwargs)
    with pytest.raises(InvalidParameterError):
        mean_sensitivity(denominator=10, **kwargs)


def test_mean_sensitivity_invalid_denominator():
    with pytest.raises(InvalidParameterError):
        mean_sensitivity(0, 10, 1, 0)
    with pytest.raises(InvalidParameterError):
        mean_sensitivity(0, 10, 1, -3.0)
    with pytest.raises(InvalidParameterError):
        mean_sensitivity(0, 10, 1, float("nan"))
