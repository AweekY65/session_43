"""贡献裁剪（contribution bounding）与取值范围裁剪测试。"""

import numpy as np
import pytest

from ldp_stats import bound_records
from ldp_stats.errors import InvalidParameterError


def test_first_c_records_kept_deterministically():
    # 用户 u1 有 5 条记录，上限 2 -> 保留最先出现的两条 [1, 2]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    users = ["u1", "u1", "u1", "u1", "u1", "u2"]
    b = bound_records(values, 0, 10, max_contributions=2, user_ids=users)
    # u1 保留 1,2；u2 保留 10；u1 丢弃 3 条
    assert b.n_records == 3
    assert b.truncated_users == {"u1": 3}
    np.testing.assert_array_equal(b.values, [1.0, 2.0, 10.0])
    assert b.n_users == 2


def test_clipping_is_deterministic_and_order_stable():
    b1 = bound_records([-5.0, 3.0, 50.0], 0, 10, 1)
    b2 = bound_records([-5.0, 3.0, 50.0], 0, 10, 1)
    np.testing.assert_array_equal(b1.values, b2.values)
    np.testing.assert_array_equal(b1.values, [0.0, 3.0, 10.0])


def test_cap_then_clip_order():
    # 先截断（保留前 2 条），再裁剪范围；被截断的超界记录不会“替换”进来
    b = bound_records([100.0, 2.0, 3.0], 0, 10, max_contributions=2,
                      user_ids=["a", "a", "a"])
    np.testing.assert_array_equal(b.values, [10.0, 2.0])


def test_no_user_ids_means_each_record_is_one_user():
    b = bound_records([1.0, 2.0, 3.0], 0, 10, max_contributions=1)
    assert b.n_records == 3 and b.n_users == 3
    assert b.truncated_users == {}


def test_bounded_data_is_empty():
    assert bound_records([], 0, 10, 1).is_empty


def test_sensitivity_bounded_by_design():
    # 极端用户：10000 条全为 100 的记录，上限 1，范围 [0,10]
    vals = [100.0] * 10000
    users = ["heavy"] * 10000
    b = bound_records(vals, 0, 10, max_contributions=1, user_ids=users)
    assert b.n_records == 1
    np.testing.assert_array_equal(b.values, [10.0])


@pytest.mark.parametrize("values,users,msg_match", [
    ([[1.0, 2.0]], None, "一维"),                       # 二维输入
    ([1.0, float("nan")], None, "有限数值"),            # NaN
    ([1.0, float("inf")], None, "有限数值"),            # Inf
    ([1.0, 2.0], ["a"], "等长"),                        # id 数量不匹配
])
def test_bounding_invalid_inputs(values, users, msg_match):
    with pytest.raises(InvalidParameterError, match=msg_match):
        bound_records(values, 0, 10, 1, user_ids=users)


def test_bounding_invalid_cap_and_bounds():
    with pytest.raises(InvalidParameterError):
        bound_records([1.0], 0, 10, 0)
    with pytest.raises(InvalidParameterError):
        bound_records([1.0], 0, 10, 1.5)
    with pytest.raises(InvalidParameterError):
        bound_records([1.0], 5, 1, 1)
