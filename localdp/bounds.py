"""Contribution bounding：限制单个用户对数据集的贡献条数。

邻接定义（add/remove，无界 DP）：两个数据集相邻，当且仅当其中一个
可通过增删**单个用户**的全部贡献得到另一个。若不限制单用户贡献条数，
敏感度将无界，因此必须按确定规则裁剪。

裁剪规则（确定性）：保持输入原始顺序，对每个 user_id 仅保留其前
max_contributions 条记录，其余丢弃。同一输入永远得到同一裁剪结果。
"""

from __future__ import annotations

import numpy as np

from .exceptions import InvalidParameterError


def check_max_contributions(max_contributions: int) -> int:
    if isinstance(max_contributions, bool) or not isinstance(
        max_contributions, (int, np.integer)
    ):
        raise InvalidParameterError(
            f"max_contributions 必须为正整数，收到 {max_contributions!r}"
        )
    if max_contributions < 1:
        raise InvalidParameterError(
            f"max_contributions 必须 >= 1，收到 {max_contributions!r}"
        )
    return int(max_contributions)


def bound_contributions(user_ids, max_contributions: int) -> np.ndarray:
    """返回保留记录的索引（按原始顺序）。

    确定性规则：按输入顺序扫描，每个用户最多保留前 max_contributions 条。
    """
    k = check_max_contributions(max_contributions)
    user_ids = np.asarray(user_ids)
    counts: dict = {}
    keep: list[int] = []
    for idx, uid in enumerate(user_ids.tolist()):
        c = counts.get(uid, 0)
        if c < k:
            counts[uid] = c + 1
            keep.append(idx)
    return np.array(keep, dtype=int)
