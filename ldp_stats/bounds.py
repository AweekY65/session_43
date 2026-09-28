"""贡献裁剪（contribution bounding）与取值范围校验。

隐私模型
--------
邻接数据集定义为：添加/移除**一个用户**（add/remove 邻接关系），
一个用户最多贡献 ``max_contributions`` 条记录，每条记录的取值先裁剪到
``[lower, upper]``。这样每个用户对任何（L1 语义下的）统计量的影响都有
确定上界，敏感度不会因为单个用户贡献海量记录而无限增大。

裁剪规则是**确定性**的：
1. 先按用户分组，保持各用户记录的原始出现顺序；
2. 每个用户只保留最先出现的 ``max_contributions`` 条记录
   （“first-C wins”，无随机性、无歧义）；
3. 对保留下来的取值做范围裁剪 ``clip(x, lower, upper)``。

非确定的“随机抽样 C 条”会让结果依赖额外随机状态，而这里的裁剪本身
不消耗隐私预算，因此采用确定性规则。
"""

from dataclasses import dataclass

import numpy as np

from .errors import InvalidParameterError


def validate_bounds(lower, upper):
    """校验数值上下界，返回 ``(float(lower), float(upper))``。

    要求 lower、upper 均为有限实数，且 ``lower < upper``（退化区间会使
    敏感度为 0 且没有统计意义，明确拒绝）。
    """
    try:
        lo = float(lower)
        hi = float(upper)
    except (TypeError, ValueError):
        raise InvalidParameterError(
            "lower/upper 必须是实数，收到 %r, %r" % (lower, upper)
        )
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise InvalidParameterError(
            "lower/upper 必须是有限值，收到 %s, %s" % (lo, hi)
        )
    if lo >= hi:
        raise InvalidParameterError(
            "要求 lower < upper，收到 lower=%s, upper=%s" % (lo, hi)
        )
    return lo, hi


@dataclass(frozen=True)
class BoundedData:
    """裁剪后的数据。

    Attributes
    ----------
    values : np.ndarray
        一维 float64 数组，已经过“每用户最多 C 条 + 取值裁剪”。
    n_records : int
        裁剪后的记录总数（注意：这是数据相关量，默认不放进对外输出）。
    n_users : int
        参与贡献的用户数（同样是数据相关量，仅供本地调用方参考）。
    truncated_users : dict[str, int]
        被截断的用户 -> 被丢弃的记录数。**仅供本地调试**，序列化对外结果时
        不得包含（它可能间接反映单个用户的贡献量）。
    """

    values: np.ndarray
    n_records: int
    n_users: int
    truncated_users: dict

    @property
    def is_empty(self):
        return self.n_records == 0


def bound_records(values, lower, upper, max_contributions, user_ids=None):
    """执行确定性贡献裁剪，返回 :class:`BoundedData`。

    Parameters
    ----------
    values : array-like
        一维数值记录。
    lower, upper : float
        取值裁剪范围（要求 ``lower < upper``，均有限）。
    max_contributions : int
        每个用户最多保留的记录数，必须为正整数。
    user_ids : array-like, optional
        与 ``values`` 等长的用户标识；为 ``None`` 时视为每条记录属于
        不同用户（此时不存在“同一用户超额贡献”，贡献上限自然满足）。
    """
    lo, hi = validate_bounds(lower, upper)

    try:
        cap = int(max_contributions)
    except (TypeError, ValueError):
        raise InvalidParameterError(
            "max_contributions 必须是正整数，收到 %r" % (max_contributions,)
        )
    if cap <= 0 or cap != max_contributions:
        raise InvalidParameterError(
            "max_contributions 必须是正整数，收到 %r" % (max_contributions,)
        )

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise InvalidParameterError("values 必须是一维数组")
    if not np.all(np.isfinite(arr)):
        raise InvalidParameterError("输入记录必须全部是有限数值（不接受 NaN/Inf）")

    if user_ids is None:
        # 每条记录一个独立用户：最多保留 cap 条/用户不会截断任何记录，
        # 但仍按规则做取值裁剪，保证后续敏感度推导一致。
        kept = arr
        n_users = arr.size
        truncated = {}
    else:
        ids = np.asarray(user_ids)
        if ids.ndim != 1 or ids.shape[0] != arr.shape[0]:
            raise InvalidParameterError(
                "user_ids 必须与 values 等长的一维数组"
            )
        keep_mask = np.zeros(arr.shape[0], dtype=bool)
        # 已见用户 -> 已保留条数（向量化分组，保持出现顺序）
        seen = {}
        truncated = {}
        for i, uid in enumerate(ids.tolist()):
            count = seen.get(uid, 0)
            if count < cap:
                keep_mask[i] = True
                seen[uid] = count + 1
            else:
                # 该用户的第 cap+1 条及以后的记录逐条计数
                truncated[uid] = truncated.get(uid, 0) + 1
        kept = arr[keep_mask]
        n_users = len(seen)

    clipped = np.clip(kept, lo, hi)
    return BoundedData(
        values=clipped,
        n_records=int(clipped.size),
        n_users=int(n_users),
        truncated_users=truncated,
    )
