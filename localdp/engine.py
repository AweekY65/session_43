"""本地差分隐私查询引擎。

邻接关系：add/remove（无界 DP）——相邻数据集相差**一个用户**的全部贡献。
配合 contribution bounding（每用户最多 max_contributions 条），各查询敏感度：

  - count:     Δ = max_contributions
  - sum:       Δ = max_contributions * max(|lower|, |upper|)（值先裁剪到 [lower, upper]）
  - mean:      预算对半拆分，分别计算 noisy sum 与 noisy count 后相除
  - histogram: L1 敏感度 Δ = max_contributions（每条记录恰好落入一个桶）

输出 QueryResult 仅包含机制、预算消耗、敏感度与噪声参数等元信息，
不包含任何可还原单条原始记录的数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .accountant import PrivacyAccountant
from .bounds import bound_contributions, check_max_contributions
from .exceptions import DPError, InvalidParameterError
from .mechanisms import (
    add_gaussian_noise,
    add_laplace_noise,
    check_bounds,
    check_delta,
    check_epsilon,
)

LAPLACE = "laplace"
GAUSSIAN = "gaussian"
_MECHANISMS = (LAPLACE, GAUSSIAN)


@dataclass(frozen=True)
class QueryResult:
    """一次差分隐私查询的安全输出（不含任何原始记录）。"""

    value: Any
    mechanism: str
    epsilon: float
    delta: float
    sensitivity: float
    noise_parameter: float  # Laplace 的 scale b 或 Gaussian 的 sigma
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        value = self.value
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return {
            "value": value,
            "mechanism": self.mechanism,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sensitivity": self.sensitivity,
            "noise_parameter": self.noise_parameter,
            "params": dict(self.params),
        }


class LocalDPEngine:
    """本地差分隐私查询引擎。

    参数
    ----
    total_epsilon / total_delta : 总隐私预算（基本组合线性累计）。
    max_contributions           : 单用户最大贡献条数（contribution bounding）。
    seed                        : 仅测试模式使用，注入固定种子以获得可重复结果；
                                  正常模式必须传 None（默认），随机性来自操作系统熵源。
    """

    def __init__(
        self,
        total_epsilon: float,
        total_delta: float = 0.0,
        max_contributions: int = 1,
        seed: Optional[int] = None,
    ):
        self.accountant = PrivacyAccountant(total_epsilon, total_delta)
        self.max_contributions = check_max_contributions(max_contributions)
        # seed=None 时 default_rng 从 OS 熵源取种子，不会退化为固定噪声
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _check_mechanism(self, mechanism: str) -> str:
        if mechanism not in _MECHANISMS:
            raise InvalidParameterError(
                f"未知机制 {mechanism!r}，可选：{_MECHANISMS}"
            )
        return mechanism

    def _spend_and_noise(
        self, true_value, sensitivity: float, epsilon: float, delta: float, mechanism: str
    ):
        """先扣预算（不足则抛异常、不计算不泄露），再加噪。"""
        budget = self.accountant.spend(epsilon, delta)
        if mechanism == LAPLACE:
            noisy, noise_param = add_laplace_noise(
                true_value, sensitivity, budget.epsilon, self._rng
            )
        else:
            noisy, noise_param = add_gaussian_noise(
                true_value, sensitivity, budget.epsilon, budget.delta, self._rng
            )
        return noisy, noise_param, budget

    def _bound(self, user_ids, values=None):
        """校验输入长度并执行 contribution bounding，返回裁剪后的 (user_ids, values)。"""
        user_ids = np.asarray(user_ids)
        if values is not None:
            values = np.asarray(values, dtype=float)
            if len(user_ids) != len(values):
                raise InvalidParameterError(
                    f"user_ids 与 values 长度不一致：{len(user_ids)} vs {len(values)}"
                )
        if user_ids.size == 0:
            return user_ids, values
        keep = bound_contributions(user_ids, self.max_contributions)
        user_ids = user_ids[keep]
        if values is not None:
            values = values[keep]
        return user_ids, values

    def _resolve_delta(self, mechanism: str, delta: Optional[float]) -> float:
        if mechanism == GAUSSIAN:
            if delta is None:
                raise InvalidParameterError("Gaussian 机制必须显式提供 delta")
            return check_delta(delta)
        if delta is not None:
            raise InvalidParameterError("Laplace 机制为纯 DP，不接受 delta 参数")
        return 0.0

    # ------------------------------------------------------------------ #
    # 查询接口
    # ------------------------------------------------------------------ #
    def count(
        self,
        user_ids: Sequence,
        epsilon: float,
        mechanism: str = LAPLACE,
        delta: Optional[float] = None,
    ) -> QueryResult:
        """带噪计数查询。敏感度 Δ = max_contributions。"""
        mechanism = self._check_mechanism(mechanism)
        eps = check_epsilon(epsilon)
        d = self._resolve_delta(mechanism, delta)
        user_ids, _ = self._bound(user_ids)
        sensitivity = float(self.max_contributions)
        noisy, noise_param, budget = self._spend_and_noise(
            float(user_ids.size), sensitivity, eps, d, mechanism
        )
        return QueryResult(
            value=float(noisy),
            mechanism=mechanism,
            epsilon=budget.epsilon,
            delta=budget.delta,
            sensitivity=sensitivity,
            noise_parameter=noise_param,
            params={"query": "count", "max_contributions": self.max_contributions},
        )

    def sum(
        self,
        user_ids: Sequence,
        values: Sequence[float],
        lower: float,
        upper: float,
        epsilon: float,
        mechanism: str = LAPLACE,
        delta: Optional[float] = None,
    ) -> QueryResult:
        """带噪求和查询，值先裁剪到 [lower, upper]。

        敏感度 Δ = max_contributions * max(|lower|, |upper|)。
        """
        mechanism = self._check_mechanism(mechanism)
        eps = check_epsilon(epsilon)
        d = self._resolve_delta(mechanism, delta)
        lo, hi = check_bounds(lower, upper)
        _, clipped_values = self._bound(user_ids, values)
        if clipped_values.size == 0:
            true_sum = 0.0
        else:
            true_sum = float(np.clip(clipped_values, lo, hi).sum())
        sensitivity = self.max_contributions * max(abs(lo), abs(hi))
        noisy, noise_param, budget = self._spend_and_noise(
            true_sum, sensitivity, eps, d, mechanism
        )
        return QueryResult(
            value=float(noisy),
            mechanism=mechanism,
            epsilon=budget.epsilon,
            delta=budget.delta,
            sensitivity=sensitivity,
            noise_parameter=noise_param,
            params={
                "query": "sum",
                "lower": lo,
                "upper": hi,
                "max_contributions": self.max_contributions,
            },
        )

    def mean(
        self,
        user_ids: Sequence,
        values: Sequence[float],
        lower: float,
        upper: float,
        epsilon: float,
        mechanism: str = LAPLACE,
        delta: Optional[float] = None,
    ) -> QueryResult:
        """带噪均值查询：预算对半拆分，noisy_sum / noisy_count。

        空数据（裁剪后无任何记录）时抛出 DPError，避免除零泄露。
        """
        mechanism = self._check_mechanism(mechanism)
        eps = check_epsilon(epsilon)
        d = self._resolve_delta(mechanism, delta)
        lo, hi = check_bounds(lower, upper)
        user_ids_bounded, clipped_values = self._bound(user_ids, values)
        if user_ids_bounded.size == 0:
            raise DPError("mean 查询在空数据集上无定义（裁剪后无记录）")

        # 先整体校验预算，避免 sum 扣费成功、count 扣费失败造成的半扣费状态
        if not self.accountant.can_afford(eps, d):
            from .exceptions import BudgetExhaustedError

            raise BudgetExhaustedError(
                "隐私预算不足，mean 查询被拒绝："
                f"请求 (eps={eps}, delta={d})，"
                f"剩余 (eps={self.accountant.remaining_epsilon}, "
                f"delta={self.accountant.remaining_delta})"
            )

        half_eps = eps / 2.0
        half_delta = d / 2.0 if d > 0.0 else 0.0

        sum_sens = self.max_contributions * max(abs(lo), abs(hi))
        count_sens = float(self.max_contributions)

        true_sum = float(np.clip(clipped_values, lo, hi).sum())
        true_count = float(user_ids_bounded.size)

        # 两次扣费任一失败都会抛 BudgetExhaustedError，结果不会返回
        noisy_sum, sum_noise_param, sum_budget = self._spend_and_noise(
            true_sum, sum_sens, half_eps, half_delta, mechanism
        )
        noisy_count, count_noise_param, count_budget = self._spend_and_noise(
            true_count, count_sens, half_eps, half_delta, mechanism
        )

        noisy_count_val = float(noisy_count)
        if noisy_count_val <= 0.0:
            # 加噪后计数非正，均值无意义；返回 None 而非泄露真实计数
            mean_value = None
        else:
            mean_value = float(noisy_sum) / noisy_count_val

        return QueryResult(
            value=mean_value,
            mechanism=mechanism,
            epsilon=sum_budget.epsilon + count_budget.epsilon,
            delta=sum_budget.delta + count_budget.delta,
            sensitivity=sum_sens,  # 主导项：sum 的敏感度
            noise_parameter=max(sum_noise_param, count_noise_param),
            params={
                "query": "mean",
                "lower": lo,
                "upper": hi,
                "max_contributions": self.max_contributions,
                "split": "epsilon/2 for sum, epsilon/2 for count",
                "noisy_count": noisy_count_val,
            },
        )

    def histogram(
        self,
        user_ids: Sequence,
        categories: Sequence,
        bins: Sequence,
        epsilon: float,
        mechanism: str = LAPLACE,
        delta: Optional[float] = None,
    ) -> QueryResult:
        """带噪直方图查询（类别型）。每个桶独立加噪，L1 敏感度 Δ = max_contributions。

        bins 为合法的类别标签列表；不在 bins 中的记录被忽略。
        """
        mechanism = self._check_mechanism(mechanism)
        eps = check_epsilon(epsilon)
        d = self._resolve_delta(mechanism, delta)
        bins = list(bins)
        if not bins:
            raise InvalidParameterError("bins 不能为空")
        if len(set(bins)) != len(bins):
            raise InvalidParameterError("bins 中存在重复标签")

        user_ids = np.asarray(user_ids)
        categories = np.asarray(categories)
        if len(user_ids) != len(categories):
            raise InvalidParameterError(
                f"user_ids 与 categories 长度不一致：{len(user_ids)} vs {len(categories)}"
            )
        keep = bound_contributions(user_ids, self.max_contributions)
        categories = categories[keep]

        index = {b: i for i, b in enumerate(bins)}
        counts = np.zeros(len(bins), dtype=float)
        for c in categories.tolist():
            i = index.get(c)
            if i is not None:
                counts[i] += 1.0

        sensitivity = float(self.max_contributions)
        noisy, noise_param, budget = self._spend_and_noise(
            counts, sensitivity, eps, d, mechanism
        )
        return QueryResult(
            value=np.asarray(noisy, dtype=float),
            mechanism=mechanism,
            epsilon=budget.epsilon,
            delta=budget.delta,
            sensitivity=sensitivity,
            noise_parameter=noise_param,
            params={
                "query": "histogram",
                "bins": bins,
                "max_contributions": self.max_contributions,
            },
        )
