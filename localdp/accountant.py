"""隐私预算账户：跟踪 (epsilon, delta) 消耗，超预算即拒绝。

组合规则采用基本组合（basic composition）：
  - 纯 DP（Laplace）：总 epsilon 线性累加，delta 保持 0；
  - 近似 DP（Gaussian）：epsilon 与 delta 分别线性累加。
"""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import BudgetExhaustedError, InvalidParameterError
from .mechanisms import check_delta, check_epsilon


@dataclass(frozen=True)
class Budget:
    """一次查询的预算消耗。"""

    epsilon: float
    delta: float = 0.0


class PrivacyAccountant:
    """本地隐私预算账户。

    total_epsilon / total_delta 为总预算；每次查询前先 try_spend，
    不足则抛出 BudgetExhaustedError，查询不得执行、不得泄露任何结果。
    """

    def __init__(self, total_epsilon: float, total_delta: float = 0.0):
        self.total_epsilon = check_epsilon(total_epsilon)
        if total_delta < 0.0 or total_delta >= 1.0:
            raise InvalidParameterError(
                f"总 delta 预算必须满足 0 <= delta < 1，收到 {total_delta!r}"
            )
        self.total_delta = float(total_delta)
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0

    @property
    def spent_epsilon(self) -> float:
        return self._spent_epsilon

    @property
    def spent_delta(self) -> float:
        return self._spent_delta

    @property
    def remaining_epsilon(self) -> float:
        return self.total_epsilon - self._spent_epsilon

    @property
    def remaining_delta(self) -> float:
        return self.total_delta - self._spent_delta

    def can_afford(self, epsilon: float, delta: float = 0.0) -> bool:
        eps = check_epsilon(epsilon)
        if delta < 0.0 or delta >= 1.0:
            raise InvalidParameterError(
                f"单次查询 delta 必须满足 0 <= delta < 1，收到 {delta!r}"
            )
        return (
            self._spent_epsilon + eps <= self.total_epsilon
            and self._spent_delta + delta <= self.total_delta
        )

    def spend(self, epsilon: float, delta: float = 0.0) -> Budget:
        """扣减预算；不足时抛出 BudgetExhaustedError 且不扣减任何预算。"""
        if delta > 0.0:
            check_delta(delta)
        if not self.can_afford(epsilon, delta):
            raise BudgetExhaustedError(
                "隐私预算不足，查询被拒绝："
                f"请求 (eps={epsilon}, delta={delta})，"
                f"剩余 (eps={self.remaining_epsilon}, delta={self.remaining_delta})"
            )
        self._spent_epsilon += float(epsilon)
        self._spent_delta += float(delta)
        return Budget(epsilon=float(epsilon), delta=float(delta))
