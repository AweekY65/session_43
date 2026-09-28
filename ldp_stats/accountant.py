"""隐私预算账户（budget accountant）。

组合方法（两种，均为严格、可证明的组合定理，不做近似乐观估计）：

- ``"basic"``：基本组合（Basic Composition）。k 次 (ε_i, δ_i)-DP 机制
  串行组合后满足 (Σε_i, Σδ_i)-DP。Laplace 查询视为 (ε, 0)。
- ``"advanced"``：高级组合（Advanced Composition，Dwork, Rothblum &
  Vadhan 2010；Dwork & Roth 2014 定理 3.20）。对 k 次 (ε_0, δ_0)-DP
  机制（要求 0 < ε_0 < 1，δ_0 ≥ 0），在整体 δ' 预算下总 ε 为
      ε_total = 2 ε_0 sqrt(2 k ln(1/δ')) + 2 k ε_0²
  总 δ = δ' + k δ_0。
  本账户采用**逐次累积**的保守实现：每次查询后用高级组合定理重算总开销，
  即 k 次 ε_0 相同的查询恰好回到定理闭式值；混批不同 ε_0 时按最大
  ε_0 重算，仍是合法上界（略保守）。

关键安全性质：
- ``check`` 只做预演，不改变账户状态；
- 只有在账户确认预算足够后查询才会执行，执行后才 ``spend``；
- 超预算查询在**结果计算与噪声生成之前**被拒绝（见 engine），
  因此拒绝路径不产生任何输出，不会泄露结果；
- δ 只对 Gaussian 查询累计；纯 Laplace 查询 δ=0。
"""

import math

import numpy as np

from .errors import InvalidParameterError


class BudgetAccountant:
    def __init__(self, epsilon, delta=0.0, composition="basic"):
        try:
            total_eps = float(epsilon)
            total_delta = float(delta)
        except (TypeError, ValueError):
            raise InvalidParameterError("epsilon/delta 必须是数值")
        if not np.isfinite(total_eps) or total_eps <= 0:
            raise InvalidParameterError("总 epsilon 必须是正的有限值，收到 %s" % total_eps)
        if not np.isfinite(total_delta) or total_delta < 0 or total_delta >= 1:
            raise InvalidParameterError("总 delta 必须满足 0 <= delta < 1，收到 %s" % total_delta)
        if composition not in ("basic", "advanced"):
            raise InvalidParameterError(
                "composition 必须是 'basic' 或 'advanced'，收到 %r" % composition
            )
        self.total_epsilon = total_eps
        self.total_delta = total_delta
        self.composition = composition
        # 已发生的逐笔开销（基础组合用）
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0
        # 高级组合用：查询笔数 / 单笔最大 ε0 / 单笔最大 δ0
        self._n_advanced = 0
        self._max_eps0 = 0.0
        self._sum_delta0 = 0.0
        self._ledger = []
        # 浮点容差：避免 0.1+0.2 这类舍入误差把合法查询误判为超预算
        self._eps_tol = 1e-12
        self._delta_tol = 1e-15

    # ---- 状态读取 ------------------------------------------------------
    @property
    def used_epsilon(self):
        return self._combined()[0]

    @property
    def used_delta(self):
        return self._combined()[1]

    @property
    def remaining_epsilon(self):
        return self.total_epsilon - self.used_epsilon

    @property
    def remaining_delta(self):
        return self.total_delta - self.used_delta

    @property
    def n_queries(self):
        return len(self._ledger)

    @property
    def is_exhausted(self):
        used_eps, used_delta = self._combined()
        return used_eps > self.total_epsilon + self._eps_tol or \
            used_delta > self.total_delta + self._delta_tol

    def ledger(self):
        """逐笔预算开销的只读副本（只含公共机制参数，不含数据）。"""
        return [dict(item) for item in self._ledger]

    # ---- 预算计算 ------------------------------------------------------
    def _combined(self, extra_eps=0.0, extra_delta=0.0):
        """返回（在已有开销上再追加一笔后的）总 (ε, δ) 开销。"""
        if self.composition == "basic":
            return self._spent_epsilon + extra_eps, self._spent_delta + extra_delta

        # advanced composition
        n = self._n_advanced + (1 if extra_eps > 0.0 else 0)
        if n == 0:
            return 0.0, 0.0
        eps0 = max(self._max_eps0, extra_eps)
        delta0_sum = self._sum_delta0 + extra_delta
        # 高级组合需要一个整体 δ' 预算：账户总 δ 与“机制 δ 之和”之差。
        # δ_total_adv = δ' + Σδ0，因此取 δ' = max(总δ − Σδ0, 0) 是合法选择
        # （δ' 必须 >0 才有 ln(1/δ') 定义；极小时用下浮点数下限保护）。
        delta_prime = max(self.total_delta - delta0_sum, 1e-300)
        if not (0.0 < eps0 < 1.0):
            raise InvalidParameterError(
                "高级组合要求每笔查询满足 0 < epsilon < 1（收到最大单笔 "
                "epsilon=%s）；请减小单笔 epsilon 或改用 basic 组合" % eps0
            )
        eps_total = (
            2.0 * eps0 * math.sqrt(2.0 * n * math.log(1.0 / delta_prime))
            + 2.0 * n * eps0 ** 2
        )
        return eps_total, delta_prime + delta0_sum

    def check(self, epsilon, delta=0.0):
        """预演：若现在再花 (epsilon, delta) 是否在总预算内。不改变状态。"""
        try:
            eps = float(epsilon)
            d = float(delta)
        except (TypeError, ValueError):
            raise InvalidParameterError("epsilon/delta 必须是数值")
        if not np.isfinite(eps) or eps <= 0:
            raise InvalidParameterError("查询 epsilon 必须为正，收到 %s" % eps)
        if not np.isfinite(d) or d < 0 or d >= 1:
            raise InvalidParameterError("查询 delta 必须满足 0 <= delta < 1，收到 %s" % d)
        if self.composition == "advanced" and not (0.0 < eps < 1.0):
            raise InvalidParameterError(
                "高级组合要求单笔 0 < epsilon < 1，收到 %s" % eps
            )
        used_eps, used_delta = self._combined(eps, d)
        if used_eps > self.total_epsilon + self._eps_tol:
            return False
        if used_delta > self.total_delta + self._delta_tol:
            return False
        return True

    def spend(self, epsilon, delta=0.0, label=""):
        """实际扣除一笔预算；不足时抛 :class:`BudgetExceededError`，状态不变。"""
        if not self.check(epsilon, delta):
            used_eps, used_delta = self._combined(float(epsilon), float(delta))
            from .errors import BudgetExceededError
            raise BudgetExceededError(
                "查询被拒绝：预算不足。该笔需要 (eps=%s, delta=%s)，"
                "执行后累计将为 (eps=%.6g, delta=%.6g)，"
                "总预算为 (eps=%s, delta=%s)。未扣除预算，且未生成结果。"
                % (epsilon, delta, used_eps, used_delta,
                   self.total_epsilon, self.total_delta)
            )
        self._spent_epsilon += float(epsilon)
        self._spent_delta += float(delta)
        self._n_advanced += 1
        self._max_eps0 = max(self._max_eps0, float(epsilon))
        self._sum_delta0 += float(delta)
        self._ledger.append({
            "label": str(label),
            "epsilon": float(epsilon),
            "delta": float(delta),
            "composition": self.composition,
        })
