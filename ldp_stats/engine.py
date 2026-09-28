"""隐私统计引擎：count / sum / mean / histogram。

安全执行顺序（每种查询一致）：
  1. 校验机制与参数（纯公共参数，不碰数据）；
  2. 向预算账户**预查**预算是否足够（``check``，不改状态）；
  3. 执行确定性贡献裁剪；
  4. 计算真实统计量并加噪；
  5. ``spend`` 实际扣除预算，返回只含安全字段的 :class:`QueryResult`。

第 2 步在数据处理之前，因此“预算不足”的拒绝不依赖任何数据；拒绝路径
既不计算结果、不生成噪声，也不扣除预算，不泄露任何查询结果。

对外输出（:class:`QueryResult`）只包含：机制名称、预算消耗、敏感度、
噪声尺度、公共查询参数、测试模式标记（仅测试模式下附带种子），**绝不**
包含原始记录、按用户的贡献明细或真实（未加噪）统计量。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .accountant import BudgetAccountant
from .bounds import bound_records, validate_bounds
from .errors import EmptyDataError, InvalidParameterError
from .mechanisms import (
    MECHANISM_GAUSSIAN,
    MECHANISM_LAPLACE,
    RandomnessProvider,
    gaussian_mechanism,
    laplace_mechanism,
)
from .sensitivity import (
    count_sensitivity,
    histogram_sensitivity,
    mean_sensitivity,
    sum_sensitivity,
)

_MECHANISMS = (MECHANISM_LAPLACE, MECHANISM_GAUSSIAN)


@dataclass
class QueryResult:
    """一次差分隐私查询的输出（全部字段可安全对外展示/序列化）。"""

    query_type: str
    mechanism: str
    value: object                      # float（数值查询）或 np.ndarray（histogram）
    epsilon_spent: float
    delta_spent: float
    sensitivity: float                 # 该机制实际使用的敏感度（输出量纲）
    noise_scale: float                 # Laplace 尺度 b 或 Gaussian 标准差 σ
    parameters: dict = field(default_factory=dict)  # 公共参数（见各查询）
    test_mode: bool = False
    seed: Optional[int] = None         # 仅 test_mode=True 时存在；正常模式恒为 None

    def to_dict(self):
        """转成可 JSON 序列化的纯字典（数组转 list）。

        不包含任何输入数据、按用户的贡献信息或未加噪真值。
        """
        v = self.value
        if isinstance(v, np.ndarray):
            v = v.tolist()
        out = {
            "query_type": self.query_type,
            "mechanism": self.mechanism,
            "value": v,
            "budget": {
                "epsilon_spent": self.epsilon_spent,
                "delta_spent": self.delta_spent,
            },
            "sensitivity": self.sensitivity,
            "noise_scale": self.noise_scale,
            "parameters": dict(self.parameters),
            "test_mode": self.test_mode,
        }
        if self.test_mode:
            out["seed"] = self.seed
        return out


class PrivacyEngine:
    """本地差分隐私统计引擎。

    Parameters
    ----------
    epsilon, delta : float
        总隐私预算。``delta=0``（默认）时只能使用 Laplace 机制。
    lower, upper : float
        所有记录取值的公开裁剪范围，要求 ``lower < upper`` 且均有限。
    max_contributions : int
        每个用户最多保留的记录数（contribution bound，正整数）。
    composition : {"basic", "advanced"}
        预算组合方法，见 :class:`~ldp_stats.accountant.BudgetAccountant`。
    seed : int or None
        None（默认，正常模式）：使用系统熵的非固定随机噪声；
        整数：测试模式，噪声可重复。正常模式绝不使用固定种子。
    gaussian_method : {"classic", "analytic"}
        Gaussian 机制 σ 的标定方法。
    """

    def __init__(self, epsilon, lower, upper, max_contributions=1,
                 delta=0.0, composition="basic", seed=None,
                 gaussian_method="classic"):
        self.lower, self.upper = validate_bounds(lower, upper)
        if not isinstance(max_contributions, (int, np.integer)) or max_contributions <= 0:
            raise InvalidParameterError(
                "max_contributions 必须是正整数，收到 %r" % (max_contributions,)
            )
        self.max_contributions = int(max_contributions)
        self.accountant = BudgetAccountant(epsilon, delta, composition)
        self.randomness = RandomnessProvider(seed)
        if gaussian_method not in ("classic", "analytic"):
            raise InvalidParameterError(
                "gaussian_method 必须是 'classic' 或 'analytic'"
            )
        self.gaussian_method = gaussian_method

    # ---- 公共查询 ------------------------------------------------------
    def count(self, values, user_ids=None, epsilon=None, delta=None,
              mechanism=MECHANISM_LAPLACE):
        """带贡献裁剪的差分隐私 count（记录条数）。"""
        eps, d = self._prepare_query(mechanism, epsilon, delta)
        bounded = self._bound(values, user_ids)
        self._require_nonempty(bounded, "count")
        sensitivity = count_sensitivity(self.max_contributions)
        true_value = float(bounded.n_records)
        return self._execute(
            "count", mechanism, true_value, sensitivity, eps, d,
            extra_params={},
        )

    def sum(self, values, user_ids=None, epsilon=None, delta=None,
            mechanism=MECHANISM_LAPLACE):
        """带上下界与贡献裁剪的差分隐私 sum。"""
        eps, d = self._prepare_query(mechanism, epsilon, delta)
        bounded = self._bound(values, user_ids)
        self._require_nonempty(bounded, "sum")
        sensitivity = sum_sensitivity(self.lower, self.upper,
                                      self.max_contributions)
        true_value = float(np.sum(bounded.values))
        return self._execute(
            "sum", mechanism, true_value, sensitivity, eps, d,
            extra_params={},
        )

    def mean(self, values, user_ids=None, epsilon=None, delta=None,
             mechanism=MECHANISM_LAPLACE, denominator=None):
        """差分隐私 mean。

        denominator 为**公开**的固定分母 n_pub（不随数据变化）。缺省时
        使用贡献裁剪后的记录数 —— 注意这隐含“数据规模是公开信息”的
        假设；若不成立，请显式传入公开分母（例如已知采样规模）。
        显式给出 denominator 时允许空数据（此时发布值就是下界，上面
        只有噪声，不反映任何记录）。
        """
        eps, d = self._prepare_query(mechanism, epsilon, delta)
        bounded = self._bound(values, user_ids)
        if denominator is None:
            self._require_nonempty(bounded, "mean")
            n_pub = float(bounded.n_records)
            denom_public = False
        else:
            try:
                n_pub = float(denominator)
            except (TypeError, ValueError):
                raise InvalidParameterError("denominator 必须是正数，收到 %r" % (denominator,))
            if not np.isfinite(n_pub) or n_pub <= 0:
                raise InvalidParameterError("denominator 必须是正的有限值，收到 %s" % n_pub)
            denom_public = True
        width = self.upper - self.lower
        sensitivity = mean_sensitivity(self.lower, self.upper,
                                       self.max_contributions, n_pub)
        # 在 [0,1] 缩放空间加噪（更紧、非负的敏感度），再线性还原
        y = (bounded.values - self.lower) / width
        true_scaled_mean = float(np.sum(y) / n_pub)
        scaled_sensitivity = self.max_contributions / n_pub
        if mechanism == MECHANISM_LAPLACE:
            noisy_scaled, scale_y = laplace_mechanism(
                true_scaled_mean, scaled_sensitivity, eps, self.randomness
            )
            noise_scale = width * scale_y
        else:
            noisy_scaled, sigma_y = gaussian_mechanism(
                true_scaled_mean, scaled_sensitivity, eps, d,
                self.randomness, method=self.gaussian_method,
            )
            noise_scale = width * sigma_y
        value = self.lower + width * float(noisy_scaled)
        # 预算在输出构造前才扣除（上面任何异常都不会扣费）
        self.accountant.spend(eps, d, label="%s:mean" % mechanism)
        return QueryResult(
            query_type="mean",
            mechanism=mechanism,
            value=value,
            epsilon_spent=eps,
            delta_spent=d,
            sensitivity=sensitivity,
            noise_scale=noise_scale,
            parameters=self._base_params(extra={
                "denominator": n_pub,
                "denominator_is_public": denom_public,
            }),
            test_mode=self.randomness.is_test,
            seed=self.randomness.seed,
        )

    def histogram(self, values, user_ids=None, epsilon=None, delta=None,
                  mechanism=MECHANISM_LAPLACE, bins=10, edges=None):
        """差分隐私直方图。

        两种分桶方式（二选一）：
        - ``bins``（正整数）：在 ``[lower, upper]`` 上均匀切出该数量的桶；
        - ``edges``（一维数组）：显式桶边界，要求严格递增且覆盖整个
          ``[lower, upper]``（``edges[0] <= lower`` 且 ``edges[-1] >= upper``），
          避免任何记录落在桶外而破坏敏感度推导。
        """
        eps, d = self._prepare_query(mechanism, epsilon, delta)
        bin_edges = self._make_edges(bins, edges)
        bounded = self._bound(values, user_ids)
        self._require_nonempty(bounded, "histogram")
        sensitivity_l1, sensitivity_l2 = histogram_sensitivity(
            self.max_contributions
        )
        idx = np.clip(
            np.digitize(bounded.values, bin_edges), 1, bin_edges.size - 1
        ) - 1
        true_counts = np.bincount(idx, minlength=bin_edges.size - 1).astype(float)
        return self._execute(
            "histogram", mechanism, true_counts,
            sensitivity_l1 if mechanism == MECHANISM_LAPLACE else sensitivity_l2,
            eps, d,
            extra_params={
                "n_bins": int(bin_edges.size - 1),
                "bin_edges": [float(x) for x in bin_edges],
            },
        )

    # ---- 内部实现 ------------------------------------------------------
    def _prepare_query(self, mechanism, epsilon, delta):
        """公共参数校验 + 预算预查（在任何数据处理之前）。"""
        if mechanism not in _MECHANISMS:
            raise InvalidParameterError(
                "mechanism 必须是 %s 之一，收到 %r" % (_MECHANISMS, mechanism)
            )
        if epsilon is None:
            raise InvalidParameterError("必须为每次查询显式指定 epsilon")
        try:
            eps = float(epsilon)
        except (TypeError, ValueError):
            raise InvalidParameterError("epsilon 必须是正实数，收到 %r" % (epsilon,))
        if not np.isfinite(eps) or eps <= 0:
            raise InvalidParameterError("epsilon 必须是正的有限值，收到 %s" % eps)
        d = self._query_delta(mechanism, delta)
        # Gaussian 标定的合法性（ε 范围等）在此提前触发，报错同样不依赖数据
        if mechanism == MECHANISM_GAUSSIAN:
            from .mechanisms import gaussian_sigma
            gaussian_sigma(1.0, eps, d, method=self.gaussian_method)
        if not self.accountant.check(eps, d):
            # 主动走 spend 以复用统一、信息完整的拒绝报错（不会真正扣费）
            self.accountant.spend(eps, d, label="%s:rejected" % mechanism)
        return eps, d

    def _query_delta(self, mechanism, delta):
        if mechanism == MECHANISM_LAPLACE:
            if delta is not None and float(delta) != 0.0:
                raise InvalidParameterError(
                    "Laplace 机制是纯 (ε,0)-DP，不接受 delta>0；"
                    "如需 (ε,δ)-DP 请使用 Gaussian 机制"
                )
            return 0.0
        if delta is None:
            d = self.accountant.total_delta
            if d <= 0:
                raise InvalidParameterError(
                    "Gaussian 机制需要 delta：请在构造引擎时设置总 delta>0，"
                    "或在查询时显式传入 delta"
                )
            return d
        try:
            d = float(delta)
        except (TypeError, ValueError):
            raise InvalidParameterError("delta 必须是 (0,1) 内实数，收到 %r" % (delta,))
        if not np.isfinite(d) or not (0.0 < d < 1.0):
            raise InvalidParameterError("Gaussian 查询要求 0 < delta < 1，收到 %s" % d)
        return d

    def _bound(self, values, user_ids):
        return bound_records(
            values, self.lower, self.upper,
            self.max_contributions, user_ids=user_ids,
        )

    @staticmethod
    def _require_nonempty(bounded, query_name):
        if bounded.is_empty:
            raise EmptyDataError(
                "%s 查询在贡献裁剪后没有任何记录；空数据下该查询不发布结果"
                % query_name
            )

    def _make_edges(self, bins, edges):
        if edges is not None:
            # 显式 edges 优先；同时传入 bins 时以 edges 为准（bins 被忽略）
            e = np.asarray(edges, dtype=np.float64)
            if e.ndim != 1 or e.size < 2:
                raise InvalidParameterError("edges 必须是长度 >=2 的一维数组")
            if not np.all(np.isfinite(e)):
                raise InvalidParameterError("edges 必须全部有限")
            if np.any(np.diff(e) <= 0):
                raise InvalidParameterError("edges 必须严格递增")
            if e[0] > self.lower or e[-1] < self.upper:
                raise InvalidParameterError(
                    "edges 必须覆盖整个取值范围：edges[0] <= lower(%s) 且 "
                    "edges[-1] >= upper(%s)，收到 [%s, %s]"
                    % (self.lower, self.upper, e[0], e[-1])
                )
            return e
        if not isinstance(bins, (int, np.integer)) or bins <= 0:
            raise InvalidParameterError("bins 必须是正整数，收到 %r" % (bins,))
        return np.linspace(self.lower, self.upper, int(bins) + 1)

    def _base_params(self, extra=None):
        params = {
            "lower": self.lower,
            "upper": self.upper,
            "max_contributions": self.max_contributions,
            "composition": self.accountant.composition,
            "total_epsilon": self.accountant.total_epsilon,
            "total_delta": self.accountant.total_delta,
        }
        if extra:
            params.update(extra)
        return params

    def _execute(self, query_type, mechanism, true_value, sensitivity,
                 epsilon, delta, extra_params):
        """加噪、扣费、构造结果。count/sum/histogram 共用。"""
        params = self._base_params(extra=extra_params)
        if mechanism == MECHANISM_GAUSSIAN:
            value, noise_scale = gaussian_mechanism(
                true_value, sensitivity, epsilon, delta,
                self.randomness, method=self.gaussian_method,
            )
            params["gaussian_method"] = self.gaussian_method
        else:
            value, noise_scale = laplace_mechanism(
                true_value, sensitivity, epsilon, self.randomness
            )
        if isinstance(value, np.ndarray):
            value = value.astype(float)
        self.accountant.spend(epsilon, delta, label="%s:%s" % (mechanism, query_type))
        return QueryResult(
            query_type=query_type,
            mechanism=mechanism,
            value=value,
            epsilon_spent=epsilon,
            delta_spent=delta,
            sensitivity=float(sensitivity),
            noise_scale=float(noise_scale),
            parameters=params,
            test_mode=self.randomness.is_test,
            seed=self.randomness.seed,
        )
