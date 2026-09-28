"""各查询类型的敏感度（sensitivity）显式计算。

约定：
  C = max_contributions   —— 每个用户最多贡献的记录数
  L, U                    —— 取值上下界，记 M = max(|L|, |U|)

邻接关系为“添加/移除一个用户”，且所有查询都经过贡献裁剪，因此敏感度
只依赖 (C, L, U, 公开分母) 这些公共参数，**与实际数据无关**。

count
  一个用户最多贡献 C 条记录，故  Δ_count = C。

sum
  每条记录取值在 [L, U]，一个用户最多 C 条，最大改变量为
  Δ_sum = C · max(|L|, |U|)。

mean（固定分母）
  均值按 sum / n_pub 发布，其中 n_pub 是调用方提供的**公开**分母
  （不随数据变化，例如已知的采样规模）。为得到更紧的、非负的敏感度，
  先把取值线性映射到 [0, 1]：y = (x - L) / (U - L)，
  发布 mean_y + Lap/Gauss 噪声，再线性还原。
  y 之和的敏感度是 C（每条 y∈[0,1]，最多 C 条），故
  Δ_mean_y = C / n_pub，还原到原始量纲后
  Δ_mean = (U - L) · C / n_pub。
  噪声同样按 (U - L) 倍缩放，因此 DP 保证不变。

histogram
  每条记录恰好落入一个桶；一个用户最多 C 条，故计数向量的
  L1 敏感度 Δ_hist,1 = C，L2 敏感度 Δ_hist,2 = C
  （全部 C 条记录落入同一个桶时取到）。
"""

import numpy as np

from .errors import InvalidParameterError
from .bounds import validate_bounds


def _check_cap(max_contributions):
    if not isinstance(max_contributions, (int, np.integer)) or max_contributions <= 0:
        raise InvalidParameterError(
            "max_contributions 必须是正整数，收到 %r" % (max_contributions,)
        )
    return int(max_contributions)


def count_sensitivity(max_contributions):
    """Δ_count = C。"""
    return float(_check_cap(max_contributions))


def sum_sensitivity(lower, upper, max_contributions):
    """Δ_sum = C · max(|L|, |U|)。"""
    lo, hi = validate_bounds(lower, upper)
    cap = _check_cap(max_contributions)
    return cap * max(abs(lo), abs(hi))


def mean_sensitivity(lower, upper, max_contributions, denominator):
    """Δ_mean = (U - L) · C / n_pub（原始量纲）。

    denominator 必须是正的有限公开分母 n_pub。
    """
    lo, hi = validate_bounds(lower, upper)
    cap = _check_cap(max_contributions)
    try:
        n_pub = float(denominator)
    except (TypeError, ValueError):
        raise InvalidParameterError("denominator 必须是正数，收到 %r" % (denominator,))
    if not np.isfinite(n_pub) or n_pub <= 0:
        raise InvalidParameterError("denominator 必须是正的有限值，收到 %s" % n_pub)
    return (hi - lo) * cap / n_pub


def histogram_sensitivity(max_contributions):
    """直方图计数向量的敏感度：返回 (Δ_L1, Δ_L2) = (C, C)。"""
    cap = _check_cap(max_contributions)
    return float(cap), float(cap)
