"""localdp: 纯本地差分隐私统计库。

所有计算均在本地内存/文件中完成，不依赖任何远程服务、数据库或云平台。
"""

from .accountant import Budget, PrivacyAccountant
from .engine import LocalDPEngine, QueryResult
from .exceptions import BudgetExhaustedError, DPError, InvalidParameterError
from .mechanisms import gaussian_sigma, laplace_scale

__all__ = [
    "Budget",
    "PrivacyAccountant",
    "LocalDPEngine",
    "QueryResult",
    "DPError",
    "InvalidParameterError",
    "BudgetExhaustedError",
    "laplace_scale",
    "gaussian_sigma",
]

__version__ = "0.1.0"
