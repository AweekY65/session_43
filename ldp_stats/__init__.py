"""ldp_stats — 一个完全本地运行的差分隐私统计库。

所有输入数据、隐私预算、随机状态与输出结果均只存在于调用方进程的内存
或本地文件中；库本身不发起任何网络请求，不依赖数据库、云平台或远程分析
服务。仅依赖 NumPy / SciPy 这类本地数值计算库。
"""

from .errors import (
    BudgetExceededError,
    DPError,
    EmptyDataError,
    InvalidParameterError,
)
from .accountant import BudgetAccountant
from .bounds import BoundedData, bound_records, validate_bounds
from .engine import PrivacyEngine, QueryResult
from .mechanisms import (
    MECHANISM_GAUSSIAN,
    MECHANISM_LAPLACE,
    RandomnessProvider,
    gaussian_sigma,
    laplace_scale,
)
from .sensitivity import (
    count_sensitivity,
    histogram_sensitivity,
    mean_sensitivity,
    sum_sensitivity,
)

__all__ = [
    "BudgetAccountant",
    "BudgetExceededError",
    "BoundedData",
    "DPError",
    "EmptyDataError",
    "InvalidParameterError",
    "MECHANISM_GAUSSIAN",
    "MECHANISM_LAPLACE",
    "PrivacyEngine",
    "QueryResult",
    "RandomnessProvider",
    "bound_records",
    "count_sensitivity",
    "gaussian_sigma",
    "histogram_sensitivity",
    "laplace_scale",
    "mean_sensitivity",
    "sum_sensitivity",
    "validate_bounds",
]

__version__ = "0.1.0"
