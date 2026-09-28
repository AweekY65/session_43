"""异常类型：所有错误均在本地抛出，不包含任何网络/服务交互。"""


class DPError(Exception):
    """库内所有自定义异常的基类。"""


class InvalidParameterError(DPError):
    """非法的 epsilon / delta / 数据范围 / 查询参数。

    错误信息只描述参数本身的问题，绝不携带输入数据中的任何记录，
    因此异常本身也不会泄露单条原始记录。
    """


class EmptyDataError(DPError):
    """在不允许空数据的查询上收到了空输入（裁剪之后没有任何记录）。"""


class BudgetExceededError(DPError):
    """本次查询需要的隐私预算超过账户剩余预算。

    该异常在任何噪声生成与结果计算之前抛出，因此不会泄露任何结果；
    账户状态保持不变（本次查询不计费）。
    """
