"""库内统一异常定义。"""


class DPError(Exception):
    """差分隐私库的基础异常。"""


class InvalidParameterError(DPError, ValueError):
    """非法参数：如 epsilon <= 0、delta 不在 (0, 1)、上下界颠倒等。"""


class BudgetExhaustedError(DPError):
    """隐私预算不足：查询被拒绝，不返回任何结果。"""
