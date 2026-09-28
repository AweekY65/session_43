# localdp — 本地差分隐私统计库

一个**纯本地**的差分隐私（Differential Privacy）统计库：所有输入数据、隐私预算、
随机状态与输出结果只存在于本地内存或本地文件中，**不依赖任何远程分析服务、
数据库、云平台或外部服务**。仅使用 NumPy 作为第三方依赖。

## 安装与运行

无需额外环境，Python 3.10+ 且装有 NumPy 即可（也可自行创建独立 conda 环境）：

```bash
pip install numpy pytest
python -m pytest        # 运行全部测试
```

## 快速上手

```python
from localdp import LocalDPEngine

# 总预算 eps=1.0；每用户最多贡献 2 条记录；seed 仅供测试，生产不要传
engine = LocalDPEngine(total_epsilon=1.0, max_contributions=2)

users = ["u1", "u1", "u2", "u3"]
values = [25.0, 30.0, 41.0, 35.0]

res = engine.mean(users, values, lower=0.0, upper=100.0, epsilon=0.5)
print(res.to_dict())
# {'value': ..., 'mechanism': 'laplace', 'epsilon': 0.5, 'delta': 0.0,
#  'sensitivity': 200.0, 'noise_parameter': ..., 'params': {...}}

engine.count(users, epsilon=0.6)   # 抛出 BudgetExhaustedError：0.5+0.6 > 1.0
```

## 隐私定义

- **邻接关系**：add/remove（无界 DP）。两个数据集相邻，当且仅当其中一个可通过
  增删**单个用户**的全部贡献得到另一个（用户级隐私，而非记录级）。
- **Laplace 机制**提供纯 ε-DP；**Gaussian 机制**提供 (ε, δ)-DP，
  σ = √(2·ln(1.25/δ))·Δ/ε（要求 ε < 1）。
- **Contribution bounding**：若不限制单用户贡献条数，用户级敏感度将无界。
  库按**确定性规则**裁剪——保持输入原始顺序，每个 `user_id` 仅保留前
  `max_contributions` 条记录，其余丢弃。同一输入永远得到同一裁剪结果。

## 敏感度规则

设 `k = max_contributions`：

| 查询 | 预处理 | 敏感度 Δ |
|---|---|---|
| `count` | contribution bounding | `k` |
| `sum` | 值裁剪到 `[lower, upper]` | `k · max(|lower|, |upper|)` |
| `mean` | 预算对半拆分：noisy_sum / noisy_count | sum 支路 `k · max(|lower|, |upper|)`，count 支路 `k` |
| `histogram` | 每条记录恰好落入一个桶 | L1 敏感度 `k`（各桶独立加 Laplace 噪声，scale = Δ/ε） |

`mean` 将 ε（及 δ）对半分给 sum 与 count 两个子查询，再相除；加噪后计数 ≤ 0 时
返回 `value=None` 而不泄露真实计数。空数据上 `mean` 抛出 `DPError`。

## 预算组合方法

`PrivacyAccountant` 采用**基本组合（basic composition）**：

- 多次查询的 ε 线性累加，δ 线性累加；
- 每次查询**先扣预算再计算**：`spent + requested > total` 时抛出
  `BudgetExhaustedError`，查询不执行、不返回任何结果、不扣减预算；
- `mean` 为原子操作：先整体校验预算，避免 sum 扣费成功而 count 失败的半扣费状态。

## 随机性

- **正常模式**（默认，`seed=None`）：`numpy.random.default_rng()` 从操作系统熵源
  取种子，不会退化为固定噪声；
- **测试模式**：构造函数传入 `seed=<int>`，结果完全可重复。请勿在生产中使用。

## 输出安全

`QueryResult`（及 `to_dict()`）只包含：查询值、使用的机制、本次预算消耗
(ε, δ)、敏感度、噪声参数（Laplace scale 或 Gaussian σ）与查询参数（上下界、
bins 等）。**不包含**任何用户 ID、原始记录或可还原单条记录的信息。

## 参数校验

以下非法输入均抛出 `InvalidParameterError`：

- `epsilon <= 0`、非有限值；`delta` 不在开区间 (0, 1)；
- `lower >= upper` 或上下界非有限值；
- Gaussian 机制缺 `delta`、ε ≥ 1，或 Laplace 机制误传 `delta`；
- `max_contributions < 1`、输入长度不一致、空/重复 bins、未知机制名。

## 适用限制

- 组合规则为基本组合，大量查询时预算消耗保守；如需更紧的界（RDP/zCDP）
  需自行扩展账户模块；
- 经典 Gaussian 机制仅在 ε < 1 时有理论保证，库在 ε ≥ 1 时拒绝；
- 直方图为类别型（显式 bins），非等宽数值分箱；
- 浮点运算的无穷小精度问题（floating-point attacks）未做特殊防护，
  高安全场景建议使用定点/整数噪声实现；
- 库只保证发布结果的差分隐私性，输入数据在本地内存中的安全由部署方负责。

## 项目结构

```
localdp/
  mechanisms.py   # Laplace / Gaussian 机制与参数校验
  accountant.py   # 隐私预算账户（基本组合）
  bounds.py       # contribution bounding（确定性裁剪）
  engine.py       # count / sum / mean / histogram 查询引擎
  exceptions.py   # DPError / InvalidParameterError / BudgetExhaustedError
tests/            # 49 个自动化测试
```

## 测试

```bash
python -m pytest            # 运行全部测试（verbose）
python -m pytest tests/test_budget.py   # 只跑预算相关
```

覆盖场景：统计正确性、预算累计与耗尽拒绝、边界裁剪、contribution bounding、
随机种子可重复性、空数据、非法参数、噪声分布基本性质（均值/方差）与输出安全性。
