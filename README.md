# ldp_stats — 本地差分隐私统计库

一个**完全本地运行**的差分隐私（Differential Privacy, DP）统计库。所有输入
数据、隐私预算、随机状态与输出结果只存在于调用方进程的内存（或调用方自行
写入的本地文件）中；库本身**不发起任何网络请求**，不依赖远程分析服务、
数据库、云平台或任何外部服务。唯一的第三方依赖是 NumPy / SciPy 这类本地
数值计算库。

支持带上下界约束与贡献裁剪的 `count`、`sum`、`mean`、`histogram` 查询，
提供 Laplace 机制与 Gaussian 机制、隐私预算账户、确定性贡献裁剪，以及可
注入固定种子的测试模式。

---

## 1. 安装

需要 Python ≥ 3.10。任选其一：

```bash
# 方式 A：独立 conda 环境（推荐）
conda env create -f environment.yml
conda activate ldp_stats

# 方式 B：pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

不安装也可以直接从仓库根目录运行测试与示例。

---

## 2. 快速开始

```python
import numpy as np
from ldp_stats import PrivacyEngine, MECHANISM_GAUSSIAN

values   = np.random.default_rng(0).uniform(0, 100, size=500)
user_ids = np.repeat(np.arange(100), 5)   # 100 个用户，每人 5 条

engine = PrivacyEngine(
    epsilon=2.0, delta=1e-5,          # 总隐私预算
    lower=0.0, upper=100.0,           # 公开取值范围
    max_contributions=2,              # 每用户最多 2 条记录
    composition="basic",              # 预算组合方法
    # seed=None 为默认的正常模式（系统熵、非固定噪声）
)

r = engine.count(values, user_ids=user_ids, epsilon=0.8)
print(r.value)                 # 加噪后的计数值（float）
print(r.to_dict())             # 含机制/预算/敏感度/噪声尺度的安全输出

engine.sum(values, user_ids=user_ids, epsilon=0.5)
engine.mean(values, user_ids=user_ids, epsilon=0.4)
engine.histogram(values, user_ids=user_ids, epsilon=0.2, bins=10)

# Gaussian 机制（(ε,δ)-DP），可显式给 delta，否则使用账户总 delta
engine.count(values, user_ids=user_ids, epsilon=0.05,
             mechanism=MECHANISM_GAUSSIAN)

print(engine.accountant.used_epsilon, engine.accountant.remaining_epsilon)
```

运行端到端演示（正常模式 / 测试模式）：

```bash
python examples/demo.py
python examples/demo.py --seed 42     # 可重复
```

---

## 3. 隐私定义

### 3.1 邻接关系与威胁模型

- 邻接数据集定义为**添加/移除一个用户**（add/remove adjacency），即保护
  “任意单个用户的全部贡献”。
- 每个用户最多保留 `max_contributions = C` 条记录；每条记录取值裁剪到
  公开区间 `[L, U]`。由此任何查询的敏感度都只依赖公共参数，**与实际数据
  无关**，单个用户无法通过贡献海量记录使敏感度无限增大。
- 机制输出仅为加噪后的聚合统计量。库的信任边界是“本地进程”：调用方
  自然能看到自己持有的原始数据；库保证的是**输出结果**不会在差分隐私
  语义下额外暴露单条记录。

### 3.2 机制保证

| 机制 | 保证 | 噪声 | 数值查询噪声尺度 |
|---|---|---|---|
| Laplace | 纯 **(ε, 0)-DP** | Laplace(0, b)，b = Δ/ε（向量按 L1 敏感度逐维加噪） | ε>0 |
| Gaussian (`classic`) | **(ε, δ)-DP** | N(0, σ²)，σ = (Δ/ε)·√(2 ln(1.25/δ))，要求 0<ε<1 | Dwork–Roth (2014) Thm A.1 |
| Gaussian (`analytic`) | **(ε, δ)-DP** | 对 Φ_Δ(ε,σ) ≤ δ 二分搜索最小 σ，界更紧，允许 ε>0 | Balle–Wang (2018) |

### 3.3 敏感度规则（均显式计算并写入输出的 `sensitivity` 字段）

设 `C = max_contributions`，`M = max(|L|, |U|)`：

| 查询 | 敏感度 | 说明 |
|---|---|---|
| `count` | **Δ = C** | 一个用户最多贡献 C 条记录 |
| `sum` | **Δ = C·M** | C 条记录，每条最多改变 M |
| `mean` | **Δ = (U−L)·C/n_pub** | 固定公开分母 n_pub；先将取值线性映射到 [0,1] 再加噪后还原，得到非负且更紧的敏感度 |
| `histogram` | **Δ₁ = C（Laplace 用），Δ₂ = C（Gaussian 用）** | 每条记录只落一个桶；C 条全落同一桶时 L1/L2 同时取到 C |

`mean` 的分母 n_pub 必须是**公开信息**（不随数据变化）。缺省使用裁剪后
的记录数，这隐含“数据集规模公开”的假设；若该假设不成立，请显式传入
`denominator=`（例如已知采样规模）。

### 3.4 预算组合方法

账户在构造时选择组合方法，多次查询的开销逐笔累计：

- **`basic`（基本组合）**：k 次 (εᵢ, δᵢ)-DP 串行组合后满足
  (Σεᵢ, Σδᵢ)-DP。Laplace 查询 δ=0。
- **`advanced`（高级组合，Dwork–Rothblum–Vadhan 2010；Dwork–Roth 2014
  Thm 3.20）**：k 次 (ε₀, δ₀)-DP（要求 0<ε₀<1）在整体 δ′ 预算下
  ε_total = 2ε₀√(2k ln(1/δ′)) + 2kε₀²，δ_total = δ′ + Σδ₀。
  混批不同 ε₀ 时按最大 ε₀ 重算，是合法但略保守的上界。

### 3.5 拒绝语义：超预算不泄露结果

每次查询的执行顺序为：

1. 校验机制与公共参数（不碰数据）；
2. 向账户**预查**预算是否足够（不改账户状态）；
3. 确定性贡献裁剪；
4. 计算真值并加噪；
5. 扣除预算并返回结果。

预算不足时在第 2 步即抛 `BudgetExceededError`：**不计算结果、不生成
噪声、不扣除预算**，因此拒绝路径不泄露任何查询结果，也无法通过账户
状态的变化推断数据。

---

## 4. 贡献裁剪规则（确定性）

1. 按 `user_ids` 分组，保持每个用户记录的原始出现顺序；
2. 每个用户只保留**最先出现的 C 条**（first-C wins；无随机、无歧义）；
3. 对保留记录做取值裁剪 `clip(x, L, U)`。

`user_ids=None` 时视为每条记录属于独立用户。被截断用户的丢弃条数只保留
在内部对象 `BoundedData.truncated_users` 中供本地调试，**不会出现在
对外输出里**。

---

## 5. 随机状态与测试模式

- `PrivacyEngine(..., seed=None)`（**默认，正常模式**）：使用操作系统熵
  构造独立的 `numpy.random.Generator`，**绝不默认固定噪声种子**，也不触碰
  NumPy 全局随机状态（`np.random.*`），多个引擎实例的噪声相互独立。
- `seed=<整数>`：测试模式，相同种子 + 相同查询得到完全一致的噪声，便于
  回归测试。输出对象会带 `test_mode: true` 与 `seed` 字段；正常模式下
  `test_mode: false`、`seed: null`。

---

## 6. 输出内容与隐私安全

每次查询返回 `QueryResult`，`to_dict()` 可直接 JSON 序列化，包含：

- `mechanism`：`"laplace"` / `"gaussian"`；
- `budget`：本笔 `epsilon_spent`、`delta_spent`；
- `sensitivity`、`noise_scale`（Laplace 尺度 b 或 Gaussian 标准差 σ）；
- `parameters`：公共查询参数（上下界、C、组合方法、总分桶边界、分母等）；
- `value`：加噪结果；
- `test_mode` 及（仅测试模式）`seed`。

输出**不包含**原始记录、按用户的贡献明细、未加噪真值或任何能还原单条
原始记录的信息。

### 非法参数（抛 `InvalidParameterError`）

- ε 非正、非有限；Gaussian 的 δ ∉ (0,1)；classic Gaussian 的 ε ≥ 1；
- `lower >= upper`、非有限边界；`max_contributions` 非正整数；
- Laplace 查询传入 δ>0；`denominator` 非正；
- 直方图 `edges` 非严格递增或不覆盖 `[L, U]`，`bins` 非正；
- 输入记录含 NaN/Inf，或 `user_ids` 与记录不等长。

### 空数据

裁剪后为空时，`count/sum/histogram`（以及未显式给分母的 `mean`）抛
`EmptyDataError` 且不扣预算；显式给出公开 `denominator` 的 `mean`
允许空数据（发布值仅为下界加噪，不反映任何记录）。

---

## 7. 适用限制（请务必阅读）

- 提供的是**中心化/可信聚合者模型**的 DP 机制（噪声在聚合端添加）；
  “本地”指计算与数据完全在本机完成，而非本地差分隐私（LDP）的逐记录
  随机响应。若需要端侧不可信聚合者的 LDP，请改用逐记录随机化机制。
- 邻接关系是“添加/移除一个用户”，依赖**正确、稳定的用户标识**与合理的
  `max_contributions`；标识缺失或被攻击时该保护退化。
- `mean` 默认分母隐含“数据规模公开”；直方图桶边界与 `[L,U]` 必须是
  公共选择，按数据挑边界会泄露信息。
- 不做后处理以外的防护：调用方若把多次输出与其它侧信道结合推断，不在
  本库控制范围内；不要把内部 `BoundedData`（含按用户明细）对外暴露。
- 高级组合是通用上界，紧度不如 RDP/zCDP 等现代会计方法；小规模预算下
  可能偏保守。
- 结果未做整数化/非负化等后处理（会改变统计量但不破坏 DP）；如需要
  可由调用方对发布值自行后处理（DP 对后免疫）。

---

## 8. 运行测试

全部测试在终端运行并显示结果：

```bash
./run_tests.sh
# 或
python -m pytest tests/ -v
```

测试覆盖（77 个用例）：

| 文件 | 覆盖场景 |
|---|---|
| `tests/test_sensitivity.py` | 四类查询敏感度公式、非对称边界、非法参数 |
| `tests/test_mechanisms.py` | Laplace/Gaussian 标定、analytic 更紧、非法 ε/δ、固定种子可重复、正常模式非固定、不触碰 NumPy 全局状态、Laplace/Gaussian 噪声分布（均值、方差、KS 检验） |
| `tests/test_accountant.py` | 基本/高级组合累计、δ 累计、耗尽拒绝后状态不变（不泄露）、高级组合闭式值核对、边界恰好耗尽 |
| `tests/test_bounding.py` | first-C 确定性截断、取值裁剪顺序、空数据、海量贡献受限、非法输入 |
| `tests/test_engine.py` | 各查询蒙特卡洛无偏性、Gaussian 端到端、预算耗尽拒绝且无结果、重用户裁剪、种子可重复、空数据、非法参数、输出不含原始记录（JSON 扫描） |

---

## 9. 项目结构

```
ldp_stats/
  __init__.py       # 公共 API
  errors.py         # DPError / InvalidParameterError / EmptyDataError / BudgetExceededError
  bounds.py         # 取值校验 + 确定性贡献裁剪
  sensitivity.py    # count/sum/mean/histogram 敏感度
  mechanisms.py     # Laplace / Gaussian(classic|analytic) + RandomnessProvider
  accountant.py     # basic / advanced 预算账户
  engine.py         # PrivacyEngine 与 QueryResult
tests/              # pytest 自动化测试
examples/demo.py    # 端到端演示
environment.yml     # 独立 conda 环境
```

## 参考文献

- Dwork, C. & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*.
- Dwork, Rothblum & Vadhan (2010). Boosting and Differential Privacy (FOCS).
- Balle & Wang (2018). Improving the Gaussian Mechanism for Differential Privacy (ICML).
