"""端到端演示：在完全本地的环境下运行差分隐私统计查询。

用法：
    python examples/demo.py                # 正常模式（系统熵随机噪声）
    python examples/demo.py --seed 42      # 测试模式（可重复）

该脚本不进行任何网络访问；所有数据仅存在于内存中。
"""

import argparse
import json
import os
import sys

import numpy as np

# 允许在未 pip install 的情况下直接从仓库根目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldp_stats import (
    MECHANISM_GAUSSIAN,
    MECHANISM_LAPLACE,
    BudgetExceededError,
    PrivacyEngine,
)


def main():
    parser = argparse.ArgumentParser(description="ldp_stats 本地差分隐私演示")
    parser.add_argument("--seed", type=int, default=None,
                        help="给定整数种子即进入测试模式（可重复）；缺省为正常模式")
    args = parser.parse_args()

    rng = np.random.default_rng(2026)
    # 200 名用户，每人 1~3 条记录，取值 [0, 100]
    user_ids = np.repeat(np.arange(200), rng.integers(1, 4, size=200))
    values = rng.uniform(0, 100, size=user_ids.size)

    print("=" * 68)
    print(" ldp_stats 本地差分隐私统计演示")
    print(" 模式: %s" % ("测试模式 seed=%d" % args.seed if args.seed is not None
                          else "正常模式（系统熵，非固定噪声）"))
    print(" 原始记录数: %d（仅本地内存，输出中不会出现）" % values.size)
    print("=" * 68)

    engine = PrivacyEngine(
        epsilon=2.0, delta=1e-5,
        lower=0.0, upper=100.0,
        max_contributions=2,
        composition="basic",
        seed=args.seed,
        gaussian_method="analytic",
    )

    def show(title, result, true_value=None):
        print("\n【%s】" % title)
        if true_value is not None:
            print("  真实值（仅本地对比，不会进入输出对象）: %.4f" % true_value)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    res = engine.count(values, user_ids=user_ids, epsilon=0.8)
    show("count · Laplace", res)

    res = engine.sum(values, user_ids=user_ids, epsilon=0.5)
    show("sum · Laplace", res, true_value=float(values.sum()))

    res = engine.mean(values, user_ids=user_ids, epsilon=0.4)
    show("mean · Laplace", res, true_value=float(values.mean()))

    res = engine.histogram(values, user_ids=user_ids, epsilon=0.2, bins=5)
    show("histogram · Laplace", res, true_value=None)

    # Gaussian 查询（计入 δ 预算）
    res = engine.count(values, user_ids=user_ids, epsilon=0.09,
                       mechanism=MECHANISM_GAUSSIAN)
    show("count · Gaussian (analytic)", res)

    print("\n预算账本:")
    for item in engine.accountant.ledger():
        print("  - %-22s eps=%.4g delta=%.2g"
              % (item["label"], item["epsilon"], item["delta"]))
    print("累计: eps=%.4f / %.1f, delta=%.2g / %.0e"
          % (engine.accountant.used_epsilon,
             engine.accountant.total_epsilon,
             engine.accountant.used_delta,
             engine.accountant.total_delta))

    # 再发一笔必然超预算的查询，演示“拒绝且不泄露”
    print("\n尝试超出剩余预算的查询 ...")
    try:
        engine.count(values, user_ids=user_ids, epsilon=1.0)
    except BudgetExceededError as exc:
        print("  已拒绝 -> %s" % exc)
    print("拒绝后已用 eps 仍为 %.4f（未扣费、未生成结果）"
          % engine.accountant.used_epsilon)


if __name__ == "__main__":
    main()
