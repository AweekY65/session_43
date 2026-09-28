"""随机种子：测试模式可重复，正常模式不固定。"""

import pytest

from localdp import LocalDPEngine


def test_same_seed_reproducible():
    users = [f"u{i}" for i in range(100)]
    r1 = LocalDPEngine(total_epsilon=10.0, seed=42).count(users, epsilon=1.0).value
    r2 = LocalDPEngine(total_epsilon=10.0, seed=42).count(users, epsilon=1.0).value
    assert r1 == r2


def test_different_seeds_differ():
    users = [f"u{i}" for i in range(100)]
    r1 = LocalDPEngine(total_epsilon=10.0, seed=1).count(users, epsilon=1.0).value
    r2 = LocalDPEngine(total_epsilon=10.0, seed=2).count(users, epsilon=1.0).value
    assert r1 != r2


def test_normal_mode_not_fixed():
    # 不传种子时随机性来自 OS 熵源，两次独立运行结果应不同
    users = [f"u{i}" for i in range(100)]
    values = {
        LocalDPEngine(total_epsilon=10.0).count(users, epsilon=1.0).value
        for _ in range(5)
    }
    assert len(values) > 1
