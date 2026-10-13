"""MAKER 最小套件 —— 运行demo（经典Iris ）

演示完整流程：
    加载 Iris → 离散化 → 建组合空间 → 统计似然 → 建参数 → 推理 → 优化 → 评估准确率

说明：Iris 是连续特征，而 MAKER 是离散证据框架，所以这里先把连续特征做
"等频分箱"离散化；并用 2 个特征演示。

运行方式（在 MAKER_Minimal 目录下）：
    python demo.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MAKER import (
    CombinationRegistry,
    LikelihoodCounter,
    ParameterManager,
    MAKERInference,
    ObjectiveFunction,
)


def load_iris_2features(n_bins=3):
    """加载 Iris，只取两个最有区分度的特征（花瓣长、花瓣宽），
    用分位数分箱离散化到 0..n_bins-1。
    注：-1 表示缺失/未知，由框架自动额外加入，这里数据无缺失所以用不到。"""
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data[:, [2, 3]]  # 花瓣长、花瓣宽
    y = iris.target           # 0=setosa, 1=versicolor, 2=virginica

    Xd = np.zeros_like(X, dtype=int)
    for j in range(X.shape[1]):
        # 等频分箱：取 n_bins-1 个分位点，把值切成 n_bins 段
        qs = np.quantile(X[:, j], np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
        Xd[:, j] = np.digitize(X[:, j], qs)
    return np.column_stack([Xd, y]), [str(n) for n in iris.target_names]


def accuracy(maker, data, lc, pm):
    """在整个数据上评估分类准确率"""
    correct = 0
    for row in data:
        p, _ = maker.infer(row, lc, pm)
        if int(np.argmax(p)) == int(row[-1]):
            correct += 1
    return correct / len(data)


def main():
    n_bins = 3
    data, class_names = load_iris_2features(n_bins)
    n_features, n_values, n_outputs = 2, [n_bins, n_bins], 3

    print(f"类别: {list(class_names)}")
    print(f"离散化后每个特征取值 0..{n_bins - 1}，框架会自动额外加入 -1 表示缺失/未知\n")

    registry = CombinationRegistry(n_features, n_values)
    lc = LikelihoodCounter(registry, data, n_outputs)
    pm = ParameterManager(registry, n_outputs)
    maker = MAKERInference(registry, n_outputs)
    obj = ObjectiveFunction(data, lc, pm, maker, registry, n_outputs, lambda_f2=0.1)

    sample = data[0].copy()
    f1, f2, total = obj.evaluate_total()
    print(f"优化前 f1={f1:.4f}, f2={f2:.4f}, total={total:.4f}, 准确率={accuracy(maker, data, lc, pm):.4f}")
    p, trust = maker.infer(sample, lc, pm)
    print(f"  样本特征={tuple(int(x) for x in sample[:n_features])} 推理 p_θ={np.round(p, 4)}, 信任度={trust:.4f}")

    # ---- 简单随机搜索优化 ----
    best = pm.flatten_parameters()
    best_val = obj(best)
    rng = np.random.default_rng(1)
    for _ in range(2000):
        cand = best + rng.normal(0.0, 0.05, len(best))
        cand = np.clip(cand, 0.0, 1.0)
        val = obj(cand)
        if val < best_val:
            best_val, best = val, cand
    pm.unflatten_parameters(best)

    f1, f2, total = obj.evaluate_total()
    print(f"\n优化后 f1={f1:.4f}, f2={f2:.4f}, total={total:.4f}")
    print(f"优化后准确率: {accuracy(maker, data, lc, pm):.4f}")
    p, trust = maker.infer(sample, lc, pm)
    print(f"  样本特征={tuple(int(x) for x in sample[:n_features])} 推理 p_θ={np.round(p, 4)}, 信任度={trust:.4f}")
    print(f"\n参数总数: {pm.get_parameter_count()}")
    print("说明：这里只用 2 个特征，这是为了简化演示。")


if __name__ == "__main__":
    main()
