# MAKER 最小套件

这是 **Maximum Likelihood Evidential Reasoning（极大似然证据推理，MAKER）** 的最小实现，
仅保留论文核心框架，用于入门学习与二次开发。

> 对应论文：Yang, J.-B., & Xu, D.-L. (2025). *Maximum Likelihood Evidential Reasoning*. Artificial Intelligence, 340, 104289.

代码实现：hepore与deepseek v4 pro
代码检查：hepore

## 目录结构

```
MAKER_Minimal/
├── MAKER/
│   ├── __init__.py              # 统一导出
│   ├── combination_registry.py  # 组合空间（输入组合 × 参考值组合）
│   ├── likelihood_counter.py    # 似然方法（频率 → 似然 → 归一化似然）
│   ├── parameter_manager.py     # 证据权重 w、可靠性 r、先验参数
│   ├── maker_inference.py       # MAKER 规则组合推理
│   └── objective_function.py    # 双目标学习（f1 数据 + f2 参考值）
├── demo.py                      # 运行示例（Iris 数据集）
└── requirements.txt             # 核心包 numpy；demo 另需 scikit-learn
```

## 模块与论文的对应

| 模块 | 论文对应 | 关键公式 |
|---|---|---|
| `CombinationRegistry` | 系统空间 / 组合空间 | 所有输入组合与参考值组合 |
| `LikelihoodCounter` | Table 5 → Table 8 | 两步似然归一化 |
| `ParameterManager` | Eq.(19)(21)，定义 7/8 | 权重 w、可靠性 r、先验 |
| `MAKERInference` | Theorem 1 / Corollary 1.4 | MAKER 规则、ER 规则 |
| `ObjectiveFunction` | Eq.(84) | f1 + λ·f2 |

## 快速开始

```bash
pip install -r requirements.txt
python demo.py
```

## 推理流程

```
数据 → 似然计数（两步归一化）
     → 证据投影到 ERPS（ω 折扣）
     → MAKER 规则递归组合多条证据（依赖指数 ᾱ、相对投影率 ω̄、相对权重 w̄）
     → 组合先验证据（独立 ER 规则）
     → 投影回 SOPS，输出 p_θ 与信任度 t
```

## 说明

- 本套件是**教学/二次开发用的最小版本**，只实现了论文的核心推理链路。
- 权重、可靠性等参数可用任意无梯度优化器学习。
