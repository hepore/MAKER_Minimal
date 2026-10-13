"""MAKER最小套件

模块与论文的对应关系：
    CombinationRegistry   —— 组合空间（输入组合 × 参考值组合）
    LikelihoodCounter     —— 似然方法（频率 → 似然 → 归一化似然），论文 Table 5 → Table 8
    ParameterManager      —— 证据权重 w、可靠性 r、先验参数，论文 Eq.(19)(21)(定义8)
    MAKERInference        —— MAKER 规则组合推理，论文 Theorem 1 / Corollary 1.4
    ObjectiveFunction     —— 双目标学习（f1 数据似然 + f2 参考值偏差），论文 Eq.(84)
"""

from .maker_inference import MAKERInference, CompositeEvidence
from .parameter_manager import ParameterManager
from .objective_function import ObjectiveFunction
from .combination_registry import CombinationRegistry
from .likelihood_counter import LikelihoodCounter

__version__ = "1.0.0"
__all__ = [
    "MAKERInference",
    "CompositeEvidence",
    "ParameterManager",
    "ObjectiveFunction",
    "CombinationRegistry",
    "LikelihoodCounter",
]
