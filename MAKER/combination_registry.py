"""组合注册器（CombinationRegistry）

负责生成"输入特征组合 × 参考值组合"的全部字典序映射。

约定（全部 0-based）：
    - 特征索引 0..L-1；
    - 每个特征的取值 0..n-1，另外用 -1 表示"未知/缺失"（每个特征自动 +1 个取值）；
    - input_combo  是无序组合，如 (0, 1)（(0,1) 与 (1,0) 是同一个）；
    - value_combo  是与之等长的有序取值，如 (2, -1)。

数据结构（可以想象成一个三维数据块）：
    x 轴 = 输入组合，y 轴 = 参考值组合，z 轴 = 列号（列联表里的列）。
"""
from typing import Dict, List, Tuple
from itertools import combinations, product


class CombinationRegistry:

    def __init__(self, n_features: int, n_values_per_feature: List[int]):
        self.n_features = n_features
        # 每个特征在原始取值数之上 +1，用于容纳 -1（未知/缺失）
        self.n_values_per_feature = [n + 1 for n in n_values_per_feature]

        # 所有输入组合（先按长度、再按字典序）
        self.input_combos: List[Tuple[int, ...]] = []
        # 每个输入组合对应的参考值组合列表
        self.value_combos: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
        # 双向映射：参考值组合 <-> 列号
        self.values_to_column: Dict[Tuple[int, ...], Dict[Tuple[int, ...], int]] = {}
        self.column_to_values: Dict[Tuple[int, ...], Dict[int, Tuple[int, ...]]] = {}

        self._build_all_mappings()

    def _build_all_mappings(self):
        for r in range(1, self.n_features + 1):
            for input_combo in combinations(range(self.n_features), r):
                self.input_combos.append(input_combo)

                n_values_list = [self.n_values_per_feature[feat] for feat in input_combo]
                # 取值范围应为 -1, 0, 1, ..., n-2，共 n 个取值
                # 注意：n 已是"含 -1"的总数（= 真实取值数 + 1），
                # 所以要用 range(-1, n-1)，否则会多出一个永远用不到的取值
                ranges = [range(-1, n - 1) for n in n_values_list]
                all_value_combos = list(product(*ranges))
                self.value_combos[input_combo] = all_value_combos

                val_to_col = {}
                col_to_val = {}
                for col_idx, value_combo in enumerate(all_value_combos):
                    val_to_col[value_combo] = col_idx
                    col_to_val[col_idx] = value_combo
                self.values_to_column[input_combo] = val_to_col
                self.column_to_values[input_combo] = col_to_val

    # ---------- 查询接口 ----------
    def get_n_cols(self, input_combo: Tuple[int, ...]) -> int:
        return len(self.value_combos[input_combo])

    def get_column(self, input_combo: Tuple[int, ...], value_combo: Tuple[int, ...]) -> int:
        return self.values_to_column[input_combo][value_combo]

    def get_value_combo(self, input_combo: Tuple[int, ...], col_idx: int) -> Tuple[int, ...]:
        return self.column_to_values[input_combo][col_idx]

    def get_all_input_combos(self) -> List[Tuple[int, ...]]:
        return self.input_combos.copy()

    def get_all_value_combos(self, input_combo: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        return self.value_combos[input_combo].copy()
