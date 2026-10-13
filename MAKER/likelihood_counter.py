"""似然计数器（LikelihoodCounter）

从训练数据里统计频率，再用"两步似然"得到任意证据的归一化似然，
见论文的 Table 5（频数表）→ Table 8（归一化似然）。

两步法：
    1. 似然：每个格子 ÷ 它所在"状态行"的总和   （频率 → 似然）
    2. 归一化：每个"观测列"再按列归一化          （似然 → 归一化似然）
"""
import numpy as np
from typing import Dict, Tuple
from itertools import combinations


class LikelihoodCounter:

    def __init__(self, registry, Data_train: np.ndarray, n_outputs: int):
        self.registry = registry
        self.n_features = registry.n_features
        self.n_outputs = n_outputs
        self.n_values = registry.n_values_per_feature

        # 先验分布（从输出标签统计）
        self.prior_distribution = None
        # raw_counts[input_combo] : 原始频数矩阵 [状态 × 列数]
        self.raw_counts: Dict[Tuple[int, ...], np.ndarray] = {}
        # normalized_probs[input_combo] : 归一化似然矩阵 [状态 × 列数]
        self.normalized_probs: Dict[Tuple[int, ...], np.ndarray] = {}

        self._compute_prior(Data_train)
        self._fit(Data_train)

    def _fit(self, Data_train: np.ndarray):
        for input_combo in self.registry.input_combos:
            n_cols = self.registry.get_n_cols(input_combo)
            self.raw_counts[input_combo] = np.zeros((self.n_outputs, n_cols), dtype=int)
            self.normalized_probs[input_combo] = np.zeros((self.n_outputs, n_cols))

        for row in Data_train:
            features = row[:self.n_features].astype(int)
            output = int(row[-1])
            for r in range(1, self.n_features + 1):
                for input_combo in combinations(range(self.n_features), r):
                    value_combo = tuple(int(features[i]) for i in input_combo)
                    col_idx = self.registry.get_column(input_combo, value_combo)
                    self.raw_counts[input_combo][output, col_idx] += 1

        self._compute_normalized_likelihoods()

    def _compute_normalized_likelihoods(self):
        for input_combo, raw in self.raw_counts.items():
            n_rows, n_cols = raw.shape
            row_sums = raw.sum(axis=1)

            # 第一步：÷ 行总和 → 似然
            likelihoods = np.zeros((n_rows, n_cols))
            for r in range(n_rows):
                if row_sums[r] > 0:
                    likelihoods[r, :] = raw[r, :] / row_sums[r]

            # 第二步：按列归一化 → 归一化似然
            normalized = np.zeros((n_rows, n_cols))
            for c in range(n_cols):
                col_sum = likelihoods[:, c].sum()
                if col_sum > 0:
                    normalized[:, c] = likelihoods[:, c] / col_sum
                else:
                    normalized[:, c] = np.ones(n_rows) / n_rows

            self.normalized_probs[input_combo] = normalized

    def get_likelihood(self, state: int, input_combo: Tuple[int, ...], value_combo: Tuple[int, ...]) -> float:
        """查询：给定状态、输入组合、参考值组合，返回归一化似然 p_θ(li)"""
        input_combo = tuple(input_combo)
        value_combo = tuple(value_combo)
        col_idx = self.registry.get_column(input_combo, value_combo)
        return self.normalized_probs[input_combo][state, col_idx]

    def _compute_prior(self, Data_train: np.ndarray):
        counts = np.zeros(self.n_outputs, dtype=int)
        for row in Data_train:
            output = int(row[-1])
            if 0 <= output < self.n_outputs:
                counts[output] += 1
        total = counts.sum()
        if total > 0:
            self.prior_distribution = counts / total
        else:
            self.prior_distribution = np.ones(self.n_outputs) / self.n_outputs
