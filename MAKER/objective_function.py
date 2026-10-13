"""目标函数（ObjectiveFunction）

双目标优化思想（Eq. 84）：
    f1：数据似然目标 —— 让预测分布尽量贴近观测分布（越大越差，需最小化）
    f2：参考值目标 —— 让学到的参数尽量贴近参考值，保证可解释性

实现上把两者加权合并：total = f1 + lambda_f2 * f2。
"""
import numpy as np
from collections import defaultdict
from typing import Dict, Optional


class ObjectiveFunction:

    def __init__(self,
                 Data_train: np.ndarray,
                 likelihoodCounter,
                 param_manager,
                 maker,
                 registry,
                 n_outputs: int,
                 lambda_f2: float = 0.1):
        self.likelihoodCounter = likelihoodCounter
        self.param_manager = param_manager
        self.maker = maker
        self.registry = registry
        self.n_outputs = n_outputs
        self.n_features = registry.n_features
        self.lambda_f2 = lambda_f2

        self._extract_unique_inputs(Data_train)
        self._count_patterns(Data_train)
        self._record_reference_params()

    # ---------- 数据统计 ----------
    def _extract_unique_inputs(self, Data_train: np.ndarray):
        input_rows = Data_train[:, :self.n_features]
        unique = set(tuple(row) for row in input_rows)
        self.unique_inputs = np.array([list(t) for t in unique], dtype=int)
        self.input_to_idx = {tuple(row): i for i, row in enumerate(self.unique_inputs)}

    def _count_patterns(self, Data_train: np.ndarray):
        self.S_total = len(Data_train)
        input_counts = defaultdict(int)
        pattern_counts = defaultdict(int)
        for row in Data_train:
            it = tuple(row[:self.n_features])
            ts = int(row[-1])
            input_counts[it] += 1
            pattern_counts[(it, ts)] += 1

        self.patterns = []
        for (it, ts), S_A in pattern_counts.items():
            self.patterns.append({
                'input_tuple': it,
                'input_idx': self.input_to_idx[it],
                'true_state': ts,
                'S_A': S_A,
                'S_1lk': input_counts[it],
                'weight': S_A / self.S_total,
            })

    def _record_reference_params(self):
        pm = self.param_manager
        self.ref_w = {ic: np.ones(self.n_outputs) for ic in pm.w.keys()}
        self.ref_r = {ic: np.full(len(pm.r[ic]), 0.9) for ic in pm.r.keys()}
        self.ref_w_prior = np.ones(self.n_outputs)
        self.ref_r_prior = 0.9

    # ---------- 预测 ----------
    def update_all_predictions(self):
        self.predictions = {}
        for i, input_row in enumerate(self.unique_inputs):
            sample = np.zeros(self.n_features + 1, dtype=int)
            sample[:self.n_features] = input_row
            p, _ = self.maker.infer(sample, self.likelihoodCounter, self.param_manager)
            self.predictions[i] = p

    # ---------- f1：数据似然目标 ----------
    def evaluate_f1(self) -> float:
        self.update_all_predictions()
        max_weighted_sd = 0.0
        for pat in self.patterns:
            pred = self.predictions[pat['input_idx']]
            obs = np.zeros(self.n_outputs)
            obs[pat['true_state']] = 1.0
            sd = 0.5 * np.sum((pred - obs) ** 2)
            weighted = pat['weight'] * sd
            if weighted > max_weighted_sd:
                max_weighted_sd = weighted
        return max_weighted_sd

    # ---------- f2：参考值目标 ----------
    def evaluate_f2(self) -> float:
        pm = self.param_manager
        max_dev = 0.0
        for ic in pm.w.keys():
            dev = np.mean((pm.w[ic] - self.ref_w[ic]) ** 2)
            max_dev = max(max_dev, dev)
        for ic in pm.r.keys():
            dev = np.mean((pm.r[ic] - self.ref_r[ic]) ** 2)
            max_dev = max(max_dev, dev)
        max_dev = max(max_dev, np.mean((pm.w_prior - self.ref_w_prior) ** 2))
        max_dev = max(max_dev, (pm.r_prior - self.ref_r_prior) ** 2)
        return max_dev

    # ---------- 组合目标（供优化器调用） ----------
    def __call__(self, flat_params: np.ndarray) -> float:
        self.param_manager.unflatten_parameters(flat_params)
        return self.evaluate_f1() + self.lambda_f2 * self.evaluate_f2()

    def evaluate_total(self):
        f1 = self.evaluate_f1()
        f2 = self.evaluate_f2()
        return f1, f2, f1 + self.lambda_f2 * f2
