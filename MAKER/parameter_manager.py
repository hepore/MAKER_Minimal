"""参数管理器（ParameterManager）

管理推理过程中的参数，即"模型本身"：
    - w[input_combo]      : 权重向量，形状 (n_outputs,)，对应论文 w_θl 与 w_θ(1∧l)
    - r[input_combo]      : 可靠性向量，形状 (n_cols,)，对应论文 r_li 与 r_(1∧l)k
    - w_prior / r_prior   : 先验证据的权重 / 可靠性

注意区分两种参数的"维度"：
    - 权重 w 依赖"输出状态"（每个状态一个权重）；
    - 可靠性 r 依赖"参考值组合"（每个取值组合一个可靠性）。
"""
import numpy as np
import json
from typing import Dict, Tuple


class ParameterManager:

    def __init__(self, registry, n_outputs: int):
        self.registry = registry
        self.n_features = registry.n_features
        self.n_values = registry.n_values_per_feature
        self.n_outputs = n_outputs

        # 权重 / 可靠性（数据证据）
        self.w: Dict[Tuple[int, ...], np.ndarray] = {}
        self.r: Dict[Tuple[int, ...], np.ndarray] = {}

        # 先验参数（独立存储）
        self.w_prior = np.full(n_outputs, 0.9)
        self.r_prior = 0.8

        # 参数边界
        self.bounds = {'w': (0.0, 1.0), 'r': (0.0, 1.0)}

        self._initialize_parameters()

    def _initialize_parameters(self):
        for input_combo in self.registry.input_combos:
            self.w[input_combo] = np.full(self.n_outputs, 0.9)
            n_cols = self.registry.get_n_cols(input_combo)
            self.r[input_combo] = np.full(n_cols, 0.9)

    # ---------- 权重接口 ----------
    def get_w_vector(self, input_combo: Tuple[int, ...]) -> np.ndarray:
        input_combo = tuple(input_combo)
        if input_combo not in self.w:
            return np.full(self.n_outputs, self.bounds['w'][0])
        return self.w[input_combo].copy()

    def set_w_vector(self, input_combo: Tuple[int, ...], vector: np.ndarray):
        input_combo = tuple(input_combo)
        if len(vector) != self.n_outputs:
            raise ValueError(f"权重向量长度应为 {self.n_outputs}，实际 {len(vector)}")
        if input_combo not in self.w:
            raise ValueError(f"输入组合 {input_combo} 不存在")
        self.w[input_combo] = np.array(vector)

    # ---------- 可靠性接口 ----------
    def get_r(self, input_combo: Tuple[int, ...], value_combo: Tuple[int, ...]) -> float:
        input_combo = tuple(input_combo)
        value_combo = tuple(value_combo)
        if value_combo not in self.registry.values_to_column[input_combo]:
            return self.bounds['r'][0]
        col_idx = self.registry.get_column(input_combo, value_combo)
        return self.r[input_combo][col_idx]

    def set_r(self, input_combo: Tuple[int, ...], value_combo: Tuple[int, ...], value: float):
        input_combo = tuple(input_combo)
        value_combo = tuple(value_combo)
        col_idx = self.registry.get_column(input_combo, value_combo)
        self.r[input_combo][col_idx] = value

    # ---------- 先验接口 ----------
    def get_w_prior_vector(self) -> np.ndarray:
        return self.w_prior

    def set_w_prior_vector(self, value: np.ndarray):
        self.w_prior = np.array(value)

    def get_r_prior(self) -> float:
        return self.r_prior

    def set_r_prior(self, value: float):
        self.r_prior = value

    # ---------- 优化用接口（展平 / 还原） ----------
    def flatten_parameters(self) -> np.ndarray:
        """把所有参数展平为一维向量，供优化器使用"""
        flat = []
        for input_combo in sorted(self.w.keys()):
            flat.extend(self.w[input_combo])
        for input_combo in sorted(self.r.keys()):
            flat.extend(self.r[input_combo])
        flat.extend(self.w_prior)
        flat.append(self.r_prior)
        return np.array(flat)

    def unflatten_parameters(self, flat: np.ndarray):
        """从优化器向量还原参数（顺序必须与 flatten 一致）"""
        idx = 0
        for input_combo in sorted(self.w.keys()):
            length = self.n_outputs
            self.w[input_combo] = flat[idx:idx + length].copy()
            idx += length
        for input_combo in sorted(self.r.keys()):
            length = len(self.r[input_combo])
            self.r[input_combo] = flat[idx:idx + length].copy()
            idx += length
        self.w_prior = flat[idx:idx + self.n_outputs].copy()
        idx += self.n_outputs
        self.r_prior = flat[idx]

    def get_parameter_count(self) -> int:
        count = 0
        for input_combo in self.w.keys():
            count += self.n_outputs
        for input_combo in self.r.keys():
            count += len(self.r[input_combo])
        count += self.n_outputs  # w_prior
        count += 1               # r_prior
        return count

    # ---------- 保存 / 加载 ----------
    def save(self, filepath: str):
        data = {
            'n_features': self.n_features,
            'n_outputs': self.n_outputs,
            'n_values': self.n_values,
            'w_prior': self.w_prior.tolist(),
            'r_prior': self.r_prior,
            'w': {str(k): v.tolist() for k, v in self.w.items()},
            'r': {str(k): v.tolist() for k, v in self.r.items()},
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.w_prior = np.array(data['w_prior'])
        self.r_prior = data['r_prior']
        for k, v in data['w'].items():
            combo = eval(k)
            if combo in self.w:
                self.w[combo] = np.array(v)
        for k, v in data['r'].items():
            combo = eval(k)
            if combo in self.r:
                self.r[combo] = np.array(v)
