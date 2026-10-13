"""MAKER 推理器（MAKERInference）

按照论文 MAKER 框架实现概率推理：
    1. EAM 获取证据（由 LikelihoodCounter 完成，得到归一化似然 p_θli）
    2. 把证据投影到 ERPS（投影率 ω，Eq. 21/22/26）
    3. 用 MAKER 规则递归组合多条证据（Theorem 1 / Lemma 2.1）
    4. 组合先验证据（独立证据的 ER 规则，Corollary 1.4）
    5. 投影回 SOPS，得到 p_θ 与信任度 t = 1 - m̃_2Θ

核心公式：
    m_θli = w_θl · p_θli                      (Eq. 23)
    ω_li  = 1 / (Σ p_θ w_θ + 1 - r_li)       (Eq. 21)
    m̃_θli = ω_li · m_θli                      (Eq. 22)
    m̃_2Θli = ω_li · (1 - r_li)                (Eq. 26)
"""
import numpy as np
from typing import Dict, Tuple, List


class CompositeEvidence:
    """复合证据：表示一条在 ERPS 空间上的证据（含对状态的支持 + 对幂集的剩余支持）"""

    def __init__(self,
                 input_combo: Tuple[int, ...],
                 value_combo: Tuple[int, ...],
                 m_tilde_theta: np.ndarray,
                 m_tilde_2theta: float):
        self.input_combo = input_combo
        self.value_combo = value_combo
        self.m_tilde_theta_li = m_tilde_theta   # m̃_θ，形状 (n_outputs,)
        self.m_tilde_2theta_li = m_tilde_2theta  # m̃_2Θ，标量


class MAKERInference:

    def __init__(self, registry, n_outputs: int):
        self.registry = registry
        self.n_features = registry.n_features
        self.n_outputs = n_outputs
        self.n_values = registry.n_values_per_feature

    # ============ 对外推理接口 ============
    def infer(self, data_row, likelihood_counter, param_manager):
        """输入一条 0-based 数据行（前 n_features 列是特征，最后一列是占位标签），
        输出 (p_θ, trustworthiness)。"""
        features = data_row[:self.n_features].astype(int)
        evidence_list = [(l, int(features[l])) for l in range(self.n_features)]

        # 1. 递归组合所有数据证据
        current = self._recursive_combine(evidence_list, likelihood_counter, param_manager)

        # 2. 组合先验证据
        prior = self._build_prior_evidence(likelihood_counter, param_manager)
        final = self._combine_two_composites(current, prior)

        # 3. 投影回 SOPS
        m_tilde = final.m_tilde_theta_li
        m_tilde_2theta = final.m_tilde_2theta_li
        denom = np.sum(m_tilde)
        if denom > 0:
            p_theta = m_tilde / denom
        else:
            p_theta = np.ones(self.n_outputs) / self.n_outputs
        trustworthiness = 1.0 - m_tilde_2theta

        return p_theta, trustworthiness

    # ============ 递归组合 ============
    def _recursive_combine(self, evidence_list, likelihood_counter, param_manager) -> CompositeEvidence:
        if len(evidence_list) == 0:
            raise ValueError("证据列表不能为空")

        current = self._evidence_to_composite(evidence_list[0], likelihood_counter, param_manager)
        for idx in range(1, len(evidence_list)):
            current = self._combine_composite_with_evidence(
                current, evidence_list[idx], likelihood_counter, param_manager)
        return current

    # ============ 单证据投影：SIPS -> ERPS ============
    def _evidence_to_composite(self, evidence, likelihood_counter, param_manager) -> CompositeEvidence:
        l, i = evidence
        p_theta_li = np.array([
            likelihood_counter.get_likelihood(theta, (l,), (i,))
            for theta in range(self.n_outputs)
        ])
        w_theta_l = param_manager.get_w_vector(input_combo=(l,))
        r_li = param_manager.get_r(input_combo=(l,), value_combo=(i,))

        m_theta_li = w_theta_l * p_theta_li
        expected_weight = np.sum(p_theta_li * w_theta_l)
        omega = 1.0 / (expected_weight + 1.0 - r_li)

        m_tilde_theta = omega * m_theta_li
        m_tilde_2theta = omega * (1.0 - r_li)

        return CompositeEvidence((l,), (i,), m_tilde_theta, m_tilde_2theta)

    # ============ 递归 MAKER 规则：复合证据 + 新证据 ============
    def _combine_composite_with_evidence(self, composite, evidence, likelihood_counter, param_manager) -> CompositeEvidence:
        n_states = self.n_outputs
        l, i = evidence

        # 新证据的未投影概率质量 m_θli
        p_theta_li = np.array([
            likelihood_counter.get_likelihood(theta, (l,), (i,))
            for theta in range(n_states)
        ])
        w_theta_l = param_manager.get_w_vector(input_combo=(l,))
        m_theta_li = w_theta_l * p_theta_li
        r_li = param_manager.get_r(input_combo=(l,), value_combo=(i,))

        # 组合所需的三类相对量
        alpha_bar = self._compute_alpha_bar(
            composite.input_combo, composite.value_combo, (l,), (i,), likelihood_counter)
        omega_bar = self._compute_omega_bar(composite, evidence, likelihood_counter, param_manager)
        w_bar = self._compute_w_bar(composite, evidence, param_manager)

        # term1：一方有剩余支持时的贡献
        term1 = (1.0 - r_li) * composite.m_tilde_theta_li + composite.m_tilde_2theta_li * m_theta_li
        # term2：双方共同支持的联合贡献（含相对投影率/权重/依赖指数）
        term2 = omega_bar * w_bar * alpha_bar * composite.m_tilde_theta_li * m_theta_li

        m_hat_theta = term1 + term2
        m_hat_2theta = (1.0 - r_li) * composite.m_tilde_2theta_li

        denom = np.sum(m_hat_theta) + m_hat_2theta
        if denom > 0:
            m_tilde_theta = m_hat_theta / denom
            m_tilde_2theta = m_hat_2theta / denom
        else:
            m_tilde_theta = np.ones(n_states) / n_states
            m_tilde_2theta = 0.0

        return CompositeEvidence(
            composite.input_combo + (l,),
            composite.value_combo + (i,),
            m_tilde_theta, m_tilde_2theta)

    # ============ 独立证据组合（ER 规则，Corollary 1.4） ============
    def _combine_two_composites(self, comp1, comp2) -> CompositeEvidence:
        """组合两个已在 ERPS 空间的证据（假设相互独立）。
        m̂_θ  = m̃_2Θ₂·m̃_θ₁ + m̃_2Θ₁·m̃_θ₂ + m̃_θ₁·m̃_θ₂
        m̂_2Θ = m̃_2Θ₁·m̃_2Θ₂
        """
        m1 = comp1.m_tilde_theta_li
        m2 = comp2.m_tilde_theta_li
        m2t1 = comp1.m_tilde_2theta_li
        m2t2 = comp2.m_tilde_2theta_li

        m_hat = m2t2 * m1 + m2t1 * m2 + m1 * m2
        m_hat_2theta = m2t1 * m2t2

        denom = np.sum(m_hat) + m_hat_2theta
        if denom > 0:
            m_tilde = m_hat / denom
            m_tilde_2 = m_hat_2theta / denom
        else:
            m_tilde = np.ones(self.n_outputs) / self.n_outputs
            m_tilde_2 = 0.0

        return CompositeEvidence(
            comp1.input_combo + comp2.input_combo,
            comp1.value_combo + comp2.value_combo,
            m_tilde, m_tilde_2)

    # ============ 先验证据 ============
    def _build_prior_evidence(self, likelihood_counter, param_manager) -> CompositeEvidence:
        w_prior = param_manager.get_w_prior_vector()
        r_prior = param_manager.get_r_prior()
        p_prior = likelihood_counter.prior_distribution

        m_theta_prior = w_prior * p_prior
        expected_weight = np.sum(p_prior * w_prior)
        denom = expected_weight + 1.0 - r_prior
        omega_prior = 1.0 / denom if denom > 0 else 1.0

        m_tilde_theta = omega_prior * m_theta_prior
        m_tilde_2theta = omega_prior * (1.0 - r_prior)

        # 注意，这里用 (-1,) 标记先验证据
        return CompositeEvidence((-1,), (-1,), m_tilde_theta, m_tilde_2theta)

    # ============ 三类相对量 ============
    def _compute_alpha_bar(self, input_combo1, value_combo1, input_combo2, value_combo2, likelihood_counter) -> np.ndarray:
        """互依指数向量 ᾱ_θ = p_AB(θ) / (p_A(θ) · p_B(θ))"""
        n_states = self.n_outputs
        alpha_bar = np.zeros(n_states)
        combined_input = tuple(sorted(input_combo1 + input_combo2))

        feat_to_val = {}
        for feat, val in zip(input_combo1, value_combo1):
            feat_to_val[feat] = val
        for feat, val in zip(input_combo2, value_combo2):
            feat_to_val[feat] = val
        combined_value = tuple(feat_to_val[feat] for feat in combined_input)

        for theta in range(n_states):
            p_A = likelihood_counter.get_likelihood(theta, input_combo1, value_combo1)
            p_B = likelihood_counter.get_likelihood(theta, input_combo2, value_combo2)
            p_AB = likelihood_counter.get_likelihood(theta, combined_input, combined_value)
            alpha_bar[theta] = p_AB / (p_A * p_B) if (p_A > 0 and p_B > 0) else 1.0
        return alpha_bar

    def _compute_omega_bar(self, composite, evidence, likelihood_counter, param_manager) -> float:
        """相对投影率 ω̄ = ω_combined / (ω_composite · ω_evidence)"""
        l, i = evidence
        composite_input = composite.input_combo
        composite_value = composite.value_combo
        evidence_input = (l,)
        evidence_value = (i,)
        combined_input = composite_input + (l,)
        combined_value = composite_value + (i,)

        omega_composite = self._compute_omega(composite_input, composite_value, likelihood_counter, param_manager)
        omega_evidence = self._compute_omega(evidence_input, evidence_value, likelihood_counter, param_manager)
        omega_combined = self._compute_omega(combined_input, combined_value, likelihood_counter, param_manager)

        denom = omega_composite * omega_evidence
        return omega_combined / denom if denom > 0 else 1.0

    def _compute_w_bar(self, composite, evidence, param_manager) -> np.ndarray:
        """相对状态权重 w̄_θ = w_combined_θ / (w_composite_θ · w_evidence_θ)"""
        l, i = evidence
        composite_input = composite.input_combo
        evidence_input = (l,)
        combined_input = composite_input + (l,)

        w_composite = param_manager.get_w_vector(composite_input)
        w_evidence = param_manager.get_w_vector(evidence_input)
        w_combined = param_manager.get_w_vector(combined_input)

        w_bar = np.ones(self.n_outputs)
        for theta in range(self.n_outputs):
            denom = w_composite[theta] * w_evidence[theta]
            if denom > 0:
                w_bar[theta] = w_combined[theta] / denom
        return w_bar
    # 大肥鱼整理后的代码真规整啊！
    def _compute_omega(self, input_combo, value_combo, likelihood_counter, param_manager) -> float:
        """投影率 ω = 1 / (Σ p_θ w_θ + 1 - r)"""
        p_theta = np.array([
            likelihood_counter.get_likelihood(theta, input_combo, value_combo)
            for theta in range(self.n_outputs)
        ])
        w_theta = param_manager.get_w_vector(input_combo)
        r = param_manager.get_r(input_combo, value_combo)
        expected_weight = np.sum(p_theta * w_theta)
        return 1.0 / (expected_weight + 1.0 - r)

