"""
DBN-EGWO-KELM 基线模型
Deep Belief Network + Enhanced Grey Wolf Optimizer + Kernel Extreme Learning Machine

Pipeline:
  1. DBN (无监督 CD-1 预训练) 提取特征
  2. EGWO 搜索 KELM 最优超参数 (C, gamma)
  3. KELM 使用 RBF 核进行解析分类

References:
  - DBN: Hinton (2006), "A fast learning algorithm for deep belief nets"
  - GWO: Mirjalili et al. (2014), "Grey Wolf Optimizer"
  - KELM: Huang et al. (2012), "Extreme learning machine for regression and multiclass classification"
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder


# ══════════════════════════════════════════════════════════════════════════════
# 1. RBM (Restricted Boltzmann Machine)
# ══════════════════════════════════════════════════════════════════════════════

class RBM(nn.Module):
    """
    受限玻尔兹曼机 (CD-1 对比散度训练)。
    PyTorch 实现，手动参数更新（带动量）。
    """

    def __init__(self, n_visible, n_hidden, learning_rate=0.01,
                 momentum=0.5, weight_decay=1e-4):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.vb = nn.Parameter(torch.zeros(n_visible))
        self.hb = nn.Parameter(torch.zeros(n_hidden))
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        # 动量缓冲（非 nn.Parameter，不参与梯度计算）
        self.W_mom = None
        self.vb_mom = None
        self.hb_mom = None

    def sample_hidden(self, v):
        """P(h=1|v) = sigmoid(v @ W + hb)"""
        p_h = torch.sigmoid(v @ self.W + self.hb)
        return p_h, torch.bernoulli(p_h)

    def sample_visible(self, h):
        """P(v|h) = sigmoid(h @ W^T + vb)，连续可见层使用概率"""
        p_v = torch.sigmoid(h @ self.W.t() + self.vb)
        return p_v

    def contrastive_divergence(self, v0, k=1):
        """
        CD-k 算法。
        返回重构误差 (MSE) 用于监控。
        """
        # 正相位
        p_h0, h0 = self.sample_hidden(v0)
        positive_grad = v0.t() @ p_h0

        # 负相位 (k 步 Gibbs 采样)
        h_k = h0
        for _ in range(k):
            v_k = self.sample_visible(h_k)
            p_hk, h_k = self.sample_hidden(v_k)
        negative_grad = v_k.t() @ p_hk

        batch_size = v0.size(0)

        # 初始化动量缓冲
        if self.W_mom is None:
            self.W_mom = torch.zeros_like(self.W.data)
            self.vb_mom = torch.zeros_like(self.vb.data)
            self.hb_mom = torch.zeros_like(self.hb.data)

        # 计算梯度
        dW = (positive_grad - negative_grad) / batch_size - self.weight_decay * self.W.data
        dvb = (v0 - v_k).mean(0)
        dhb = (p_h0 - p_hk).mean(0)

        # 应用动量更新
        self.W_mom = self.momentum * self.W_mom + self.lr * dW
        self.vb_mom = self.momentum * self.vb_mom + self.lr * dvb
        self.hb_mom = self.momentum * self.hb_mom + self.lr * dhb

        with torch.no_grad():
            self.W.add_(self.W_mom)
            self.vb.add_(self.vb_mom)
            self.hb.add_(self.hb_mom)

        recon_error = torch.mean((v0 - v_k) ** 2).item()
        return recon_error

    def transform(self, v):
        """前向传播：返回隐层概率（特征表示）"""
        with torch.no_grad():
            p_h = torch.sigmoid(v @ self.W + self.hb)
        return p_h


# ══════════════════════════════════════════════════════════════════════════════
# 2. DBN (Deep Belief Network)
# ══════════════════════════════════════════════════════════════════════════════

class DBN:
    """
    深度置信网络：RBM 逐层贪婪预训练，无监督特征提取。
    架构: input_dim → 256 → 128 → 64
    """

    def __init__(self, layer_sizes, learning_rate=0.01, momentum=0.5,
                 weight_decay=1e-4, device='cpu'):
        """
        Args:
            layer_sizes: 层大小列表，如 [122, 256, 128, 64]
            device: 'cpu' 或 'cuda'
        """
        self.layer_sizes = layer_sizes
        self.device = device
        self.rbms = []
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(
                n_visible=layer_sizes[i],
                n_hidden=layer_sizes[i + 1],
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            ).to(device)
            self.rbms.append(rbm)

    def pretrain(self, X, epochs=50, batch_size=128):
        """
        逐层贪婪预训练。
        X: numpy 数组 (n_samples, input_dim)
        """
        print(f"  DBN 预训练: 层结构 {self.layer_sizes}")
        current_input = torch.FloatTensor(X).to(self.device)

        for layer_idx, rbm in enumerate(self.rbms):
            print(f"    RBM 层 {layer_idx + 1}/{len(self.rbms)}: "
                  f"{self.layer_sizes[layer_idx]} -> {self.layer_sizes[layer_idx + 1]}")

            n_samples = current_input.size(0)
            for epoch in range(epochs):
                perm = torch.randperm(n_samples, device=self.device)
                epoch_error = 0.0
                n_batches = 0
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch = current_input[perm[start:end]]
                    error = rbm.contrastive_divergence(batch, k=1)
                    epoch_error += error
                    n_batches += 1

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    avg_error = epoch_error / n_batches
                    print(f"      Epoch {epoch + 1}/{epochs}, "
                          f"Reconstruction Error: {avg_error:.6f}")

            # 变换到下一层输入
            current_input = rbm.transform(current_input)

        print("  DBN 预训练完成")

    def transform(self, X):
        """
        将数据通过所有 RBM 层变换。
        X: numpy 数组 (n_samples, input_dim)
        返回: numpy 数组 (n_samples, last_hidden_size)
        """
        current = torch.FloatTensor(X).to(self.device)
        for rbm in self.rbms:
            current = rbm.transform(current)
        return current.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 3. KELM (Kernel Extreme Learning Machine)
# ══════════════════════════════════════════════════════════════════════════════

class KELM:
    """
    核极限学习机。
    解析解: beta = (K + I/C)^{-1} @ T
    核函数: RBF K(xi, xj) = exp(-gamma * ||xi - xj||^2)
    """

    def __init__(self, C=1.0, gamma=0.1):
        self.C = C
        self.gamma = gamma
        self.X_train = None
        self.beta = None
        self.encoder = None

    def _rbf_kernel(self, X1, X2):
        """计算 RBF 核矩阵"""
        dists_sq = cdist(X1, X2, metric='sqeuclidean')
        return np.exp(-self.gamma * dists_sq)

    def train(self, X_train, y_train):
        """
        解析训练。
        求解: beta = (K + I/C)^{-1} @ T
        """
        self.X_train = X_train.copy()
        n_samples = X_train.shape[0]

        # One-hot 编码
        self.encoder = OneHotEncoder(sparse_output=False, categories='auto')
        T = self.encoder.fit_transform(y_train.reshape(-1, 1))

        # 计算核矩阵
        K = self._rbf_kernel(X_train, X_train)

        # 解析求解 (np.linalg.solve 比显式求逆更稳定)
        A = K + np.eye(n_samples) / self.C
        self.beta = np.linalg.solve(A, T)

    def predict(self, X_test):
        """预测类别标签"""
        K_test = self._rbf_kernel(X_test, self.X_train)
        output = K_test @ self.beta
        return np.argmax(output, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 4. EGWO (Enhanced Grey Wolf Optimizer)
# ══════════════════════════════════════════════════════════════════════════════

class EGWO:
    """
    增强型灰狼优化器，用于搜索 KELM 最优超参数 (C, gamma)。

    增强策略:
      1. 非线性收敛因子: a = 2 * (1 - (t/T)^2)
      2. Levy 飞行扰动 alpha 狼
      3. 对立学习初始化
    """

    def __init__(self, n_wolves=20, max_iter=30,
                 C_range=(0.01, 1000), gamma_range=(0.0001, 10), seed=42):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        # 在 log10 空间搜索
        self.bounds = np.array([
            [np.log10(C_range[0]), np.log10(C_range[1])],
            [np.log10(gamma_range[0]), np.log10(gamma_range[1])],
        ])
        self.seed = seed
        self.dim = 2

    def _levy_flight(self, dim, beta=1.5):
        """Mantegna 算法生成 Levy 飞行步长"""
        from math import gamma as math_gamma
        sigma_u = (
            math_gamma(1 + beta) * np.sin(np.pi * beta / 2)
            / (math_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.randn(dim) * sigma_u
        v = np.random.randn(dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def _clip_position(self, position):
        """将狼的位置裁剪到搜索边界内"""
        clipped = position.copy()
        for d in range(self.dim):
            clipped[d] = np.clip(clipped[d], self.bounds[d, 0], self.bounds[d, 1])
        return clipped

    def _evaluate_fitness(self, position, X_train, y_train, X_val, y_val):
        """
        适应度函数：在给定 (C, gamma) 下训练 KELM，返回验证集 weighted F1。
        """
        C = 10 ** position[0]
        gamma_val = 10 ** position[1]

        try:
            kelm = KELM(C=C, gamma=gamma_val)
            kelm.train(X_train, y_train)
            y_pred = kelm.predict(X_val)
            fitness = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            fitness = 0.0

        return fitness

    def optimize(self, X_train, y_train, X_val, y_val):
        """
        运行 EGWO 搜索最优 (C, gamma)。

        返回: (best_C, best_gamma)
        """
        np.random.seed(self.seed)

        # 初始化种群 + 对立学习
        positions = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(self.n_wolves, self.dim)
        )
        opp_positions = self.bounds[:, 0] + self.bounds[:, 1] - positions

        for i in range(self.n_wolves):
            fit_orig = self._evaluate_fitness(positions[i], X_train, y_train, X_val, y_val)
            fit_opp = self._evaluate_fitness(opp_positions[i], X_train, y_train, X_val, y_val)
            if fit_opp > fit_orig:
                positions[i] = opp_positions[i]

        # 评估初始适应度
        fitness = np.array([
            self._evaluate_fitness(pos, X_train, y_train, X_val, y_val)
            for pos in positions
        ])

        # 识别 alpha, beta, delta
        sorted_idx = np.argsort(-fitness)
        alpha_pos = positions[sorted_idx[0]].copy()
        alpha_fit = fitness[sorted_idx[0]]
        beta_pos = positions[sorted_idx[1]].copy()
        beta_fit = fitness[sorted_idx[1]]
        delta_pos = positions[sorted_idx[2]].copy()
        delta_fit = fitness[sorted_idx[2]]

        print(f"    EGWO 初始化: alpha F1={alpha_fit:.4f}, "
              f"C={10 ** alpha_pos[0]:.4f}, gamma={10 ** alpha_pos[1]:.6f}")

        for t in range(self.max_iter):
            # 非线性收敛因子
            a = 2 * (1 - (t / self.max_iter) ** 2)

            for i in range(self.n_wolves):
                for d in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[d] - positions[i, d])
                    X1 = alpha_pos[d] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[d] - positions[i, d])
                    X2 = beta_pos[d] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[d] - positions[i, d])
                    X3 = delta_pos[d] - A3 * D_delta

                    positions[i, d] = (X1 + X2 + X3) / 3

                positions[i] = self._clip_position(positions[i])

            # Levy 飞行扰动 alpha 狼
            levy_step = self._levy_flight(self.dim) * 0.01 * (self.bounds[:, 1] - self.bounds[:, 0])
            candidate = self._clip_position(alpha_pos + levy_step)
            candidate_fit = self._evaluate_fitness(candidate, X_train, y_train, X_val, y_val)
            if candidate_fit > alpha_fit:
                alpha_pos = candidate.copy()
                alpha_fit = candidate_fit

            # 重新评估所有狼
            fitness = np.array([
                self._evaluate_fitness(pos, X_train, y_train, X_val, y_val)
                for pos in positions
            ])

            # 更新层级
            all_positions = np.vstack([positions, [alpha_pos], [beta_pos], [delta_pos]])
            all_fitness = np.concatenate([fitness, [alpha_fit, beta_fit, delta_fit]])
            sorted_idx = np.argsort(-all_fitness)

            alpha_pos = all_positions[sorted_idx[0]].copy()
            alpha_fit = all_fitness[sorted_idx[0]]
            beta_pos = all_positions[sorted_idx[1]].copy()
            beta_fit = all_fitness[sorted_idx[1]]
            delta_pos = all_positions[sorted_idx[2]].copy()
            delta_fit = all_fitness[sorted_idx[2]]

            if (t + 1) % 5 == 0 or t == 0:
                print(f"    EGWO Iter {t + 1}/{self.max_iter}: "
                      f"alpha F1={alpha_fit:.4f}, "
                      f"C={10 ** alpha_pos[0]:.4f}, gamma={10 ** alpha_pos[1]:.6f}")

        best_C = 10 ** alpha_pos[0]
        best_gamma = 10 ** alpha_pos[1]
        print(f"    EGWO 优化完成: best C={best_C:.4f}, gamma={best_gamma:.6f}, F1={alpha_fit:.4f}")
        return best_C, best_gamma


# ══════════════════════════════════════════════════════════════════════════════
# 5. DBN-EGWO-KELM 完整 Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class DBN_EGWO_KELM:
    """
    DBN-EGWO-KELM 完整管线。

    Pipeline:
        1. DBN 无监督预训练 → 特征提取
        2. EGWO 优化 KELM 超参数 (C, gamma)
        3. KELM 解析训练 → 分类
    """

    DEFAULT_CONFIG = {
        'dbn_hidden_sizes': [256, 128, 64],
        'dbn_epochs': 50,
        'dbn_batch_size': 128,
        'dbn_lr': 0.01,
        'dbn_momentum': 0.5,
        'dbn_weight_decay': 1e-4,
        'n_wolves': 20,
        'max_iter': 30,
        'C_range': (0.01, 1000),
        'gamma_range': (0.0001, 10),
        'kelm_max_train_samples': 5000,
        'egwo_val_ratio': 0.2,
    }

    def __init__(self, device='cpu', **kwargs):
        self.config = {**self.DEFAULT_CONFIG, **kwargs}
        self.device = device
        self.dbn = None
        self.kelm = None
        self.best_C = None
        self.best_gamma = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        完整训练管线。

        若未提供 X_val/y_val，则从 X_train 中划分 20% 作为 EGWO 验证集。
        """
        cfg = self.config
        n_features = X_train.shape[1]

        # ── Step 1: DBN 无监督特征提取 ──────────────────────────────────
        print("\n  [Step 1/3] DBN 无监督特征提取")
        layer_sizes = [n_features] + cfg['dbn_hidden_sizes']
        self.dbn = DBN(
            layer_sizes=layer_sizes,
            learning_rate=cfg['dbn_lr'],
            momentum=cfg['dbn_momentum'],
            weight_decay=cfg['dbn_weight_decay'],
            device=torch.device('cpu'),
        )
        self.dbn.pretrain(
            X_train,
            epochs=cfg['dbn_epochs'],
            batch_size=cfg['dbn_batch_size'],
        )

        X_train_dbn = self.dbn.transform(X_train)
        print(f"    DBN 特征维度: {n_features} -> {X_train_dbn.shape[1]}")

        # ── 准备 EGWO 验证集 ────────────────────────────────────────────
        if X_val is not None and y_val is not None:
            X_val_dbn = self.dbn.transform(X_val)
            y_train_use = y_train
            y_val_use = y_val
        else:
            X_train_dbn, X_val_dbn, y_train_use, y_val_use = train_test_split(
                X_train_dbn, y_train,
                test_size=cfg['egwo_val_ratio'],
                random_state=42,
                stratify=y_train,
            )

        # ── 子采样 KELM 训练集（内存限制） ──────────────────────────────
        max_samples = cfg['kelm_max_train_samples']
        if len(y_train_use) > max_samples:
            print(f"    KELM 采样: {len(y_train_use)} -> {max_samples} (内存限制)")
            indices = np.random.RandomState(42).choice(
                len(y_train_use), max_samples, replace=False
            )
            X_kelm_train = X_train_dbn[indices]
            y_kelm_train = y_train_use[indices]
        else:
            X_kelm_train = X_train_dbn
            y_kelm_train = y_train_use

        # 子采样验证集（加速 EGWO 评估）
        max_val = min(2000, len(y_val_use))
        if len(y_val_use) > max_val:
            val_indices = np.random.RandomState(42).choice(
                len(y_val_use), max_val, replace=False
            )
            X_kelm_val = X_val_dbn[val_indices]
            y_kelm_val = y_val_use[val_indices]
        else:
            X_kelm_val = X_val_dbn
            y_kelm_val = y_val_use

        # ── Step 2: EGWO 优化 KELM 超参数 ──────────────────────────────
        print("\n  [Step 2/3] EGWO 优化 KELM 超参数")
        egwo = EGWO(
            n_wolves=cfg['n_wolves'],
            max_iter=cfg['max_iter'],
            C_range=cfg['C_range'],
            gamma_range=cfg['gamma_range'],
        )
        self.best_C, self.best_gamma = egwo.optimize(
            X_kelm_train, y_kelm_train,
            X_kelm_val, y_kelm_val,
        )

        # ── Step 3: 训练最终 KELM ──────────────────────────────────────
        print(f"\n  [Step 3/3] KELM 训练 (C={self.best_C:.4f}, gamma={self.best_gamma:.6f})")
        self.kelm = KELM(C=self.best_C, gamma=self.best_gamma)
        self.kelm.train(X_kelm_train, y_kelm_train)

        print("  DBN-EGWO-KELM 训练完成")

    def predict(self, X_test):
        """
        预测标签。
        X_test: numpy 数组 (原始特征空间)
        """
        X_test_dbn = self.dbn.transform(X_test)
        return self.kelm.predict(X_test_dbn)
