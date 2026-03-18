import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import List, Dict, Tuple
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


def _xgb_tree_method():
    """Auto-detect best XGBoost tree_method: gpu_hist if CUDA available, else hist."""
    try:
        import torch as _t
        if _t.cuda.is_available():
            return 'hist'
    except Exception:
        pass
    return 'hist'

# 断点续训 checkpoint 目录
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _clone_model(model):
    """克隆 sklearn 模型（保留超参数，重置训练状态）"""
    from sklearn.base import clone
    return clone(model)


class SRFCNNBlock1D(nn.Module):
    """1D SRFCNN块: 分离-残差-融合卷积 (适配表格/序列数据)
    - 分离: kernel=3 和 kernel=5 两个并行 1D 卷积
    - 残差: 1x1 卷积跳跃连接
    - 融合: MaxPool1d + AvgPool1d 逐元素相加, 序列长度减半
    输出通道数 = 2 x filters
    """
    def __init__(self, in_channels, filters):
        super().__init__()
        self.conv_a = nn.Conv1d(in_channels, filters, kernel_size=3, padding=1)
        self.conv_b = nn.Conv1d(in_channels, filters, kernel_size=5, padding=2)
        self.conv_res = nn.Conv1d(in_channels, 2 * filters, kernel_size=1)
        self.gn = nn.GroupNorm(min(16, 2 * filters), 2 * filters)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # x: (N, C_in, L)
        x1 = F.relu(self.conv_a(x))           # (N, filters, L)
        x2 = F.relu(self.conv_b(x))           # (N, filters, L)
        merged = torch.cat([x1, x2], dim=1)   # (N, 2*filters, L)
        res = self.conv_res(x)                 # (N, 2*filters, L)
        merged = F.relu(self.gn(merged + res))
        return self.maxpool(merged) + self.avgpool(merged)  # (N, 2*filters, L//2)


class CNNSVM(nn.Module):
    """论文 FedPCNN 3层CNN结构 (32→64→128):
    Layer1: Conv1d(32) + GroupNorm + ReLU + MaxPool1d  — 浅层局部特征
    Layer2: Conv1d(64) + GroupNorm + ReLU + MaxPool1d  — 中层模式特征
    Layer3: Conv1d(128) + GroupNorm + ReLU + GlobalAvgPool1d — 深层语义特征
    Layer4: Dense(256) + ReLU + Dropout — 全连接
    Layer5: Dense(num_classes) — 输出层
    注: 原论文用 Conv2D，表格数据用 1D 卷积等效
    """
    FEATURE_DIM = 256  # CNN 输出特征维度（供外部引用）

    def __init__(self, input_channels=1, input_height=None, num_classes=2):
        super(CNNSVM, self).__init__()

        # Layer1: Conv1d(32) + GroupNorm + ReLU + MaxPool
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(min(16, 64), 64)

        # Layer2: Conv1d(64) + GroupNorm + ReLU + MaxPool
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(min(16, 128), 128)

        # Layer3: Conv1d(128) + GroupNorm + ReLU + GlobalAvgPool
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.GroupNorm(min(16, 256), 256)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Layer4: Dense + ReLU + Dropout
        self.fc1 = nn.Linear(256, self.FEATURE_DIM)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Layer5: 输出层
        self.fc2 = nn.Linear(self.FEATURE_DIM, num_classes)

    def forward(self, x, return_features=False):
        x = self.relu(self.bn1(self.conv1(x)))     # (N, 32, L)
        x = self.maxpool(x)                         # (N, 32, L//2)
        x = self.relu(self.bn2(self.conv2(x)))     # (N, 64, L//2)
        x = self.maxpool(x)                         # (N, 64, L//4)
        x = self.relu(self.bn3(self.conv3(x)))     # (N, 128, L//4)
        x = self.global_avg_pool(x)                 # (N, 128, 1)
        x = x.view(x.size(0), -1)                  # (N, 256)
        features = self.relu(self.fc1(x))            # (N, 256)
        x = self.dropout(features)
        x = self.fc2(x)                             # (N, num_classes)

        if return_features:
            return x, features
        return x

    def extract_features(self, x):
        """提取 fc1 特征（256 维）"""
        with torch.no_grad():
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.maxpool(x)
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.global_avg_pool(x).view(x.size(0), -1)  # (N, 256)
            features = self.relu(self.fc1(x))
        return features


class MultiClassHingeLoss(nn.Module):
    """多分类Hinge损失 (SVM损失)"""
    def __init__(self, C=1.0):
        super().__init__()
        self.C = C

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        correct_scores = outputs.gather(1, targets.view(-1, 1))
        margins = torch.clamp(1 - correct_scores + outputs, min=0)
        margins.scatter_(1, targets.view(-1, 1), 0)
        return self.C * margins.sum() / batch_size


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(pt) = -α(1-pt)^γ * log(pt)
    用于处理类别不平衡问题

    参数:
        alpha: 类别权重 (tensor 或 None)
        gamma: 聚焦参数，默认2.0
        reduction: 'mean' 或 'sum'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                   label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss




class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax Loss for long-tailed recognition.
    Adjusts logits by log class prior before cross-entropy.
    """
    def __init__(self, cls_counts, label_smoothing=0.0):
        super().__init__()
        prior = torch.tensor(cls_counts, dtype=torch.float32)
        self.register_buffer('log_prior', torch.log(prior / prior.sum() + 1e-12))
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        adjusted = logits + self.log_prior.unsqueeze(0)
        return F.cross_entropy(adjusted, target, label_smoothing=self.label_smoothing)

class CenterLoss(nn.Module):
    """Center Loss: 类内聚拢正则项

    对每个类别维护一个中心向量，拉近同类样本的 embedding。
    L_center = (1/2) * sum_i ||f_i - c_{y_i}||^2
    """
    def __init__(self, num_classes, feat_dim, device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # 中心向量不参与反向传播，手动更新
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device), requires_grad=False)

    def forward(self, features, labels):
        """计算 center loss，只对 batch 中存在的类别计算"""
        batch_centers = self.centers[labels]  # (N, feat_dim)
        loss = ((features - batch_centers) ** 2).sum(dim=1).mean() * 0.5
        return loss

    @torch.no_grad()
    def update_centers(self, features, labels, alpha=0.5):
        """动量更新中心向量: c_j = c_j - alpha * (c_j - mean(f_i for y_i=j))"""
        for c in labels.unique():
            mask = labels == c
            if mask.sum() > 0:
                class_mean = features[mask].mean(dim=0)
                self.centers[c] = (1 - alpha) * self.centers[c] + alpha * class_mean


class FedPCNN:
    def __init__(self,
                 num_devices: int = 10,
                 num_classes: int = 2,  # 改为二分类
                 input_shape: Tuple = (1, 20),  # 1D: (channels, n_features)
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 n_continuous: int = None):
        """
        参数:
            num_devices: 设备总数K
            num_classes: 分类类别数
            input_shape: 输入数据形状 (channels, n_features)
            device: 计算设备
            n_continuous: 保留兼容，不再使用
        """
        self.num_devices = num_devices
        self.num_classes = num_classes
        self.device = device
        self.input_shape = input_shape

        # 初始化全局模型
        self.global_model = CNNSVM(
            input_channels=input_shape[0],
            num_classes=num_classes
        ).to(device)

        print(f"FedPCNN初始化完成")
        print(f"设备数: {num_devices}, 类别数: {num_classes}")
        print(f"  CNN 输入: (1, {input_shape[1]}) 1D")
        print(f"  特征维度: {CNNSVM.FEATURE_DIM}")
        print(f"计算设备: {device}")
    
    def preprocess_data(self, X, y):
        """数据预处理: 重构为 1D 序列 (N, 1, n_features)
        注意: 假设数据已经被外部归一化（data_preprocessing.py）
        """
        n_samples = X.shape[0]
        X = X.reshape(n_samples, 1, -1)
        return X, y
    
    def split_data_non_iid(self, X, y, alpha=0.5):
        """Non-IID数据划分 (Dirichlet分布)"""
        n_classes = len(np.unique(y))
        class_indices = [np.where(y == c)[0] for c in range(n_classes)]
        client_data = [[] for _ in range(self.num_devices)]

        for c in range(n_classes):
            indices = class_indices[c]
            np.random.shuffle(indices)
            proportions = np.random.dirichlet(np.repeat(alpha, self.num_devices))
            split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            splits = np.split(indices, split_points)

            for device_id, split in enumerate(splits):
                client_data[device_id].extend(split)

        local_datasets = []
        for device_id in range(self.num_devices):
            idx = np.array(client_data[device_id], dtype=int)
            np.random.shuffle(idx)
            local_datasets.append((X[idx], y[idx]))

        return local_datasets
    
    def local_train(self, global_weights, epochs=5, batch_size=32,
                   lr=0.01, mu=0.01, gamma=0.5, focal_gamma=1.0,
                   dataset=None, local_data=None, class_weights=None,
                   center_loss_fn=None, lambda_center=0.05):
        """局部训练 (FedProx + Focal Loss + Center Loss + 学习率调度 + γ-不精确解)

        γ-不精确解 (FedProx 论文 Section 3):
            在全局模型处计算初始近端目标梯度范数 G₀ = ||∇h_k(w^t)||，
            每个 epoch 结束时若 avg||∇h_k(w)|| ≤ γ·G₀ 则提前停止。
            γ 越大 → 越早停止（防止 client drift）；γ 越小 → 训练越充分。

        参数:
            gamma:        γ-不精确解参数，[0,1)，默认 0.5
            class_weights: 全局类别权重 tensor（non-IID 时由 train() 传入，避免本地统计偏差）
            center_loss_fn: CenterLoss 实例（可选），用于类内聚拢正则
            lambda_center:  Center Loss 权重系数

        返回:
            (local_weights, avg_loss): 局部模型权重和平均损失
        """
        # 支持预构建 dataset（快速路径）和原始数据（兼容路径）
        if dataset is not None:
            local_y = dataset.tensors[1].cpu().numpy()
        else:
            local_X, local_y = local_data
            dataset = TensorDataset(
                torch.FloatTensor(local_X).to(self.device),
                torch.LongTensor(local_y).to(self.device)
            )

        # 复用预创建的本地模型（避免每个客户端都 new + to(device)）
        if not hasattr(self, '_local_model'):
            self._local_model = CNNSVM(
                input_channels=self.input_shape[0],
                input_height=self.input_shape[1],
                num_classes=self.num_classes
            ).to(self.device)
        local_model = self._local_model
        local_model.load_state_dict(global_weights)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 计算类别权重处理不平衡
        # 优先使用传入的全局权重；non-IID 下本地统计会严重偏差导致模型崩溃
        if class_weights is not None:
            cw_tensor = class_weights
        else:
            local_counts = np.maximum(
                np.bincount(local_y, minlength=self.num_classes), 1
            ).astype(float)
            cw = 1.0 / local_counts
            cw = cw / cw.sum() * self.num_classes
            cw_tensor = torch.FloatTensor(cw).to(self.device)

        # 损失函数：focal_gamma>0 使用 Focal Loss；focal_gamma==0 退化为标准 CE
        if focal_gamma > 0:
            ls = 0.1 if self.num_classes > 5 else 0.05
            criterion = FocalLoss(alpha=cw_tensor, gamma=focal_gamma, label_smoothing=ls)
        else:
            criterion = nn.CrossEntropyLoss()

        # SGD 优化器（momentum 加速收敛，weight_decay 提供 L2 正则）
        wd = 1e-4 if self.num_classes > 5 else 5e-4  # 减轻L2正则 (FL已有FedProx近端正则)
        optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

        # 学习率调度器 - CosineAnnealingLR（eta_min=10%初始lr，避免本地训练后期学习率过低）
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr * 0.1
        )

        # 预先将全局权重移到 device 并 detach（梯度注入法只需 .data）
        global_weights_on_device = {k: v.to(self.device).detach() for k, v in global_weights.items()}

        # ── 计算初始近端目标梯度范数 G₀（在全局模型处，训练开始前一次性计算）──
        # 此时 local_model == global_model，近端项梯度为 0，G₀ = ||∇F_k(w^t)||
        local_model.zero_grad()
        init_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        init_bx, init_by = next(iter(init_loader))
        init_loss = criterion(local_model(init_bx), init_by)
        init_loss.backward()
        G0 = sum(
            torch.norm(p.grad) ** 2 for p in local_model.parameters() if p.grad is not None
        ) ** 0.5
        G0 = G0.detach()
        local_model.zero_grad()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            local_model.train()
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            batch_count = 0
            num_batches_in_epoch = len(dataloader)
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # 前向传播：同时获取 logits 和 256 维 embedding
                if center_loss_fn is not None:
                    outputs, features = local_model(batch_X, return_features=True)
                else:
                    outputs = local_model(batch_X)

                # Focal Loss
                base_loss = criterion(outputs, batch_y)

                # Center Loss：类内聚拢正则
                if center_loss_fn is not None:
                    c_loss = center_loss_fn(features, batch_y)
                    total = base_loss + lambda_center * c_loss
                    total.backward()
                    # 更新中心向量（不参与梯度计算）
                    center_loss_fn.update_centers(features.detach(), batch_y)
                else:
                    base_loss.backward()

                # FedProx 梯度直接注入：∇L_prox = μ(w - w_global)
                # 数学等价于 loss += (μ/2)||w - w_global||²，但完全绕过 autograd
                with torch.no_grad():
                    for name, param in local_model.named_parameters():
                        if param.grad is not None:
                            param.grad.add_(mu * (param.data - global_weights_on_device[name]))

                batch_count += 1

                # 仅在最后一个 batch 计算梯度范数（backward 之后、clip 之前）
                # 相比原始每 batch 都算，减少 (N-1)/N 的开销，同时保持判据准确
                if batch_count == num_batches_in_epoch and G0 > 0:
                    epoch_grad_norm = sum(
                        torch.norm(p.grad).item() ** 2 for p in local_model.parameters()
                        if p.grad is not None
                    ) ** 0.5

                torch.nn.utils.clip_grad_norm_(local_model.parameters(),
                                               max_norm=1.0)
                optimizer.step()

                epoch_loss += base_loss.item()
                num_batches += 1

            total_loss += epoch_loss
            scheduler.step()

            # ── γ-不精确解判断（epoch 末，用最后 batch 的未裁剪梯度范数）──
            if G0 > 0 and epoch_grad_norm <= gamma * G0.item():
                break

        avg_loss = total_loss / max(num_batches, 1)
        return local_model.state_dict(), avg_loss

    def aggregate(self, local_weights_list, local_sizes):
        """
        Step 4.3: 聚合局部模型
        标准 FedAvg 加权聚合
        local_sizes: 每个客户端的样本数量
        """
        aggregated_weights = {}
        total_samples = sum(local_sizes)

        for key in local_weights_list[0].keys():
            aggregated_weights[key] = sum(
                local_weights_list[i][key] * (local_sizes[i] / total_samples)
                for i in range(len(local_weights_list))
            )

        return aggregated_weights

    def aggregate_dynamic(self, local_weights_list, local_sizes, local_losses):
        """
        动态加权聚合: 权重 ∝ 样本数 × (1 / 局部损失)
        损失越小的客户端获得更高的权重

        参数:
            local_weights_list: 局部模型权重列表
            local_sizes: 每个客户端的样本数量
            local_losses: 每个客户端的平均损失
        """
        weights = []
        for size, loss in zip(local_sizes, local_losses):
            # 用 loss+0.3 代替 loss+1e-6，防止 loss≈0 的客户端权重爆炸
            # 典型 loss 范围 0.1-1.0，此时权重最大倍差约 4x，合理
            w = size * (1.0 / (loss + 0.3))
            weights.append(w)

        # 进一步裁剪：单个客户端权重不超过均值的 3 倍，防止极端客户端主导聚合
        weights = np.array(weights, dtype=np.float64)
        weights = np.clip(weights, 0.0, weights.mean() * 3.0)

        # 归一化权重
        total_weight = weights.sum()
        weights = weights / total_weight

        # 加权聚合
        aggregated_weights = {}
        for key in local_weights_list[0].keys():
            aggregated_weights[key] = sum(
                local_weights_list[i][key] * weights[i]
                for i in range(len(local_weights_list))
            )

        return aggregated_weights

    def train(self, X_train=None, y_train=None, client_data=None, global_rounds=50, local_epochs=5,
             client_fraction=0.5, batch_size=32, lr=0.01, mu=0.01, gamma=0.5, alpha=0.5,
             focal_gamma=1.0,
             X_val=None, y_val=None, eval_interval=5,
             pre_smote_class_weights=None,
             client_data_raw=None, smote_warmup_rounds=0,
             checkpoint_tag=None):
        """FedPCNN完整训练流程（纯FedProx + 标准FedAvg聚合）

        参数:
            gamma:        γ-不精确解参数（越大越早停止，防 drift；越小训练越充分）
            focal_gamma:  Focal Loss 的 γ 参数
            X_val / y_val: 验证集（可选），用于绘制验证损失曲线
            eval_interval: 每隔多少轮计算一次验证损失
            client_data_raw:     SMOTE之前的原始客户端数据（用于预热）
            smote_warmup_rounds: 预热轮数，前N轮使用原始数据，之后切换到SMOTE数据
            checkpoint_tag:      断点续训标识（如 'UNSW_noniid_multi'），非None时启用自动保存/恢复

        返回:
            (train_loss_history, val_loss_history): 训练损失和验证损失列表
        """
        # cuDNN benchmark: 自动选择最优卷积算法（首轮略慢，后续加速）
        torch.backends.cudnn.benchmark = True

        print("\n" + "="*60)
        print("开始FedPCNN训练")
        print("="*60)

        if client_data is not None:
            print("\n使用外部预划分的客户端数据...")
            local_datasets = []
            for X_local, y_local in client_data:
                X_local, y_local = self.preprocess_data(X_local, y_local)
                local_datasets.append((X_local, y_local))
        else:
            print("\nStep 1: 数据预处理...")
            X_train, y_train = self.preprocess_data(X_train, y_train)

            print("\nStep 2: Non-IID数据划分...")
            local_datasets = self.split_data_non_iid(X_train, y_train, alpha)

        # 预先将所有客户端数据转为 GPU Tensor，避免每轮重复转换
        print("\n预构建客户端数据集到设备...")
        client_tensors = []
        for X_local, y_local in local_datasets:
            tx = torch.FloatTensor(X_local).to(self.device)
            ty = torch.LongTensor(y_local).to(self.device)
            client_tensors.append(TensorDataset(tx, ty))

        # SMOTE 预热：构建原始数据 tensor（前 warmup 轮使用）
        client_tensors_raw = None
        if client_data_raw is not None and smote_warmup_rounds > 0:
            print(f"\n  SMOTE预热: 前{smote_warmup_rounds}轮使用原始数据，之后切换到Borderline-SMOTE数据")
            client_tensors_raw = []
            for X_local, y_local in client_data_raw:
                X_local, y_local = self.preprocess_data(X_local, y_local)
                tx = torch.FloatTensor(X_local).to(self.device)
                ty = torch.LongTensor(y_local).to(self.device)
                client_tensors_raw.append(TensorDataset(tx, ty))

        # 计算全局类别权重
        # 优先使用 pre_smote_class_weights（SMOTE前的真实分布），避免SMOTE后虚假分布误导Focal Loss
        # 若未提供则从当前（SMOTE后）数据统计（兼容旧接口）
        if pre_smote_class_weights is not None:
            global_class_weights = pre_smote_class_weights.to(self.device)
            print(f"  全局类别权重(SMOTE前): {dict(enumerate(pre_smote_class_weights.numpy().round(3).tolist()))}")
        else:
            all_labels = np.concatenate([ds.tensors[1].cpu().numpy() for ds in client_tensors])
            global_counts = np.maximum(np.bincount(all_labels, minlength=self.num_classes), 1).astype(float)
            global_cw = 1.0 / global_counts
            min_cw = global_cw.min()
            if self.num_classes <= 2:
                max_ratio = 3.0
            elif self.num_classes <= 5:
                max_ratio = 5.0
            else:
                max_ratio = 15.0
            global_cw = np.clip(global_cw, 0, min_cw * max_ratio)
            global_cw = global_cw / global_cw.sum() * self.num_classes
            global_class_weights = torch.FloatTensor(global_cw).to(self.device)
            print(f"  全局类别权重(SMOTE后,cap={max_ratio:.0f}x): {dict(enumerate(global_cw.round(3).tolist()))}")

        print(f"\nStep 3-4: 开始{global_rounds}轮全局训练...")
        print(f"  聚合方式: 标准FedAvg (样本数加权, p_k = n_k/n)")

        # 预处理验证集（一次性，避免每轮重复）
        val_tensor = None
        if X_val is not None and y_val is not None:
            X_val_proc, y_val_proc = self.preprocess_data(X_val, y_val)
            vx = torch.FloatTensor(X_val_proc).to(self.device)
            vy = torch.LongTensor(y_val_proc).to(self.device)
            val_tensor = TensorDataset(vx, vy)
            print(f"  验证集: {len(y_val)} 样本，每 {eval_interval} 轮评估一次")

        train_loss_history = []
        val_loss_history = []

        # Center Loss：类内聚拢正则（多分类时启用）
        center_loss_fn = None
        lambda_center = 0.05
        if self.num_classes > 2:
            center_loss_fn = CenterLoss(self.num_classes, CNNSVM.FEATURE_DIM, device=self.device)
            print(f"  Center Loss: ON (lambda={lambda_center})")

        # 早停机制：验证损失连续 patience 次评估未改善则停止
        # patience=5, eval_interval=5 → 5×5=25轮无改善即停（50轮内可触发）
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        best_weights = None

        # 构建评估用损失函数（与训练一致的 FocalLoss，确保 train/val loss 量纲统一）
        if focal_gamma > 0:
            ls = 0.1 if self.num_classes > 5 else 0.05
            eval_criterion = FocalLoss(alpha=global_class_weights, gamma=focal_gamma,
                                       label_smoothing=ls, reduction='mean')
        else:
            eval_criterion = nn.CrossEntropyLoss()

        # 全局 lr 余弦衰减
        lr_init = lr
        print(f"  全局lr衰减: 余弦退火 {lr_init} → {lr_init * 0.1:.5f}")

        # ── 断点续训: 尝试加载 checkpoint ──
        start_round = 0
        ckpt_path = None
        if checkpoint_tag:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'{checkpoint_tag}_fl.pt')
            if os.path.exists(ckpt_path):
                print(f"\n  发现 checkpoint: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.global_model.load_state_dict(ckpt['model_state'])
                start_round = ckpt['round'] + 1
                best_val_loss = ckpt['best_val_loss']
                patience_counter = ckpt['patience_counter']
                train_loss_history = ckpt['train_loss_history']
                val_loss_history = ckpt['val_loss_history']
                if ckpt['best_weights'] is not None:
                    best_weights = ckpt['best_weights']
                # 恢复随机状态（确保客户端选择一致性）
                if 'rng_state' in ckpt:
                    np.random.set_state(ckpt['rng_state'])
                    rng_state = ckpt['torch_rng_state']
                    if not isinstance(rng_state, torch.ByteTensor):
                        rng_state = rng_state.cpu().byte()
                    torch.set_rng_state(rng_state)
                print(f"  从 Epoch {start_round + 1}/{global_rounds} 继续训练 "
                      f"(best_val_loss={best_val_loss:.4f}, patience={patience_counter}/{patience})")
            else:
                print(f"\n  无 checkpoint，从头训练 (将保存到 {ckpt_path})")

        # 全局进度条（最后一行动态更新）
        global_bar = tqdm(total=global_rounds, initial=start_round, desc="Global", unit="epoch",
                          leave=True, dynamic_ncols=True)

        round_idx = max(start_round - 1, 0)  # default if loop is skipped
        for round_idx in range(start_round, global_rounds):
            # 全局 lr 余弦衰减
            lr_round = lr_init * 0.5 * (1 + math.cos(math.pi * round_idx / global_rounds))
            lr_round = max(lr_round, lr_init * 0.1)
            num_selected = max(int(client_fraction * self.num_devices), 1)
            selected_devices = np.random.choice(self.num_devices, num_selected, replace=False)

            global_weights = self.global_model.state_dict()

            local_weights_list = []
            local_sizes = []
            local_losses = []

            # SMOTE 预热：前 warmup 轮用原始数据，之后切换到 SMOTE 数据
            use_raw = (client_tensors_raw is not None and round_idx < smote_warmup_rounds)
            active_tensors = client_tensors_raw if use_raw else client_tensors
            if use_raw and round_idx == 0:
                print("  [预热阶段] 使用原始数据训练")
            elif not use_raw and client_tensors_raw is not None and round_idx == smote_warmup_rounds:
                print("  [预热结束] 切换到Borderline-SMOTE数据")

            n_clients = len(selected_devices)
            # 客户端训练：用 \r 在同一行刷新
            for ci, device_id in enumerate(selected_devices):
                dataset = active_tensors[device_id]
                local_y = dataset.tensors[1].cpu().numpy()

                if len(np.unique(local_y)) < 2:
                    continue

                local_weights, avg_loss = self.local_train(
                    dataset=dataset,
                    global_weights=global_weights,
                    epochs=local_epochs,
                    batch_size=batch_size,
                    lr=lr_round,
                    mu=mu,
                    gamma=gamma,
                    focal_gamma=focal_gamma,
                    class_weights=global_class_weights,
                    center_loss_fn=center_loss_fn,
                    lambda_center=lambda_center,
                )
                local_weights_list.append(local_weights)
                local_sizes.append(len(local_y))
                local_losses.append(avg_loss)

            round_avg_loss = float(np.mean(local_losses)) if local_losses else 0.0
            train_loss_history.append(round_avg_loss)

            if not local_weights_list:
                global_bar.update(1)
                continue

            # 标准 FedAvg 聚合: p_k = n_k / n（论文原文定义）
            aggregated_weights = self.aggregate(local_weights_list, local_sizes)

            self.global_model.load_state_dict(aggregated_weights)

            # ── 每轮评估 train_acc / train_loss / val_loss / val_acc ──
            # 使用与训练一致的 eval_criterion（FocalLoss），确保量纲统一
            self.global_model.eval()
            with torch.no_grad():
                t_loss_sum, t_correct, t_total = 0.0, 0, 0
                for did in selected_devices:
                    ds = client_tensors[did]
                    dl = DataLoader(ds, batch_size=256, shuffle=False)
                    for tbx, tby in dl:
                        t_out = self.global_model(tbx)
                        t_loss_sum += eval_criterion(t_out, tby).item() * len(tby)
                        t_correct += (t_out.argmax(1) == tby).sum().item()
                        t_total += len(tby)
                train_acc = 100.0 * t_correct / max(t_total, 1)
                train_loss_eval = t_loss_sum / max(t_total, 1)

                val_loss_cur, val_acc_cur = 0.0, 0.0
                if val_tensor is not None:
                    vl = DataLoader(val_tensor, batch_size=256, shuffle=False)
                    v_loss, v_correct, v_total = 0.0, 0, 0
                    for vbx, vby in vl:
                        out = self.global_model(vbx)
                        v_loss += eval_criterion(out, vby).item() * len(vby)
                        v_correct += (out.argmax(1) == vby).sum().item()
                        v_total += len(vby)
                    val_loss_cur = v_loss / max(v_total, 1)
                    val_acc_cur = 100.0 * v_correct / max(v_total, 1)

            val_loss_history.append(val_loss_cur)

            # 打印每轮指标
            marker = " *" if val_loss_cur < best_val_loss else ""
            global_bar.clear()
            print(f"  Epoch {round_idx+1:3d}/{global_rounds} | "
                  f"t_loss={train_loss_eval:.4f} t_acc={train_acc:.1f}% | "
                  f"v_loss={val_loss_cur:.4f} v_acc={val_acc_cur:.1f}%{marker}")
            global_bar.set_postfix_str(
                f"t_loss={train_loss_eval:.4f} t_acc={train_acc:.1f}% "
                f"v_loss={val_loss_cur:.4f} v_acc={val_acc_cur:.1f}%{marker}")
            global_bar.update(1)
            global_bar.refresh()

            # 早停判断（每 eval_interval 轮检查一次）
            if val_tensor is not None and (round_idx + 1) % eval_interval == 0:
                if val_loss_cur < best_val_loss:
                    best_val_loss = val_loss_cur
                    patience_counter = 0
                    best_weights = deepcopy(self.global_model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        tqdm.write(f"\n  早停 @ Epoch {round_idx+1}: "
                                   f"验证损失连续 {patience} 次评估未改善 (best={best_val_loss:.4f})")
                        self.global_model.load_state_dict(best_weights)
                        break

            # ── 断点续训: 每 eval_interval 轮保存 checkpoint ──
            if ckpt_path and (round_idx + 1) % eval_interval == 0:
                torch.save({
                    'round': round_idx,
                    'model_state': self.global_model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'best_weights': best_weights,
                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    'rng_state': np.random.get_state(),
                    'torch_rng_state': torch.get_rng_state(),
                }, ckpt_path)

        global_bar.close()

        # 训练结束后恢复验证集上最优的权重
        if best_weights is not None:
            self.global_model.load_state_dict(best_weights)

        # 全局训练总结
        final_epoch = round_idx + 1
        print(f"\n{'='*60}")
        print(f"  FedPCNN 训练完成  |  总轮次: {final_epoch}/{global_rounds}")
        print(f"  最终 t_loss={train_loss_eval:.4f} t_acc={train_acc:.1f}%  "
              f"v_loss={val_loss_history[-1]:.4f} v_acc={val_acc_cur:.1f}%")
        if best_weights is not None:
            print(f"  最优 v_loss={best_val_loss:.4f} (已恢复该轮权重)")
        print(f"{'='*60}")

        # 训练完成，保留 checkpoint（由调用方统一清理，避免后续阶段失败时需重训）

        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history
        return train_loss_history, val_loss_history


    def calibrate_thresholds(self, X_val, y_val, batch_size=256):
        """在验证集上网格搜索最优logit偏置，最大化macro-F1（同时保证Normal召回不大幅下降）。

        原理：模型从不预测某些稀有类是因为其logit始终低于多数类。
        通过为稀有类logit加偏置，将决策边界移向这些类，无需重新训练。

        动态识别少数类：统计验证集各类别样本数，低于平均值的类为少数类。
        当少数类≤3时做全组合网格搜索；>3时退化为逐类独立搜索（避免指数爆炸）。

        参数:
            X_val / y_val: 验证集（归一化后，与训练集相同预处理）
            normal_recall_floor: Normal类召回率下限（确保不牺牲太多正常流量检测）

        返回:
            logit_bias (torch.Tensor, shape=num_classes): 各类logit偏置
        """
        from sklearn.metrics import f1_score as skf1

        print("\n校准推理阈值（验证集网格搜索）...")
        X_val_proc, y_val_np = self.preprocess_data(X_val, y_val)
        ds = TensorDataset(
            torch.FloatTensor(X_val_proc).to(self.device),
            torch.LongTensor(y_val_np).to(self.device)
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        self.global_model.eval()
        logit_list, label_list = [], []
        with torch.no_grad():
            for bx, by in loader:
                logit_list.append(self.global_model(bx).cpu())
                label_list.append(by.cpu())
        all_logits = torch.cat(logit_list)   # (N, C)
        all_labels = torch.cat(label_list).numpy()

        normal_mask = (all_labels == 0)

        # 自适应 normal_recall_floor：基于当前模型实际 Normal Recall 动态计算
        # 固定 floor=0.75 在多分类中经常高于 CNN 实际 Normal Recall，导致所有调整被否决
        baseline_preds = all_logits.argmax(1).numpy()
        baseline_normal_recall = (baseline_preds[normal_mask] == 0).mean() if normal_mask.any() else 1.0
        normal_recall_floor = baseline_normal_recall * 0.85  # 允许下降15%
        print(f"  当前Normal Recall={baseline_normal_recall*100:.1f}%, 自适应floor={normal_recall_floor*100:.1f}%")

        # 动态识别少数类：样本数低于各类平均值且不是class 0 (Normal)
        # 二分类特殊处理：始终对 Class 1 (攻击类) 搜索最优偏置
        class_counts = np.bincount(all_labels, minlength=self.num_classes)
        avg_count = class_counts.mean()
        if self.num_classes == 2:
            minority_classes = [1]
        else:
            minority_classes = [c for c in range(1, self.num_classes) if class_counts[c] < avg_count]
        print(f"  少数类: {minority_classes} (样本数<平均值{avg_count:.0f})")

        best_bias = torch.zeros(self.num_classes)
        best_f1 = -1.0
        best_result = {}

        bias_candidates = [i * 0.5 for i in range(11)]  # 0, 0.5, …, 5.0

        if len(minority_classes) <= 3:
            # 少数类≤3：全组合网格搜索
            from itertools import product as iproduct
            combos = list(iproduct(bias_candidates, repeat=len(minority_classes)))

            for combo in combos:
                bias = torch.zeros(self.num_classes)
                for cls_idx, cls_id in enumerate(minority_classes):
                    bias[cls_id] = combo[cls_idx]

                preds = (all_logits + bias).argmax(1).numpy()

                if normal_mask.any():
                    normal_recall = (preds[normal_mask] == 0).mean()
                    if normal_recall < normal_recall_floor:
                        continue

                f1 = skf1(all_labels, preds, average='macro', zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_bias = bias.clone()
                    per_cls = [
                        (preds[all_labels == c] == c).mean() if (all_labels == c).any() else 0.0
                        for c in range(self.num_classes)
                    ]
                    best_result = {'f1': f1, 'per_cls_recall': per_cls}
        else:
            # 少数类>3：逐类独立搜索（贪心），避免 11^N 指数爆炸
            current_bias = torch.zeros(self.num_classes)
            for cls_id in minority_classes:
                local_best_f1 = -1.0
                local_best_val = 0.0
                for val in bias_candidates:
                    bias = current_bias.clone()
                    bias[cls_id] = val
                    preds = (all_logits + bias).argmax(1).numpy()

                    if normal_mask.any():
                        normal_recall = (preds[normal_mask] == 0).mean()
                        if normal_recall < normal_recall_floor:
                            continue

                    f1 = skf1(all_labels, preds, average='macro', zero_division=0)
                    if f1 > local_best_f1:
                        local_best_f1 = f1
                        local_best_val = val
                current_bias[cls_id] = local_best_val

            best_bias = current_bias
            preds = (all_logits + best_bias).argmax(1).numpy()
            best_f1 = skf1(all_labels, preds, average='macro', zero_division=0)
            per_cls = [
                (preds[all_labels == c] == c).mean() if (all_labels == c).any() else 0.0
                for c in range(self.num_classes)
            ]
            best_result = {'f1': best_f1, 'per_cls_recall': per_cls}

        bias_str = {c: f"{best_bias[c].item():.1f}" for c in minority_classes}
        print(f"  最优偏置: {bias_str}")
        print(f"  验证集macro-F1(校准后): {best_result.get('f1', 0):.4f}")
        if 'per_cls_recall' in best_result:
            for c, rec in enumerate(best_result['per_cls_recall']):
                print(f"    类别 {c:2d}: {rec*100:.1f}%")
        return best_bias

    def classifier_retrain(self, X_train, y_train, epochs=10, batch_size=256, lr=0.01):
        """cRT: 冻结CNN backbone，用类别平衡采样重训分类器fc2

        原理: 联邦训练后CNN特征提取器已学到合理表示，但fc2分类器偏向多数类。
        冻结backbone，用平衡采样重训fc2可直接修正分类偏差，不影响fc1特征（SVM分支不受影响）。
        """
        from torch.utils.data import WeightedRandomSampler

        print("\n分类器重训练")
        X_proc, y_proc = self.preprocess_data(X_train, y_train)

        # 类别平衡采样器: 每个样本的采样权重 = 1/该类样本数
        class_counts = np.bincount(y_proc, minlength=self.num_classes)
        sample_weights = 1.0 / np.maximum(class_counts[y_proc], 1)
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(y_proc),
            replacement=True
        )

        tx = torch.FloatTensor(X_proc).to(self.device)
        ty = torch.LongTensor(y_proc).to(self.device)
        dataset = TensorDataset(tx, ty)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        # 冻结所有参数，只解冻 fc2
        for param in self.global_model.parameters():
            param.requires_grad = False
        for param in self.global_model.fc2.parameters():
            param.requires_grad = True

        optimizer = optim.SGD(self.global_model.fc2.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        self.global_model.train()
        for epoch in range(epochs):
            epoch_loss, n_batches = 0.0, 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.global_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  cRT Epoch {epoch+1}/{epochs}: loss={epoch_loss/n_batches:.4f}")

        # 恢复所有参数的 requires_grad
        for param in self.global_model.parameters():
            param.requires_grad = True

        print("  cRT 完成")

    def evaluate(self, X_test, y_test, batch_size=512, logit_bias=None):
        """评估模型性能

        参数:
            logit_bias: 可选，shape=(num_classes,) 的Tensor，
                        对各类logit加偏置（用于calibrate_thresholds后的校准推理）
        """
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

        print("\n评估模型性能...")
        X_test, y_test = self.preprocess_data(X_test, y_test)

        dataset = TensorDataset(
            torch.FloatTensor(X_test).to(self.device),
            torch.LongTensor(y_test).to(self.device)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.global_model.eval()
        all_preds, all_labels = [], []

        # 将 logit_bias 提前移到设备
        bias_tensor = None
        if logit_bias is not None:
            bias_tensor = logit_bias.to(self.device)
            print(f"  使用logit偏置校准: {[f'{v:.1f}' for v in logit_bias.tolist()]}")

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                outputs = self.global_model(batch_X)
                if bias_tensor is not None:
                    outputs = outputs + bias_tensor
                predicted = outputs.argmax(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100

        # macro 平均（所有类别等权，反映少数类真实表现）
        macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

        # 打印逐类别 Recall，便于定位稀有类（R2L/U2R）学习效果
        per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        print(f"\n  逐类别 Recall:")
        for cls_idx, rec in enumerate(per_class_recall):
            print(f"    类别 {cls_idx}: {rec*100:.1f}%")
        print(f"\n  Macro平均: Precision={macro_precision:.2f}%, Recall={macro_recall:.2f}%, F1={macro_f1:.2f}%")

        # FAR 通用公式（class 0 = Normal，支持二分类和多分类）
        # FPR = Normal 被预测为任意攻击的比例
        # FNR = 攻击被预测为 Normal 的比例
        cm = confusion_matrix(all_labels, all_preds)
        normal_total = cm[0, :].sum()
        attack_total = cm[1:, :].sum()
        FPR = cm[0, 1:].sum() / normal_total if normal_total > 0 else 0.0
        FNR = cm[1:, 0].sum() / attack_total if attack_total > 0 else 0.0
        FAR = (FPR + FNR) / 2 * 100

        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Macro-Precision': macro_precision,
            'Macro-Recall': macro_recall,
            'Macro-F1': macro_f1,
            'FAR': FAR
        }

        return metrics

    def _extract_features_batch(self, X, y, batch_size=256):
        """批量提取 CNN 特征，并拼接原始特征

        参数:
            X, y: 原始数据（未预处理，含全部特征列）
        返回:
            features (np.ndarray): (N, cnn_dim + raw_dim) 特征矩阵
            labels (np.ndarray): (N,) 标签
        """
        X_proc, y_proc = self.preprocess_data(X, y)
        ds = TensorDataset(
            torch.FloatTensor(X_proc).to(self.device),
            torch.LongTensor(y_proc).to(self.device)
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        self.global_model.eval()
        feat_list, label_list = [], []
        with torch.no_grad():
            for bx, by in loader:
                feats = self.global_model.extract_features(bx)
                feat_list.append(feats.cpu().numpy())
                label_list.append(by.cpu().numpy())

        cnn_features = np.concatenate(feat_list)
        # 拼接原始特征: XGBoost 同时利用 CNN 特征 + 原始特征
        combined = np.concatenate([cnn_features, X], axis=1)
        return combined, np.concatenate(label_list)

    def train_svm(self, X_train, y_train, C=1.0, kernel='rbf'):
        """用 CNN 特征训练 Stacking 集成分类器

        论文架构: CNN 特征 → RF + KNN + XGBoost (基学习器) → Meta-XGBoost (元学习器)
        使用 5折 out-of-fold 策略生成元特征，避免过拟合。

        参数:
            X_train, y_train: 训练数据（原始特征，未预处理）
            C, kernel: 保留接口兼容
        返回:
            训练好的 Stacking 分类器 (self.svm_classifier 为 meta 模型)
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from xgboost import XGBClassifier
        from sklearn.utils.class_weight import compute_sample_weight

        print("\n训练 Stacking 集成分类器 (RF + KNN + XGBoost → Meta-XGBoost)...")

        # 提取 CNN 特征
        features, labels = self._extract_features_batch(X_train, y_train)
        print(f"  特征矩阵: {features.shape}")

        # 标准化特征
        self.svm_scaler = StandardScaler()
        features_scaled = self.svm_scaler.fit_transform(features)

        n_classes = self.num_classes
        n_samples = len(labels)
        sample_weights = compute_sample_weight('balanced', labels)

        # ── 定义基学习器 ──
        base_learners = {
            'RF': RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=5,
                class_weight='balanced', random_state=42, n_jobs=-1,
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=7, weights='distance', n_jobs=-1,
            ),
            'XGB': XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                objective='multi:softprob', num_class=n_classes,
                tree_method="hist", device="cuda" if __import__("torch").cuda.is_available() else "cpu", random_state=42, n_jobs=-1, verbosity=0,
            ),
        }

        # ── 5折 Out-of-Fold 生成元特征 ──
        print(f"  5折 Out-of-Fold 生成元特征...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # 每个基学习器输出 n_classes 维概率 → 元特征维度 = 3 * n_classes
        meta_features = np.zeros((n_samples, len(base_learners) * n_classes))

        for name_idx, (name, base_model) in enumerate(base_learners.items()):
            print(f"    基学习器 [{name}]...", end=" ")
            col_start = name_idx * n_classes
            col_end = col_start + n_classes

            for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(features_scaled, labels)):
                X_tr, X_va = features_scaled[tr_idx], features_scaled[va_idx]
                y_tr = labels[tr_idx]
                sw_tr = sample_weights[tr_idx]

                model_clone = _clone_model(base_model)
                if name == 'XGB':
                    model_clone.fit(X_tr, y_tr, sample_weight=sw_tr)
                elif name == 'RF':
                    model_clone.fit(X_tr, y_tr)  # RF 用 class_weight='balanced'
                else:
                    model_clone.fit(X_tr, y_tr)

                proba = model_clone.predict_proba(X_va)
                # 处理 KNN 等可能不输出全部类别的情况
                if proba.shape[1] < n_classes:
                    full_proba = np.zeros((len(X_va), n_classes))
                    full_proba[:, :proba.shape[1]] = proba
                    proba = full_proba
                meta_features[va_idx, col_start:col_end] = proba

            print("done")

        # ── 全量训练基学习器（用于预测阶段）──
        print(f"  全量训练基学习器...")
        self._base_learners = {}
        for name, base_model in base_learners.items():
            model_clone = _clone_model(base_model)
            if name == 'XGB':
                model_clone.fit(features_scaled, labels, sample_weight=sample_weights)
            else:
                model_clone.fit(features_scaled, labels)
            self._base_learners[name] = model_clone
            print(f"    [{name}] 全量训练完成")

        # ── 训练元学习器 (Meta-XGBoost) ──
        print(f"  训练 Meta-XGBoost (输入维度: {meta_features.shape[1]})...")
        self.svm_classifier = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            objective='multi:softprob', num_class=n_classes,
            tree_method="hist", device="cuda" if __import__("torch").cuda.is_available() else "cpu", random_state=42, n_jobs=-1, verbosity=0,
        )
        meta_weights = compute_sample_weight('balanced', labels)
        self.svm_classifier.fit(meta_features, labels, sample_weight=meta_weights)
        print(f"  Stacking 训练完成")
        return self.svm_classifier

    def train_svm_bohb(self, X_train, y_train, n_trials=30, checkpoint_tag=None):
        """Stacking + BOHB: 先训练基学习器生成元特征，再搜索 Meta-XGBoost 最优超参

        流程:
          1. 提取 CNN 特征
          2. 5折 OOF 训练 RF + KNN + XGBoost 基学习器 → 元特征
          3. Optuna 搜索 Meta-XGBoost 超参数 (3折CV on 元特征)
          4. 用最优参数全量训练 Meta-XGBoost

        参数:
            X_train, y_train: 训练数据（原始特征）
            n_trials: 搜索试验次数，默认 30
            checkpoint_tag: 断点续训标识
        """
        import optuna
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score
        from sklearn.utils.class_weight import compute_sample_weight
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from xgboost import XGBClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"\n{'='*60}")
        print(f"Stacking + BOHB 超参搜索 (目标: Meta-XGBoost Macro-F1)")
        print(f"{'='*60}")

        # ── 1. 提取 CNN 特征 ──
        features, labels = self._extract_features_batch(X_train, y_train)
        print(f"  特征矩阵: {features.shape}, 类别分布: {np.bincount(labels.astype(int))}")

        self.svm_scaler = StandardScaler()
        features_scaled = self.svm_scaler.fit_transform(features)

        n_classes = self.num_classes
        n_samples = len(labels)
        sample_weights = compute_sample_weight('balanced', labels)

        # ── 2. 基学习器 OOF 生成元特征 ──
        base_learners = {
            'RF': RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=5,
                class_weight='balanced', random_state=42, n_jobs=-1,
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=7, weights='distance', n_jobs=-1,
            ),
            'XGB': XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                objective='multi:softprob', num_class=n_classes,
                tree_method="hist", device="cuda" if __import__("torch").cuda.is_available() else "cpu", random_state=42, n_jobs=-1, verbosity=0,
            ),
        }

        print(f"  5折 Out-of-Fold 生成元特征...")
        skf_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((n_samples, len(base_learners) * n_classes))

        for name_idx, (name, base_model) in enumerate(base_learners.items()):
            print(f"    基学习器 [{name}]...", end=" ")
            col_start = name_idx * n_classes
            col_end = col_start + n_classes
            for tr_idx, va_idx in skf_oof.split(features_scaled, labels):
                X_tr, X_va = features_scaled[tr_idx], features_scaled[va_idx]
                y_tr = labels[tr_idx]
                sw_tr = sample_weights[tr_idx]
                model_clone = _clone_model(base_model)
                if name == 'XGB':
                    model_clone.fit(X_tr, y_tr, sample_weight=sw_tr)
                else:
                    model_clone.fit(X_tr, y_tr)
                proba = model_clone.predict_proba(X_va)
                if proba.shape[1] < n_classes:
                    full_proba = np.zeros((len(X_va), n_classes))
                    full_proba[:, :proba.shape[1]] = proba
                    proba = full_proba
                meta_features[va_idx, col_start:col_end] = proba
            print("done")

        # 全量训练基学习器（用于预测阶段）
        print(f"  全量训练基学习器...")
        self._base_learners = {}
        for name, base_model in base_learners.items():
            model_clone = _clone_model(base_model)
            if name == 'XGB':
                model_clone.fit(features_scaled, labels, sample_weight=sample_weights)
            else:
                model_clone.fit(features_scaled, labels)
            self._base_learners[name] = model_clone
            print(f"    [{name}] 全量训练完成")

        # ── 3. BOHB 搜索 Meta-XGBoost 超参 ──
        print(f"\n  BOHB 搜索 Meta-XGBoost ({n_trials} trials)...")
        num_classes = self.num_classes
        meta_dim = meta_features.shape[1]

        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'objective': 'multi:softprob',
                'num_class': num_classes,
                'tree_method': 'hist', 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0,
            }

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            fold_scores = []
            for train_idx, val_idx in skf.split(meta_features, labels):
                X_tr, X_va = meta_features[train_idx], meta_features[val_idx]
                y_tr, y_va = labels[train_idx], labels[val_idx]
                weights = compute_sample_weight('balanced', y_tr)
                model = XGBClassifier(**param)
                model.fit(X_tr, y_tr, sample_weight=weights,
                          eval_set=[(X_va, y_va)],
                          early_stopping_rounds=20, verbose=False)
                preds = model.predict(X_va)
                fold_scores.append(f1_score(y_va, preds, average='macro'))
            return np.mean(fold_scores)

        study_name = 'Stacking_Meta_BOHB'
        storage = None
        db_path = None
        if checkpoint_tag:
            db_path = os.path.join(CHECKPOINT_DIR, f'{checkpoint_tag}_bohb.db')
            storage = f'sqlite:///{db_path}'
            study_name = f'{checkpoint_tag}_bohb'

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

        completed = len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = max(0, n_trials - completed)
        if completed > 0:
            print(f"\n  断点续训: 已完成 {completed}/{n_trials} trials，剩余 {remaining}")
            if remaining == 0:
                print(f"  所有 trials 已完成，跳过搜索")
            else:
                print(f"  当前最佳: Macro-F1={study.best_value*100:.2f}%")

        import time
        t0 = time.time()
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, show_progress_bar=True)
        elapsed = time.time() - t0

        print(f"\n  搜索完成! 耗时 {elapsed/60:.1f} 分钟")
        print(f"  最佳 3折CV Macro-F1: {study.best_value*100:.2f}%")
        print(f"  最佳参数:")
        for k, v in study.best_params.items():
            fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"    {k}: {fmt}")

        # ── 4. 用最优参数全量训练 Meta-XGBoost ──
        best = study.best_params
        best.update({
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'tree_method': 'hist', 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
        })

        self.bohb_best_params = dict(study.best_params)
        self.bohb_best_score = study.best_value

        print(f"\n  用最优参数全量训练 Meta-XGBoost ({n_samples} 样本, 输入维度: {meta_dim})...")
        meta_weights = compute_sample_weight('balanced', labels)
        self.svm_classifier = XGBClassifier(**best)
        self.svm_classifier.fit(meta_features, labels, sample_weight=meta_weights)
        print(f"  Stacking + BOHB 训练完成")
        print(f"{'='*60}")

        # 清理 BOHB checkpoint
        if db_path and os.path.exists(db_path):
            del study
            import gc; gc.collect()
            import time
            for _retry in range(5):
                try:
                    os.remove(db_path)
                    print(f"  已清理 BOHB checkpoint: {db_path}")
                    break
                except PermissionError:
                    time.sleep(0.5)
            else:
                print(f"  警告: 无法删除 BOHB checkpoint (文件被占用): {db_path}")

        return self.svm_classifier

    def _build_meta_features(self, features_scaled):
        """用基学习器生成元特征（预测阶段使用）"""
        n_classes = self.num_classes
        meta_parts = []
        for name, model in self._base_learners.items():
            proba = model.predict_proba(features_scaled)
            if proba.shape[1] < n_classes:
                full_proba = np.zeros((len(features_scaled), n_classes))
                full_proba[:, :proba.shape[1]] = proba
                proba = full_proba
            meta_parts.append(proba)
        return np.hstack(meta_parts)

    def evaluate_with_svm(self, X_test, y_test, batch_size=256, normal_threshold=None):
        """用 CNN + Stacking 集成进行评估

        参数:
            normal_threshold: Normal 类概率门限。若设置，P(Normal) >= threshold 判为 Normal，
                              否则取攻击类中 argmax。用于单阶段降低 FAR。None 则用默认 argmax。
        """
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

        use_stacking = hasattr(self, '_base_learners') and self._base_learners
        classifier_name = "Stacking" if use_stacking else "XGBoost"
        print(f"\n评估 CNN + {classifier_name} 性能...")

        # 提取特征并标准化
        features, labels = self._extract_features_batch(X_test, y_test, batch_size)
        features_scaled = self.svm_scaler.transform(features)

        # 生成预测输入
        if use_stacking:
            pred_input = self._build_meta_features(features_scaled)
        else:
            pred_input = features_scaled

        # 预测（支持门限控制）
        if normal_threshold is not None and hasattr(self.svm_classifier, 'predict_proba'):
            proba = self.svm_classifier.predict_proba(pred_input)
            all_preds = np.where(
                proba[:, 0] >= normal_threshold,
                0,  # Normal
                proba[:, 1:].argmax(axis=1) + 1  # 攻击类 argmax (index 1-9)
            )
        else:
            raw_preds = self.svm_classifier.predict(pred_input)
            raw_preds = np.asarray(raw_preds)
            # XGBoost multi:softprob 二分类时 predict 可能返回概率矩阵
            if raw_preds.ndim == 2:
                all_preds = raw_preds.argmax(axis=1)
            else:
                all_preds = raw_preds

        all_preds = np.array(all_preds, dtype=np.int64).ravel()
        labels = np.array(labels, dtype=np.int64).ravel()
        accuracy = 100.0 * np.mean(all_preds == labels)
        precision = precision_score(labels, all_preds, average='weighted', zero_division=0) * 100
        recall = recall_score(labels, all_preds, average='weighted', zero_division=0) * 100
        f1 = f1_score(labels, all_preds, average='weighted', zero_division=0) * 100

        # macro 平均（所有类别等权）
        macro_precision = precision_score(labels, all_preds, average='macro', zero_division=0) * 100
        macro_recall = recall_score(labels, all_preds, average='macro', zero_division=0) * 100
        macro_f1 = f1_score(labels, all_preds, average='macro', zero_division=0) * 100

        per_class_recall = recall_score(labels, all_preds, average=None, zero_division=0)
        threshold_info = f", Normal门限={normal_threshold}" if normal_threshold is not None else ""
        print(f"\n  逐类别 Recall (CNN+XGBoost{threshold_info}):")
        for cls_idx, rec in enumerate(per_class_recall):
            print(f"    类别 {cls_idx}: {rec*100:.1f}%")
        print(f"\n  Macro平均: Precision={macro_precision:.2f}%, Recall={macro_recall:.2f}%, F1={macro_f1:.2f}%")

        cm = confusion_matrix(labels, all_preds)
        normal_total = cm[0, :].sum()
        attack_total = cm[1:, :].sum()
        FPR = cm[0, 1:].sum() / normal_total if normal_total > 0 else 0.0
        FNR = cm[1:, 0].sum() / attack_total if attack_total > 0 else 0.0
        FAR = (FPR + FNR) / 2 * 100

        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Macro-Precision': macro_precision,
            'Macro-Recall': macro_recall,
            'Macro-F1': macro_f1,
            'FAR': FAR
        }

        return metrics, all_preds, labels

    def search_normal_threshold(self, X_val, y_val, batch_size=256):
        """在验证集上搜索最优 Normal 门限

        目标：最大化 Macro-F1，同时 FAR 不超过单阶段基线水平
        搜索范围：0.3 ~ 0.7，步长 0.05
        """
        from sklearn.metrics import f1_score, confusion_matrix

        print("\n搜索最优 Normal 门限...")
        features, labels = self._extract_features_batch(X_val, y_val, batch_size)
        features_scaled = self.svm_scaler.transform(features)

        use_stacking = hasattr(self, '_base_learners') and self._base_learners
        if use_stacking:
            pred_input = self._build_meta_features(features_scaled)
        else:
            pred_input = features_scaled
        proba = self.svm_classifier.predict_proba(pred_input)

        best_threshold = None
        best_score = -1.0
        results = []

        for t in np.arange(0.30, 0.75, 0.05):
            preds = np.where(proba[:, 0] >= t, 0, proba[:, 1:].argmax(axis=1) + 1)
            macro_f1 = f1_score(labels, preds, average='macro', zero_division=0) * 100
            acc = 100 * (preds == labels).mean()
            cm = confusion_matrix(labels, preds)
            normal_total = cm[0, :].sum()
            attack_total = cm[1:, :].sum()
            fpr = cm[0, 1:].sum() / normal_total if normal_total > 0 else 0.0
            fnr = cm[1:, 0].sum() / attack_total if attack_total > 0 else 0.0
            far = (fpr + fnr) / 2 * 100
            results.append((t, acc, macro_f1, far))
            print(f"  threshold={t:.2f} → Acc={acc:.2f}%, Macro-F1={macro_f1:.2f}%, FAR={far:.2f}%")

            if macro_f1 > best_score:
                best_score = macro_f1
                best_threshold = t

        print(f"\n  最优门限: {best_threshold:.2f} (Macro-F1={best_score:.2f}%)")
        return best_threshold

    def predict_with_svm(self, X, batch_size=256):
        """CNN+Stacking 纯预测（无需标签，用于两阶段推理链）

        参数:
            X: 原始特征矩阵 (未预处理)
        返回:
            predictions: np.ndarray, shape=(N,) 预测类别
        """
        dummy_y = np.zeros(len(X), dtype=int)
        features, _ = self._extract_features_batch(X, dummy_y, batch_size)
        features_scaled = self.svm_scaler.transform(features)

        use_stacking = hasattr(self, '_base_learners') and self._base_learners
        if use_stacking:
            pred_input = self._build_meta_features(features_scaled)
        else:
            pred_input = features_scaled
        raw_preds = np.asarray(self.svm_classifier.predict(pred_input))
        if raw_preds.ndim == 2:
            return raw_preds.argmax(axis=1)
        return raw_preds


# 示例使用
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data_preprocessing import NSLKDDPreprocessor

    np.random.seed(42)
    torch.manual_seed(42)

    print("加载NSL-KDD数据集")
    preprocessor = NSLKDDPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.load_and_preprocess()

    n_classes = len(preprocessor.label_encoder.classes_)
    print(f"数据集: 训练{X_train.shape}, 测试{X_test.shape}, 类别{n_classes}")

    fedpcnn = FedPCNN(
        num_devices=10,
        num_classes=n_classes,
        input_shape=(1, 122)  # 1D: (channels, n_features)
    )

    fedpcnn.train(
        X_train=X_train,
        y_train=y_train,
        global_rounds=10,
        local_epochs=5,
        client_fraction=0.5,
        batch_size=32,
        lr=0.01,
        mu=2.0,   #原来0.01
        gamma=0.8,  # 原来0.5
        alpha=0.5
    )

    metrics = fedpcnn.evaluate(X_test, y_test)
