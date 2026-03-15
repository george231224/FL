import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
import warnings

warnings.filterwarnings("ignore")


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(pt) = -α(1-pt)^γ * log(pt)
    用于处理类别不平衡问题
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
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


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256,
                 num_layers=2, num_classes=2, dropout=0.3):  # 改为二分类
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ================================
# 分段式联邦学习
# ================================
class SegmentedFederatedLearning:

    def __init__(self, num_devices=20, num_classes=2,  # 改为二分类
                 input_size=41, sequence_length=1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.num_devices = num_devices
        self.num_classes = num_classes
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.device = device

        self.scaler = None
        self.client_data = None

        self.global_model = LSTMModel(
            input_size=input_size,
            num_classes=num_classes
        ).to(device)

        print("分段式联邦学习初始化完成")
        print(f"  设备数: {num_devices}, 类别数: {num_classes}")
        print(f"  使用设备: {device}")

    # ================================
    # 数据预处理
    # ================================
    def preprocess_data(self, X, y, fit_scaler=False):
        """数据预处理：MinMax归一化 + 时序窗口"""
        if fit_scaler:
            self.scaler = MinMaxScaler()
            X = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("必须先在训练集上 fit scaler")
            X = self.scaler.transform(X)

        if self.sequence_length == 1:
            X_seq = X.reshape(len(X), 1, -1)
            y_seq = y
        else:
            X_seq, y_seq = [], []
            for i in range(len(X) - self.sequence_length):
                X_seq.append(X[i:i + self.sequence_length])
                y_seq.append(y[i + self.sequence_length])
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)

        return X_seq, y_seq

    # ================================
    # Non-IID 分段（原始方法）
    # ================================
    def sort_and_partition(self, X, y):
        """论文 Sort-and-Partition 方法"""
        idx = np.argsort(y)
        X, y = X[idx], y[idx]

        num_shards = self.num_devices * 2
        shard_size = len(X) // num_shards
        shards = [(X[i*shard_size:(i+1)*shard_size],
                   y[i*shard_size:(i+1)*shard_size])
                  for i in range(num_shards)]

        np.random.shuffle(shards)

        datasets = []
        for i in range(self.num_devices):
            X1, y1 = shards[2*i]
            X2, y2 = shards[2*i+1]
            datasets.append((np.vstack([X1, X2]), np.concatenate([y1, y2])))

        return datasets

    def create_non_iid_datasets(self, parts):
        """创建 Non-IID 数据集"""
        if len(parts) == 0:
            raise ValueError("parts 不能为空列表")
        while len(parts) < self.num_devices * 2:
            parts = parts + parts

        np.random.shuffle(parts)
        datasets = []

        for i in range(self.num_devices):
            Xs, ys = [], []
            for p in parts[2 * i:2 * i + 2]:
                Xs.append(p[0])
                ys.append(p[1])
            X = np.vstack(Xs)
            y = np.concatenate(ys)

            idx = np.random.permutation(len(X))
            datasets.append((X[idx], y[idx]))

        return datasets

    # ================================
    # 局部训练（FedProx + Focal Loss + 学习率调度）
    # ================================
    def local_train(self, local_data, global_weights,
                    epochs, batch_size, lr, mu, gamma, focal_gamma=2.0,
                    class_weights=None):
        """
        局部训练 (FedProx + Focal Loss + 学习率调度)

        返回:
            (local_weights, avg_loss): 局部模型权重和平均损失
            avg_loss 仅包含 base_loss（不含 proximal term），用于动态聚合权重计算
        """
        X, y = local_data

        if len(X) == 0:
            return global_weights, 0.0

        # 复用预创建的本地模型（避免每个客户端都 new + to(device)）
        if not hasattr(self, '_local_model'):
            self._local_model = LSTMModel(
                self.input_size,
                num_classes=self.num_classes
            ).to(self.device)
        model = self._local_model
        model.load_state_dict(global_weights)

        # 预搬运数据到 GPU（避免每个 batch 的 CPU→GPU 传输）
        loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X).to(self.device),
                torch.LongTensor(y).to(self.device)
            ),
            batch_size=min(batch_size, len(X)),
            shuffle=True
        )

        # 优先使用传入的全局类别权重（避免本地统计在 non-IID 下极端偏差）
        if class_weights is not None:
            cw_tensor = class_weights
        else:
            class_counts = np.bincount(y, minlength=self.num_classes).astype(float)
            weights = 1.0 / (class_counts + 1e-6)
            min_w = weights.min()
            max_ratio = 10.0 if self.num_classes <= 2 else 15.0
            weights = np.clip(weights, 0, min_w * max_ratio)
            weights = weights / weights.sum() * self.num_classes
            cw_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # 使用 Focal Loss
        loss_fn = FocalLoss(alpha=cw_tensor, gamma=focal_gamma)

        opt = optim.Adam(model.parameters(), lr=lr)

        # 学习率调度器 - CosineAnnealingLR
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=epochs,
            eta_min=lr * 0.01
        )

        total_base_loss = 0.0
        num_batches = 0

        model.train()
        # 预先将全局权重移到 device，避免每个 batch 重复搬运
        if mu > 0:
            global_weights_on_device = [global_weights[name].to(self.device)
                                         for name in global_weights]
        for epoch in range(epochs):
            for bx, by in loader:
                opt.zero_grad()
                out = model(bx)
                base_loss = loss_fn(out, by)

                if mu > 0:
                    prox = 0.0
                    for param, global_param in zip(model.parameters(),
                                                   global_weights_on_device):
                        prox += torch.norm(param - global_param) ** 2
                    total_loss = base_loss + (mu / 2) * prox
                else:
                    total_loss = base_loss

                total_loss.backward()
                opt.step()

                total_base_loss += base_loss.item()
                num_batches += 1

            scheduler.step()

        avg_loss = total_base_loss / max(num_batches, 1)
        return {k: v.cpu() for k, v in model.state_dict().items()}, avg_loss

    # ================================
    # 客户端模型评估
    # ================================
    def evaluate_local(self, model, X_val_seq, y_val_seq, val_loader=None):
        """评估单个客户端模型（使用 F1-Score）

        参数:
            X_val_seq: 已经过归一化和序列化的验证集特征 (n, seq_len, features)
            y_val_seq: 对应的标签向量
            val_loader: 预构建的 GPU DataLoader（可选，避免重复创建）
        """
        model.eval()

        if val_loader is None:
            val_loader = DataLoader(
                TensorDataset(
                    torch.FloatTensor(X_val_seq).to(self.device),
                    torch.LongTensor(y_val_seq).to(self.device)
                ),
                batch_size=512,
                shuffle=False
            )

        all_preds, all_labels = [], []

        with torch.no_grad():
            for bx, by in val_loader:
                pred = torch.argmax(model(bx), 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(by.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        return f1
    # ================================
    # FedAvg
    # ================================
    def aggregate(self, weights_list, sample_counts):
        """加权 FedAvg 聚合"""
        total_samples = sum(sample_counts)
        agg = {}
        for k in weights_list[0]:
            agg[k] = sum(
                weights_list[i][k] * (sample_counts[i] / total_samples)
                for i in range(len(weights_list))
            )
        return agg

    def aggregate_dynamic(self, weights_list, sample_counts, losses):
        """
        动态加权聚合: 权重 ∝ 样本数 × (1 / 局部损失)
        损失越小的客户端获得更高的权重

        修正: loss+0.3 防止 loss≈0 时权重爆炸（与 FedPCNN 对齐）
        clip 3× 均值防止极端客户端主导聚合
        """
        weights = []
        for size, loss in zip(sample_counts, losses):
            # loss+0.3: 典型 loss 范围 0.1-1.0，权重最大倍差约 4x
            w = size * (1.0 / (loss + 0.3))
            weights.append(w)

        # 裁剪：单个客户端权重不超过均值的 3 倍
        weights = np.array(weights, dtype=np.float64)
        weights = np.clip(weights, 0.0, weights.mean() * 3.0)

        total_weight = weights.sum()
        weights = weights / total_weight

        agg = {}
        for k in weights_list[0]:
            agg[k] = sum(
                weights_list[i][k] * weights[i]
                for i in range(len(weights_list))
            )
        return agg

    # ================================
    # 🔥 训练（支持外部 client_data）
    # ================================
    def train(self, client_data=None, X_train=None, y_train=None,
              X_val=None, y_val=None,
              global_rounds=20,
              local_epochs=5,
              client_fraction=0.5,
              batch_size=32,
              lr=0.001,
              mu=0.01,
              gamma=0.3,
              eval_interval=5,
              threshold=0.5,
              focal_gamma=2.0,
              dynamic_aggregation=True,
              pre_smote_class_weights=None):
        """
        分段式联邦学习模型

        参数:
            client_data: dict, 外部划分的客户端数据 {client_id: (X, y)}
            X_train, y_train: 原始训练数据（如果 client_data 为 None）
            X_val, y_val: 验证数据
            global_rounds: 全局轮数
            local_epochs: 本地训练轮数
            client_fraction: 客户端采样比例
            batch_size: 批次大小
            lr: 学习率
            mu: FedProx 参数
            gamma: γ-不精确解参数
            eval_interval: 评估间隔
            threshold: 客户端筛选阈值
            focal_gamma: Focal Loss 的 γ 参数
            dynamic_aggregation: 是否使用动态加权聚合
        """

        print("\n" + "=" * 60)
        print("开始分段式联邦学习训练")
        print("=" * 60)

        #  Step 1: 决定使用外部数据还是内部划分
        if client_data is not None:
            print("\n 使用外部划分的客户端数据")

            # 先 fit scaler（使用所有客户端数据）
            print("\nStep 1: Fit Scaler（所有客户端数据）...")
            all_X = np.vstack([client_data[i][0] for i in range(self.num_devices)])
            all_y = np.concatenate([client_data[i][1] for i in range(self.num_devices)])

            self.scaler = MinMaxScaler()
            self.scaler.fit(all_X)

            # 预处理每个客户端的数据
            print("\nStep 2: 预处理客户端数据...")
            local_sets = {}
            for client_id in range(self.num_devices):
                X_c, y_c = client_data[client_id]
                X_c = self.scaler.transform(X_c)

                if self.sequence_length == 1:
                    # sequence_length=1：每个样本独立，不丢失数据
                    X_seq = X_c.reshape(len(X_c), 1, -1)
                    y_seq = y_c
                else:
                    # 滑动窗口：标签取窗口最后一个时间步
                    X_seq, y_seq = [], []
                    for i in range(len(X_c) - self.sequence_length + 1):
                        X_seq.append(X_c[i:i + self.sequence_length])
                        y_seq.append(y_c[i + self.sequence_length - 1])
                    X_seq = np.array(X_seq) if X_seq else np.empty((0, self.sequence_length, X_c.shape[1]))
                    y_seq = np.array(y_seq)

                local_sets[client_id] = (X_seq, y_seq)

            # 打印数据分布
            print(f"\n 客户端数据统计（预处理后）:")
            for client_id in range(min(5, self.num_devices)):
                X_c, y_c = local_sets[client_id]
                unique, counts = np.unique(y_c, return_counts=True)
                dist = {int(k): int(v) for k, v in zip(unique, counts)}
                print(f"   客户端 {client_id}: {len(y_c):5d} 样本, 类别 {dist}")
            if self.num_devices > 5:
                print(f"   ... (省略其余 {self.num_devices - 5} 个客户端)")

        else:
            if X_train is None or y_train is None:
                raise ValueError("必须提供 client_data 或 (X_train, y_train)")

            print("\n  使用内部分段式 Non-IID 划分")

            # 1️ 预处理（训练集fit）
            print("\nStep 1: 预处理训练集...")
            X_train, y_train = self.preprocess_data(
                X_train, y_train, fit_scaler=True
            )

            # 2️ Non-IID 划分
            print("\nStep 2: 分段式 Non-IID 划分...")
            local_sets_list = self.sort_and_partition(X_train, y_train)
            local_sets = {i: local_sets_list[i] for i in range(len(local_sets_list))}

        # 计算全局类别权重
        # 优先使用 pre_smote_class_weights（SMOTE前真实分布），避免SMOTE后虚假分布误导Focal Loss
        # SMOTE 会合成大量少数类样本 → 类别分布变均衡 → 权重差异被压缩 → Focal Loss 无法区分
        if pre_smote_class_weights is not None:
            global_class_weights = pre_smote_class_weights.to(self.device)
            print(f"  全局类别权重(SMOTE前): {dict(enumerate(pre_smote_class_weights.numpy().round(3).tolist()))}")
        else:
            all_labels_global = np.concatenate([local_sets[i][1] for i in range(self.num_devices)])
            global_counts = np.maximum(
                np.bincount(all_labels_global, minlength=self.num_classes), 1
            ).astype(float)
            global_cw = 1.0 / global_counts
            min_gcw = global_cw.min()
            max_ratio = 10.0 if self.num_classes <= 2 else 15.0
            global_cw = np.clip(global_cw, 0, min_gcw * max_ratio)
            global_cw = global_cw / global_cw.sum() * self.num_classes
            global_class_weights = torch.FloatTensor(global_cw).to(self.device)
            print(f"  全局类别权重(SMOTE后): {dict(enumerate(global_cw.round(3).tolist()))}")

        # 3️ 训练
        print(f"\nStep 3: 开始 {global_rounds} 轮全局训练...")
        print(f"  本地轮数: {local_epochs}")
        print(f"  客户端采样率: {client_fraction}")
        print(f"  评估间隔: {eval_interval} 轮")
        print(f"  筛选阈值: {threshold}")
        print(f"  Focal Loss γ: {focal_gamma}")
        print(f"  动态聚合: {'启用' if dynamic_aggregation else '禁用'}")

        # 预处理验证集（使用与训练集相同的 scaler，统一在此处完成）
        X_val_seq, y_val_seq = None, None
        if X_val is not None and y_val is not None:
            print(f"\n验证集: {X_val.shape}")
            X_val_norm = self.scaler.transform(X_val)
            if self.sequence_length == 1:
                X_val_seq = X_val_norm.reshape(len(X_val_norm), 1, -1)
                y_val_seq = y_val
            else:
                X_val_seq_list, y_val_seq_list = [], []
                for i in range(len(X_val_norm) - self.sequence_length + 1):
                    X_val_seq_list.append(X_val_norm[i:i + self.sequence_length])
                    y_val_seq_list.append(y_val[i + self.sequence_length - 1])
                X_val_seq = np.array(X_val_seq_list)
                y_val_seq = np.array(y_val_seq_list)
        else:
            print(f"\n  未提供验证集，将跳过客户端筛选")

        # 预构建验证集 GPU DataLoader（避免每个客户端评估时重复创建和搬运）
        val_loader = None
        if X_val_seq is not None:
            val_loader = DataLoader(
                TensorDataset(
                    torch.FloatTensor(X_val_seq).to(self.device),
                    torch.LongTensor(y_val_seq).to(self.device)
                ),
                batch_size=512,
                shuffle=False
            )

        client_metrics = []  # 存储每轮客户端指标
        train_loss_history = []
        val_loss_history = []   # 使用 mean F1 作为验证指标（反向：1-f1 作为验证"损失"）

        global_bar = tqdm(range(global_rounds), desc="Global", unit="epoch",
                          ncols=100, colour='blue', position=0, leave=True)
        for r in global_bar:
            m = max(1, int(client_fraction * self.num_devices))
            clients = np.random.choice(self.num_devices, m, replace=False)

            global_w = self.global_model.state_dict()
            local_ws = []
            sample_counts = []
            local_losses = []
            round_metrics = []

            client_bar = tqdm(clients, desc=f"Epoch {r+1}/{global_rounds}",
                              unit="client", ncols=120, colour='green', position=1, leave=False)
            for c in client_bar:
                w, avg_loss = self.local_train(
                    local_sets[c], global_w,
                    local_epochs, batch_size,
                    lr, mu, gamma, focal_gamma,
                    class_weights=global_class_weights
                )

                if val_loader is not None and (r + 1) % eval_interval == 0:
                    self._local_model.load_state_dict(w)
                    f1 = self.evaluate_local(self._local_model, X_val_seq, y_val_seq, val_loader=val_loader)
                    round_metrics.append(f1)

                local_ws.append(w)
                sample_counts.append(len(local_sets[c][1]))
                local_losses.append(avg_loss)

            round_avg_loss = float(np.mean(local_losses))
            train_loss_history.append(round_avg_loss)

            #  周期性筛选
            if X_val_seq is not None and (r + 1) % eval_interval == 0 and len(round_metrics) > 0:
                mean_f1 = np.mean(round_metrics)
                val_loss_history.append(1.0 - mean_f1)

                filtered_ws, filtered_counts, filtered_losses = [], [], []
                for i, f1 in enumerate(round_metrics):
                    d_i = f1 - mean_f1
                    ke = 1 / (1 + np.exp(-d_i))
                    if ke >= threshold:
                        filtered_ws.append(local_ws[i])
                        filtered_counts.append(sample_counts[i])
                        filtered_losses.append(local_losses[i])

                if len(filtered_ws) > 0:
                    local_ws = filtered_ws
                    sample_counts = filtered_counts
                    local_losses = filtered_losses
                    global_bar.write(f"  [Epoch {r+1}] 筛选后保留 {len(filtered_ws)}/{m} 个客户端 "
                                     f"(mean_f1={mean_f1:.4f})")
                else:
                    global_bar.write(f"  [Epoch {r+1}] 所有客户端被丢弃，保留全部")

            if len(local_ws) == 0:
                global_bar.write(f"Epoch {r+1:3d}/{global_rounds} | train_loss={round_avg_loss:.4f} | clients=0")
                continue

            if dynamic_aggregation:
                aggregated = self.aggregate_dynamic(local_ws, sample_counts, local_losses)
            else:
                aggregated = self.aggregate(local_ws, sample_counts)
            self.global_model.load_state_dict(aggregated)

            # ── 每轮评估 train_acc / val_loss / val_acc ──
            self.global_model.eval()
            with torch.no_grad():
                # train_acc（在本轮参与训练的客户端数据上评估）
                t_correct, t_total = 0, 0
                for c_id in clients:
                    X_c, y_c = local_sets[c_id]
                    tbx = torch.FloatTensor(X_c).to(self.device)
                    tby = torch.LongTensor(y_c).to(self.device)
                    t_correct += (self.global_model(tbx).argmax(1) == tby).sum().item()
                    t_total += len(tby)
                train_acc = 100.0 * t_correct / max(t_total, 1)

                # val_loss / val_acc
                val_loss_cur, val_acc_cur = 0.0, 0.0
                if val_loader is not None:
                    v_loss, v_correct, v_total = 0.0, 0, 0
                    for vbx, vby in val_loader:
                        out = self.global_model(vbx)
                        v_loss += F.cross_entropy(out, vby).item() * len(vby)
                        v_correct += (out.argmax(1) == vby).sum().item()
                        v_total += len(vby)
                    val_loss_cur = v_loss / max(v_total, 1)
                    val_acc_cur = 100.0 * v_correct / max(v_total, 1)

            # 打印轮次信息（进度条 + 四项指标）
            bar = '█' * 20
            global_bar.write(
                f"Epoch {r+1:3d}/{global_rounds} |{bar}| {m}/{m} "
                f"train_loss={round_avg_loss:.4f}, train_acc={train_acc:.2f}%, "
                f"val_loss={val_loss_cur:.4f}, val_acc={val_acc_cur:.2f}%")
            global_bar.set_postfix(
                t_loss=f"{round_avg_loss:.4f}", t_acc=f"{train_acc:.1f}%",
                v_loss=f"{val_loss_cur:.4f}", v_acc=f"{val_acc_cur:.1f}%")

        global_bar.close()
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)

        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history
        return train_loss_history, val_loss_history

    # ================================
    # 测试评估
    # ================================
    def evaluate(self, X_test, y_test):
        """评估模型性能"""

        print("\n评估模型性能...")

        X_test, y_test = self.preprocess_data(
            X_test, y_test, fit_scaler=False
        )

        loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_test).to(self.device),
                torch.LongTensor(y_test).to(self.device)
            ),
            batch_size=512,
            shuffle=False
        )

        self.global_model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for bx, by in loader:
                pred = torch.argmax(self.global_model(bx), 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(by.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100

        # 打印逐类别 Recall，便于定位稀有类（R2L/U2R）学习效果
        per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        print(f"\n  逐类别 Recall:")
        for cls_idx, rec in enumerate(per_class_recall):
            print(f"    类别 {cls_idx}: {rec*100:.1f}%")

        # FAR 通用公式（class 0 = Normal，支持二分类和多分类）
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
            'FAR': FAR
        }

        return metrics