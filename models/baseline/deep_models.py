import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class CNN(nn.Module):
    """
    深层CNN模型: 3层卷积 + GlobalAvgPool + FC
    升级版: 与 FedPCNN 的 CNNSVM 保持结构一致
    """
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        side = int(np.ceil(np.sqrt(input_size)))
        self.side = side

        # 3层卷积结构
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 1, self.side, self.side)

        # 3层卷积特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Global Average Pooling
        x = self.global_avg_pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
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
        return focal_loss.sum()

class DNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class DeepModelTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    def train(self, X_train, y_train, epochs=50, batch_size=32, lr=0.001, focal_gamma=2.0):
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        # 计算类别权重
        class_counts = np.bincount(y_train, minlength=len(np.unique(y_train)))
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        # 使用 Focal Loss
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        dataset = torch.utils.data.TensorDataset(X_train, y_train_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

    def evaluate(self, X_test, y_test):
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test_np = y_test

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.cpu().numpy()

        cm = confusion_matrix(y_test_np, y_pred)
        tn = cm[0, 0] if cm.shape[0] > 1 else 0
        fp = cm[0, 1:].sum() if cm.shape[0] > 1 else 0
        far = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0

        return {
            'Accuracy': accuracy_score(y_test_np, y_pred) * 100,
            'Precision': precision_score(y_test_np, y_pred, average='weighted', zero_division=0) * 100,
            'Recall': recall_score(y_test_np, y_pred, average='weighted', zero_division=0) * 100,
            'F1-Score': f1_score(y_test_np, y_pred, average='weighted', zero_division=0) * 100,
            'FAR': far
        }
