

## 项目结构

```
├── data/                      # 数据集目录
│   ├── NSL-KDD/              # NSL-KDD数据集
│   └── UNSW-NB15/            # UNSW-NB15数据集
├── models/                    # 模型实现
│   ├── fedpcnn.py            # FedPCNN模型 
│   └── segmented_fl.py       # 分段式联邦学习 
├── utils/                     # 工具模块
│   └── result_logger.py      # 结果记录器
├── results/                   # 实验结果
│   ├── summary.csv           # 汇总表格
│   └── figures/              # 可视化图表
├── data_preprocessing.py      # 数据预处理与IID/Non-IID划分
├── data_loader.py            # 原始数据加载器
├── main.py                    # 主程序入口
├── run_all_experiments.py     # 批量实验运行
├── visualize_results.py       # 结果可视化
└── plot_results.py            # 快速绘图脚本
```

## 功能特性

✅ **真实数据集**: NSL-KDD和UNSW-NB15网络入侵检测数据集
✅ **数据预处理**: 特征编码、标准化、归一化
✅ **IID/Non-IID划分**: 支持独立同分布和非独立同分布数据划分
✅ **联邦学习算法**:
  - FedPCNN: 基于FedProx的CNN模型
  - 分段式联邦学习: 基于LSTM的Non-IID优化模型
✅ **性能评估**: Accuracy, Precision, Recall, F1-Score, FAR
✅ **结果记录**: 自动保存实验结果到JSON和CSV
✅ **可视化**: 自动生成对比图表和汇总表格

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

将NSL-KDD数据集放置在 `data/NSL-KDD/` 目录下：
- `KDDTrain+.txt`
- `KDDTest+.txt`

### 3. 运行实验

**单次实验 (分段式联邦学习 - IID):**
```bash
python main.py --model segmented --dataset NSL-KDD --partition iid
```

**单次实验 (分段式联邦学习 - Non-IID):**
```bash
python main.py --model segmented --dataset NSL-KDD --partition non-iid --alpha 0.5
```

**单次实验 (FedPCNN):**
```bash
python main.py --model fedpcnn --dataset NSL-KDD --partition iid
```

**批量运行所有实验:**
```bash
python run_all_experiments.py
```

**生成可视化图表:**
```bash
python plot_results.py
```

## 参数说明

- `--model`: 模型类型 (`fedpcnn` 或 `segmented`)
- `--dataset`: 数据集 (`NSL-KDD` 或 `UNSW-NB15`)
- `--partition`: 数据划分方式 (`iid` 或 `non-iid`)
- `--alpha`: Non-IID Dirichlet参数 (0.1-1.0, 越小越不均衡)
- `--global-rounds`: 全局训练轮次 (默认50)
- `--local-epochs`: 本地训练轮次 (默认10)
- `--seed`: 随机种子 (默认42)
- `--no-cuda`: 禁用GPU加速

## 算法实现

### 算法4.2: FedPCNN
- 基于FedProx的联邦深度学习
- CNN网络架构
- γ-不精确解优化
- Non-IID数据支持

### 算法5.1: 分段式联邦学习
- LSTM网络架构
- Non-IID数据分段划分
- 周期性模型评估与筛选
- FedAvg聚合策略

## 完整实验流程

### 1. 数据预处理
- 标签编码（LabelEncoder）
- 特征标准化（StandardScaler）
- 归一化（MinMaxScaler）
- 序列化（LSTM格式）

### 2. IID/Non-IID划分
- **IID**: 随机均匀分配数据
- **Non-IID**: Dirichlet分布控制数据不均衡程度

### 3. 联邦训练
- 局部训练：FedProx + γ-不精确解
- 周期评估：验证集性能筛选
- 全局聚合：FedAvg加权平均

### 4. 性能评估
- Accuracy: 准确率
- Precision: 精确率
- Recall: 召回率
- F1-Score: F1分数
- FAR: 误报率

## 实验结果

运行实验后，结果将自动保存：
- `results/summary.csv`: 所有实验的汇总表格
- `results/*.json`: 每次实验的详细结果
- `results/figures/`: 自动生成的对比图表
  - IID vs Non-IID对比图
  - 不同模型性能对比
  - Alpha参数敏感性分析
  - 训练历史曲线

查看结果：
```bash
# 查看CSV汇总
cat results/summary.csv

# 生成所有图表
python plot_results.py
```

## 数据预处理模块

`data_preprocessing.py` 提供：
- `NSLKDDPreprocessor`: 完整的NSL-KDD数据加载与预处理
  - 41个特征完整命名
  - 分类特征编码（protocol_type, service, flag）
  - 标准化与归一化
- `partition_iid()`: IID数据划分
- `partition_non_iid()`: Non-IID数据划分（Dirichlet分布）

## 使用示例

### 使用预处理器
```python
from data_preprocessing import NSLKDDPreprocessor, partition_non_iid

preprocessor = NSLKDDPreprocessor()
X_train, y_train, X_test, y_test = preprocessor.load_and_preprocess()

# Non-IID划分
client_data = partition_non_iid(X_train, y_train, num_clients=20, alpha=0.5)
```

### 训练模型
```python
from models.segmented_fl import SegmentedFederatedLearning

model = SegmentedFederatedLearning(
    num_devices=20,
    num_classes=5,
    input_size=41,
    sequence_length=10
)

model.train(X_train, y_train, X_val, y_val, global_rounds=50)
metrics = model.evaluate(X_test, y_test)
```









# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

联邦学习(Federated Learning)网络入侵检测系统，实现FedPCNN和分段式联邦学习两种算法，支持NSL-KDD和UNSW-NB15数据集。

## Common Commands

```bash
# 运行单次实验
python main.py --model fedpcnn --dataset NSL-KDD --partition iid
python main.py --model segmented --dataset NSL-KDD --partition non-iid --alpha 0.5

# 批量运行所有实验
python run_experiments.py

# 生成可视化图表
python plot_results.py

# 测试模型结构
python -c "from models.fedpcnn import CNNSVM; print(CNNSVM())"
```

**命令行参数:**
- `--model`: `fedpcnn` | `segmented`
- `--dataset`: `NSL-KDD` | `UNSW-NB15`
- `--partition`: `iid` | `non-iid`
- `--alpha`: Dirichlet参数 (0.1-1.0, 越小数据越不均衡)
- `--global-rounds`, `--local-epochs`, `--seed`, `--no-cuda`

## Architecture

```
main.py                    # 入口: 解析参数, 调用模型训练
├── data_preprocessing.py  # 数据加载/预处理/IID/Non-IID分区
├── models/
│   ├── fedpcnn.py         # FedPCNN: CNNSVM(3层CNN) + FocalLoss + FedProx + 动态聚合
│   ├── segmented_fl.py    # 分段式FL: LSTMModel + 周期筛选 + 阈值机制
│   └── baseline/          # 基准模型(CNN/DNN/SVM)
├── config.py              # 全局配置参数
└── utils/result_logger.py # 结果保存(JSON/CSV)
```

## Key Components

**FedPCNN** (`models/fedpcnn.py`):
- `CNNSVM`: 3层Conv(32→64→128) + GlobalAvgPool + FC(128→256→classes)
- `FocalLoss`: γ=2.0, 处理类别不平衡
- `local_train()`: 返回(weights, loss), 支持FedProx(mu)和γ-不精确解
- `aggregate_dynamic()`: 基于损失的动态权重聚合

**SegmentedFL** (`models/segmented_fl.py`):
- `LSTMModel`: LSTM(hidden=256, layers=2) + FC
- `local_train()`: 返回(weights, loss), FocalLoss + CosineAnnealingLR
- `train()`: 周期性评估(eval_interval=5), 阈值筛选(threshold=0.45)

**数据分区** (`data_preprocessing.py`):
- `partition_iid()`: 随机均匀分配
- `partition_non_iid()`: Dirichlet(alpha)分布

## Configuration

`config.py`中的关键参数:
- `focal_gamma`: FocalLoss γ参数 (默认2.0)
- `dynamic_aggregation`: 启用基于损失的动态聚合
- `mu`: FedProx近端项系数
- `lr_scheduler`: 学习率调度类型 ('cosine')

## Evaluation Metrics

Accuracy, Precision, Recall, F1-Score, FAR (误报率)

结果保存到 `results/summary.csv` 和 `results/*.json`




