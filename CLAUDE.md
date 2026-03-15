# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

联邦学习(Federated Learning)网络入侵检测系统。实现了两种联邦学习算法（FedPCNN、分段式FL）和多种集中式基线模型，在NSL-KDD、UNSW-NB15、CIC-IDS2017数据集上进行入侵检测实验。项目语言为中文。

## Common Commands

```bash
# 安装依赖 (Python 3.8, CUDA 12.1)
pip install -r requirements.txt

# 运行单次联邦学习实验
python main.py --model fedpcnn --dataset NSL-KDD --partition iid --classification multi
python main.py --model segmented --dataset NSL-KDD --partition non-iid --alpha 0.5
python main.py --model fedpcnn-2stage --dataset UNSW-NB15 --partition iid

# 运行基线模型对比 (集中式: LIBSVM/CNN/DNN/DBN-EGWO-KELM)
python run_baselines.py --dataset NSL-KDD --classification multi
python run_baselines.py --dataset NSL-KDD --skip-dbn  # 跳过耗时的DBN-EGWO-KELM

# 消融实验 (动态聚合 & Focal Loss γ参数)
python run_ablation.py --dataset NSL-KDD --partition iid --classification multi

# XGBoost基线
python baseline_xgboost.py

# 可视化
python plot_results.py
python plot_comparison.py
```

**关键命令行参数:**
- `--model`: `fedpcnn` | `fedpcnn-2stage` (两阶段分类) | `segmented`
- `--dataset`: `NSL-KDD` | `UNSW-NB15` | `CIC-IDS2017`
- `--partition`: `iid` | `non-iid`
- `--alpha`: Dirichlet参数 (0.1-1.0, 越小数据越不均衡)
- `--classification`: `binary` | `multi`
- `--bohb N`: XGBoost BOHB超参搜索试验次数 (0=禁用)
- `--global-rounds`, `--local-epochs`, `--seed`, `--no-cuda`

## Architecture

**入口与流程:** `main.py` 解析参数 → 加载数据集 → 数据预处理 → IID/Non-IID分区 → SMOTE过采样 → 联邦训练 → 评估 → 保存结果/绘图

**核心模块:**

- `models/fedpcnn.py` — FedPCNN联邦学习
  - `CNNSVM`: 1D CNN (Conv→GroupNorm→MaxPool→GlobalAvgPool→FC) 输出128维特征
  - `SRFCNNBlock1D`: 分离-残差-融合卷积块 (kernel=3和5并行 + 残差连接)
  - `FedPCNN`: 联邦训练协调器，支持FedProx(mu)近端项、γ-不精确解、动态聚合(基于损失加权)
  - 训练后可选: cRT分类器重训练、Logit偏置校准、XGBoost二级分类器 + Normal门限搜索
  - 支持断点续训 (checkpoint保存到 `checkpoints/`)

- `models/segmented_fl.py` — 分段式联邦学习
  - `LSTMModel`: LSTM(hidden=256, layers=2) + FC
  - `FocalLoss`: γ=2.0, 处理类别不平衡
  - 周期性评估(eval_interval=5) + 阈值筛选(threshold=0.45)
  - CosineAnnealingLR学习率调度

- `data_preprocessing.py` — 数据加载与预处理
  - `NSLKDDPreprocessor` / `UNSWNB15Preprocessor` / `CICIDS2017Preprocessor`
  - 预处理流程: LabelEncoder → StandardScaler → MinMaxScaler
  - `partition_iid()` / `partition_non_iid()`: Dirichlet分布数据分区
  - `apply_smote_per_client()`: 各客户端本地SMOTE过采样

- `models/baseline/` — 集中式基线模型
  - `traditional_models.py`: LIBSVM
  - `deep_models.py`: CNN, DNN
  - `dbn_egwo_kelm.py`: DBN-EGWO-KELM

- `utils/result_logger.py` — 结果保存 (JSON + CSV到 `results/`)
- `utils/plot_utils.py` — 可视化 (损失曲线、混淆矩阵、模型对比图)

**FedPCNN两阶段分类** (`run_fedpcnn_two_stage`): Stage1二分类(Normal vs Attack) → Stage2 9类攻击分类 → 合并为10类预测。专为UNSW-NB15多分类设计。

**数据流:** 原始数据 → Preprocessor → (X_train, y_train) → partition_data() → client_data dict → apply_smote_per_client() → client_data_list → model.train()

## Evaluation Metrics

Accuracy, Precision, Recall, F1-Score, Macro-F1, FAR (误报率 = (FPR+FNR)/2)

结果自动保存到 `results/summary.csv` 和 `results/*.json`，模型权重保存到 `results/models/`

## Key Conventions

- 超参数按数据集分别配置 (NSL-KDD vs UNSW-NB15)，在 `main.py` 的 `run_fedpcnn()` 函数中硬编码
- Non-IID场景使用SMOTE预热策略 (前N轮用原始数据稳定全局模型)
- 默认10个联邦客户端，50轮全局训练
- 使用GroupNorm而非BatchNorm，适配联邦Non-IID场景
- 可视化失败不影响已保存的实验结果 (try-except包裹)
