# FL联邦学习实验进度报告

> 来源：飞书文档 XeJJdvASmoIYE5xYgtCcusGynFN

## 实验环境
- GPU：NVIDIA RTX 4090 24GB
- PyTorch 2.5.1 + CUDA 12.6
- XGBoost 3.2（GPU加速）
- 代码分支：optimize-v1

---

## 第四章 — FedPCNN 二分类（IID）

模型架构：3层CNN(32→64→128) + FocalLoss + FedProx + Stacking集成(RF+KNN+XGBoost)

### NSL-KDD ✅ 全部达标

| 指标 | 实际 | 目标 | 状态 |
|------|------|------|------|
| Accuracy | 99.46% | 99.35% | ✅ |
| Precision | 99.47% | 94.26% | ✅ |
| FAR | 0.54% | 0.62% | ✅ |
| Recall | 99.46% | 97.13% | ✅ |
| F1 | 99.46% | 95.67% | ✅ |

### UNSW-NB15 — 部分达标，优化中

| 指标 | 实际 | 目标 | 状态 |
|------|------|------|------|
| Accuracy | 95.91% | 99.04% | ❌ 差3.1% |
| Precision | 95.26% | 93.17% | ✅ |
| FAR | 4.01% | 1.87% | ❌ |
| Recall | 95.99% | 97.92% | ❌ 差2% |
| F1 | 95.60% | 95.49% | ✅ |

### CIC-IDS2017 — 待真实数据
当前使用合成数据验证代码流程，Accuracy 81.42%。CIC-IDS2017真实数据集约1.3GB，需从Kaggle或官方下载。

---

## 第五章 — 分段式FL（Non-IID）

模型架构：LSTM + 周期筛选 + 阈值机制

### 已完成的优化
- CNN架构升级：1层(64 filters) → 3层(32→64→128)，匹配论文描述
- 特征维度：UNSW-NB15 从20维扩展到38维（MRMR阈值选择）
- XGBoost兼容：适配3.x版本GPU加速
- SMOTE过采样：修复新版库兼容问题
- 断点续训bug修复

---

### UNSW-NB15 多分类 Non-IID v1（基础版）
- 模型: FedPCNN (CNN+FedProx+XGBoost Stacking)
- 数据: 10类, Non-IID (Dirichlet α=0.5), 10客户端
- 配置: 100轮, local_epochs=5, lr=0.005

| 指标 | 结果 |
|------|------|
| Accuracy | 79.35% |
| Macro-F1 | 58.95% |
| FAR | 6.81% |

分析: 类别极度不平衡（Worms仅174样本 vs Normal 93000样本），Non-IID下部分客户端完全缺失某些攻击类别。

更新时间：2026年3月18日 19:30

---

### UNSW-NB15 二分类 Non-IID v4
- 模型: FedPCNN (CNN+FedProx+XGBoost Stacking)
- 数据: 二分类 (Normal / Attack), Non-IID (Dirichlet α=0.5), 10客户端, 保底每类5%
- 配置: 150轮, local_epochs=8, lr=0.008, μ=0.08, Grad Clip=1.0

| 指标 | 实际 | 目标 | 状态 |
|------|------|------|------|
| Accuracy | 94.39% | 97.83% | ❌ 差3.44% |
| F1 | 94.44% | 94.55% | ≈ 持平 ✅ |
| FAR | 5.14% | 3.06% | ❌ 差2.08% |

更新时间：2026年3月18日 21:17

---

### UNSW-NB15 多分类 Non-IID v3（CNN加宽版）
- 模型: FedPCNN (CNN 64→128→256 + FedProx + XGBoost Stacking)
- 配置: 150轮, local_epochs=2, lr=0.003, μ=0.15, FocalLoss(γ=2.0)

| 指标 | 结果 |
|------|------|
| Accuracy | 79.05% (v1: 79.35%) ⬇️ |
| Macro-F1 | 58.18% (v1: 58.95%) ⬇️ |
| FAR | 6.86% |

结论: CNN加宽未带来提升，反而因参数增多+local_epochs=2导致欠训练。

更新时间：2026年3月18日 22:25

---

### UNSW-NB15 多分类 Non-IID v4（BalancedSoftmax + DRW）
- 模型: FedPCNN (CNN 32→64→128 + FedProx + XGBoost Stacking)
- 特征数: 38
- 配置: 60轮

| 指标 | 结果 |
|------|------|
| Accuracy | 78.92% |
| Macro-F1 | 59.54% |
| FAR | 7.00% |

更新时间：2026年3月19日 07:00

---

### UNSW-NB15 多分类 Non-IID v5（类别特征拆分 + 动态聚合 + FAR约束）
- 特征数: 38 (连续: 35, 类别: 3)
- 改动: CNN只处理连续特征，动态聚合，Threshold+FAR约束(λ=10)
- 配置: 60轮, lr=0.005, mu=0.10, local_epochs=5

| 指标 | 结果 |
|------|------|
| Accuracy | 78.78% |
| Macro-F1 | 58.22% |
| FAR | 7.08% |

分析: 三个结构性改动均未带来提升。

更新时间：2026年3月19日 11:16

---

### UNSW-NB15 多分类 Non-IID v6（Phase1 回退+增量优化）
- 模型: FedPCNN (CNN 1层×64 + FedProx + XGBoost Stacking)
- 特征数: 20 (MRMR top_k=20)
- 改动: CNN回退1层, FocalLoss(γ=1.5,ls=0.1), CB权重(β=0.999), 禁用动态聚合, CosineAnnealingWarmRestarts
- 配置: 60轮, lr=0.005, mu=0.05, local_epochs=5

| 指标 | 结果 |
|------|------|
| Accuracy | 78.90% |
| Macro-F1 | 59.60% |
| FAR | 7.11% |

更新时间：2026年3月19日 14:06

---

### ⚠️ v7 实验说明（代码问题未修复）

发现时间：2026年3月19日 22:56（GPT 5.4 代码审查）

**问题 1（🔴 高）：** checkpoint 版本标识不匹配 — CNN已从1层改为2层，但checkpoint文件名仍为_v3
**问题 2（🟡 中）：** class weights 未按计划修改 — commit说明写"√逆频率"，代码仍为CB effective-number

### UNSW-NB15 多分类 Non-IID v7（2层CNN + CenterLoss，⚠️权重未修复）
- 特征数: 20
- 配置: 60轮, lr=0.005, mu=0.05, local_epochs=5

| 指标 | 结果 |
|------|------|
| Accuracy | 78.39% |
| Macro-F1 | 58.56% |
| FAR | 7.80% |

结论: 不修复权重就加复杂度无效。

更新时间：2026年3月20日 07:30

---

### UNSW-NB15 多分类 Non-IID v8（回退历史最佳架构 + 扩展特征）
- 模型: FedPCNN (CNN 1层×64 + FedProx + XGBoost Stacking)
- 特征数: 38 (MRMR threshold=0.95)
- 改动: CNN回退1层, class weights回退逆频率1/counts, 关闭CenterLoss, 100轮
- 配置: 100轮, lr=0.005, mu=0.05, local_epochs=5

| 指标 | 结果 |
|------|------|
| Accuracy | 78.84% |
| Macro-F1 | 59.84% |
| FAR | 7.29% |

更新时间：2026年3月20日 12:20

---

## 版本对比总览

| 版本 | Accuracy | Macro-F1 | FAR | 说明 |
|------|----------|----------|-----|------|
| 历史最佳 | 80.68% | 62.63% | 6.41% | 仓库3/14结果 |
| v1 (基础) | 79.35% | 58.95% | 6.81% | 3层CNN+38维 |
| v3 (CNN加宽) | 79.05% | 58.18% | 6.86% | 64→128→256 |
| v4 (BSL) | 78.92% | 59.54% | 7.00% | BalancedSoftmax |
| v5 (拆分) | 78.78% | 58.22% | 7.08% | 类别拆分+动态聚合 |
| v6 (回退) | 78.90% | 59.60% | 7.11% | 1层CNN+20维+CB权重 |
| v7 (权重未修) | 78.39% | 58.56% | 7.80% | 2层CNN+CenterLoss |
| v8 (本次) | 78.84% | 59.84% | 7.29% | 1层CNN+38维+逆频率 |
| cloud-base60 | 79.01% | 60.64% | 7.41% | RTX4090, 60轮, 门限0.30 |

---

## 云端实验记录（2026年3月21日）

### UNSW-NB15 多分类 Non-IID smoke（远端流程验证）
- 目的：验证云服务器环境、GPU训练、`--exp-tag` 产物隔离、图表归档流程
- 服务器：SeeTa Cloud RTX 4090 24GB（PyTorch 2.5.1 + CUDA 12.4, Python 3.12）
- 配置：5轮, `alpha=0.5`, `seed=42`, `local_epochs=5`, `lr=0.005`, `exp_tag=smoke`
- 归档目录：`results/archive/2026-03-21_230201_smoke_2026-03-21_230234/`
- 图表产物：`loss / confusion_matrix / metrics / per_class / comparison` 共5张

| 指标 | 结果 |
|------|------|
| Accuracy | 78.48% |
| Precision | 86.20% |
| Recall | 78.48% |
| F1-Score | 80.84% |
| Macro-Precision | 57.36% |
| Macro-Recall | 73.26% |
| Macro-F1 | 59.64% |
| FAR | 7.97% |

结论：远端 4090 环境和完整实验归档流程已验证通过；5轮 smoke 结果落在近期多分类 Non-IID 区间内，可作为正式 `base60` / `bohb50` 实验前的环境基线。

### UNSW-NB15 多分类 Non-IID base60（远端正式基线）
- 目的：在 RTX 4090 上跑一轮可复现正式基线，验证当前代码在云端的真实上限
- 服务器：SeeTa Cloud RTX 4090 24GB（PyTorch 2.5.1 + CUDA 12.4, Python 3.12）
- 配置：60轮, `alpha=0.5`, `seed=42`, `local_epochs=5`, `lr=0.005`, `exp_tag=base60`
- 最终分类器：`CNN+XGBoost(门限=0.30)`
- 归档目录：`results/archive/2026-03-21_235621_base60_2026-03-21_235639/`
- 图表产物：`loss / confusion_matrix / metrics / per_class / comparison` 共5张

| 指标 | 结果 |
|------|------|
| Accuracy | 79.01% |
| Precision | 86.23% |
| Recall | 79.01% |
| F1-Score | 81.12% |
| Macro-Precision | 58.14% |
| Macro-Recall | 74.03% |
| Macro-F1 | 60.64% |
| FAR | 7.41% |

结论：相较日志中的 `v8`（78.84 / 59.84 / 7.29），`base60` 把 `Macro-F1` 提升了 `+0.80`，`Accuracy` 提升了 `+0.17`，但 `FAR` 略高 `+0.12`。当前已证明远端环境可稳定复现 `60+ Macro-F1`，下一步看 `bohb50` 是否能进一步逼近历史最佳 `62.63 / 6.41`。
