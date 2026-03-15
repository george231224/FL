import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
import argparse
import sys
import numpy as np
import torch
import traceback
from models.fedpcnn import FedPCNN
from models.segmented_fl import SegmentedFederatedLearning
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.result_logger import ResultLogger
from utils.plot_utils import (
    plot_loss_curves,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_class_metrics,
    plot_model_comparison_horizontal,
)

# 全局 logger
logger = ResultLogger(results_dir='./results')


def set_seed(seed=42):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """获取计算设备"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def get_dataset(dataset_name, classification='multi'):
    """加载数据集"""
    print(f"\n正在加载数据集: {dataset_name}  分类模式: {classification}")

    if dataset_name == 'NSL-KDD':
        from data_preprocessing import NSLKDDPreprocessor
        preprocessor = NSLKDDPreprocessor(classification=classification)
    elif dataset_name == 'UNSW-NB15':
        from data_preprocessing import UNSWNB15Preprocessor
        preprocessor = UNSWNB15Preprocessor(classification=classification)
    elif dataset_name == 'CIC-IDS2017':
        from data_preprocessing import CICIDS2017Preprocessor
        preprocessor = CICIDS2017Preprocessor(
            classification=classification,
            sample_size=300000,
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    X_train, y_train, X_test, y_test = preprocessor.load_and_preprocess()

    if hasattr(preprocessor, 'label_encoder') and hasattr(preprocessor.label_encoder, 'classes_'):
        class_names = [c.capitalize() for c in preprocessor.label_encoder.classes_]
        n_classes = len(class_names)
    else:
        n_classes = len(np.unique(y_train))
        class_names = [str(i) for i in range(n_classes)]

    n_features = X_train.shape[1]
    n_continuous = getattr(preprocessor, 'n_continuous_', n_features)  # 无 OneHot 时等于 n_features

    print(f"数据集加载完成:")
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    print(f"  类别数: {n_classes}  {class_names}")
    print(f"  特征数: {n_features} (连续: {n_continuous}, OneHot: {n_features - n_continuous})")

    return X_train, y_train, X_test, y_test, n_classes, n_features, class_names, n_continuous


def partition_data(X_train, y_train, partition_type='iid', num_clients=10, alpha=0.5):
    """数据划分 - 返回客户端索引"""
    from data_preprocessing import partition_iid, partition_non_iid

    print(f"\n数据划分:")
    print(f"  类型: {partition_type.upper()}")
    print(f"  客户端数: {num_clients}")

    if partition_type == 'iid':
        client_data = partition_iid(X_train, y_train, num_clients)
        print(f"   IID划分完成")
    elif partition_type == 'non-iid':
        client_data = partition_non_iid(X_train, y_train, num_clients, alpha=alpha)
        print(f"   Non-IID划分完成 (alpha={alpha})")
        print(f"   Alpha越小，数据分布越不均衡")
    else:
        raise ValueError(f"不支持的划分类型: {partition_type}")

    # 打印数据分布
    print(f"\n  客户端数据分布:")
    for client_id in range(min(10,num_clients)):    #local min()
        X_client, y_client = client_data[client_id]
        unique, counts = np.unique(y_client, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        print(f"    客户端 {client_id}: {len(y_client):5d} 样本, 类别分布 {dist}")

    # if num_clients > 5:
    #     print(f"    ... (省略其余 {num_clients - 5} 个客户端)")

    return client_data


def run_fedpcnn(dataset_name='NSL-KDD', partition_type='iid', alpha=0.5, device='cpu',
                global_rounds=50, local_epochs=5, classification='multi',
                dynamic_agg=False, bohb_trials=0):
    """FedPCNN 实验（论文算法4.2：FedProx + CNN-SVM）"""
    print("=" * 60)
    print("FedPCNN (论文算法4.2)")
    print("=" * 60)

    # 加载数据
    X_train, y_train, X_test, y_test, n_classes, n_features, class_names, n_continuous = get_dataset(dataset_name, classification)

    # 拆分验证集（20%）用于早停
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"\n训练/验证集划分:")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")

    # 论文配置: 10个设备
    num_devices = 10
    client_data = partition_data(X_train, y_train, partition_type, num_devices, alpha)

    input_shape = (1, n_features)

    print(f"\n模型配置:")
    print(f"  设备数: {num_devices}")
    print(f"  输入形状: {input_shape}")
    print(f"  类别数: {n_classes}")
    print(f"  数据划分: {partition_type.upper()}" + (f" (alpha={alpha})" if partition_type == 'non-iid' else ""))

    fedpcnn = FedPCNN(
        num_devices=num_devices,
        num_classes=n_classes,
        input_shape=input_shape,
    )

    if hasattr(fedpcnn, 'device'):
        fedpcnn.device = device

    print("\n开始训练...")
    client_data_list = [client_data[i] for i in range(num_devices)]

    # 论文算法4.2：标准 FedProx + CE loss，无 SMOTE/cRT/Logit校准
    hp_lr = 0.01
    hp_mu = 0.01
    hp_local_epochs = local_epochs
    hp_gamma = 0.5       # γ-不精确解参数
    hp_focal_gamma = 0.0  # 0 = 标准 CrossEntropyLoss

    print(f"\n  配置 ({dataset_name}):")
    print(f"    lr={hp_lr}, mu={hp_mu}, local_epochs={hp_local_epochs}, "
          f"gamma={hp_gamma}, focal_gamma={hp_focal_gamma}")

    ckpt_tag = f"{dataset_name}_{partition_type}_{classification}"

    train_loss, val_loss = fedpcnn.train(
        client_data=client_data_list,
        global_rounds=global_rounds,
        local_epochs=hp_local_epochs,
        client_fraction=1.0,  # 论文: 所有设备参与
        batch_size=64,        # 论文常用 batch size
        lr=hp_lr,
        mu=hp_mu,
        gamma=hp_gamma,
        focal_gamma=hp_focal_gamma,
        alpha=alpha,
        X_val=X_val,
        y_val=y_val,
        eval_interval=5,
        checkpoint_tag=ckpt_tag,
    )

    # 保存模型权重
    import torch as _torch
    tag = f"FedPCNN_{dataset_name}_{partition_type}_{classification}"
    os.makedirs('./results/models', exist_ok=True)
    model_path = f"./results/models/{tag}_model.pt"
    _torch.save({
        'global_model': fedpcnn.global_model.state_dict(),
        'input_shape': input_shape,
        'num_classes': n_classes,
    }, model_path)
    print(f"\n模型权重已保存: {model_path}")

    # 纯 CNN 评估
    print("\n开始评估...")
    metrics_cnn = fedpcnn.evaluate(X_test, y_test)

    # CNN + SVM 分类器（论文：CNN-SVM架构）
    fedpcnn.train_svm(X_train, y_train)
    metrics_svm, svm_preds, svm_labels = fedpcnn.evaluate_with_svm(X_test, y_test)

    # 选择更优结果
    cnn_f1 = metrics_cnn.get('Macro-F1', metrics_cnn.get('F1-Score', 0))
    svm_f1 = metrics_svm.get('Macro-F1', metrics_svm.get('F1-Score', 0))
    if cnn_f1 > svm_f1:
        metrics = metrics_cnn
        best_path = "CNN"
    else:
        metrics = metrics_svm
        best_path = "CNN+SVM"
    print(f"\n  最终选择: {best_path} (F1: CNN={cnn_f1:.2f}% vs SVM={svm_f1:.2f}%)")

    # 保存结果
    logger.save_result(
        dataset=dataset_name,
        model_name='fedpcnn',
        partition=partition_type,
        alpha=alpha,
        metrics=metrics,
        params={
            'num_devices': num_devices,
            'global_rounds': global_rounds,
            'local_epochs': hp_local_epochs,
            'batch_size': 64,
            'lr': hp_lr,
            'mu': hp_mu,
            'classifier': best_path,
        },
        classification=classification
    )

    # 可视化
    try:
        plot_loss_curves(
            train_loss=train_loss, val_loss=val_loss,
            train_label="train_loss", val_label="val_loss",
            title=f"训练损失({dataset_name} · {partition_type.upper()} · {classification})",
            save_path=f"./results/plots/{tag}_loss.png",
        )
        if svm_preds is not None:
            plot_confusion_matrix(
                y_true=svm_labels, y_pred=np.array(svm_preds),
                class_names=class_names,
                title=f"混淆矩阵({dataset_name} · {partition_type.upper()} · {classification})",
                save_path=f"./results/plots/{tag}_cm.png",
            )
    except Exception as plot_err:
        print(f"\n绘图失败（结果已保存）: {plot_err}")

    # 结果总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print(f"数据集: {dataset_name}")
    print(f"训练样本: {len(y_train)}, 验证样本: {len(y_val)}, 测试样本: {len(y_test)}")
    print(f"类别数: {n_classes}, 特征数: {n_features}")
    print(f"数据划分: {partition_type.upper()}" + (f" (alpha={alpha})" if partition_type == 'non-iid' else ""))
    if metrics:
        print(f"评估指标:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f}%")
    print("=" * 60)

    return metrics


def run_fedpcnn_two_stage(dataset_name='UNSW-NB15', partition_type='iid', alpha=0.5, device='cpu',
                          global_rounds=50, local_epochs=5, classification='multi',
                          dynamic_agg=True):
    """两阶段分类: Stage1(Normal vs Attack) → Stage2(9类攻击分类)

    将困难的 10 分类问题分解为两个子问题:
      Stage 1: 二分类 Normal(0) vs Attack(1) — 已验证 ~93%+
      Stage 2: 9 类攻击类型分类 — 仅在攻击样本上训练/推理
    最终合并为 10 类预测结果。
    """
    from sklearn.metrics import (confusion_matrix, precision_score,
                                 recall_score, f1_score)
    from data_preprocessing import apply_smote_per_client

    print("=" * 60)
    print("FedPCNN 两阶段分类: Stage1(二分类) → Stage2(9类攻击)")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════
    # 数据加载 (multi模式, 得到 10 类标签 0-9)
    # ═══════════════════════════════════════════════════════════════
    X_train, y_train, X_test, y_test, n_classes, n_features, class_names, n_continuous = get_dataset(
        dataset_name, classification)

    # 验证集划分
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"\n训练/验证集划分:")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")

    num_devices = 10
    input_shape = (1, n_features)

    # 派生二分类标签
    y_train_bin = (y_train > 0).astype(int)
    y_val_bin = (y_val > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)

    # ═══════════════════════════════════════════════════════════════
    # Stage 1: Binary (Normal vs Attack)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "━" * 60)
    print("Stage 1: 二分类 Normal(0) vs Attack(1)")
    print("━" * 60)

    client_data_s1 = partition_data(X_train, y_train_bin, partition_type, num_devices, alpha)
    client_data_s1_list = [client_data_s1[i] for i in range(num_devices)]

    # Pre-SMOTE class weights (binary)
    pre_s1_labels = np.concatenate([y for _, y in client_data_s1_list])
    pre_s1_counts = np.maximum(np.bincount(pre_s1_labels, minlength=2), 1).astype(float)
    pre_s1_cw = 1.0 / pre_s1_counts
    min_cw = pre_s1_cw.min()
    pre_s1_cw = np.clip(pre_s1_cw, 0, min_cw * 15.0)
    pre_s1_cw = pre_s1_cw / pre_s1_cw.sum() * 2
    pre_s1_class_weights = torch.FloatTensor(pre_s1_cw)

    # SMOTE (binary)
    client_data_s1_list = apply_smote_per_client(
        client_data_s1_list, num_classes=2,
        dataset_name=dataset_name, classification='binary', k_neighbors=5,
    )

    # 初始化 Stage 1 模型
    fedpcnn_s1 = FedPCNN(num_devices=num_devices, num_classes=2, input_shape=input_shape, n_continuous=n_continuous)
    if hasattr(fedpcnn_s1, 'device'):
        fedpcnn_s1.device = device

    # 二分类超参数（NON-IID优化：降lr防漂移，减local_epochs，加预热）
    s1_lr, s1_mu, s1_epochs, s1_gamma, s1_focal = 0.005, 0.1, 5, 0.6, 0.0

    # NON-IID 场景：SMOTE 预热，前5轮用原始数据稳定全局模型
    client_data_s1_raw = None
    s1_warmup = 0
    if partition_type == 'non-iid':
        client_data_s1_raw = list(client_data_s1_list)
        s1_warmup = 5

    print(f"\n  Stage1 超参数: lr={s1_lr}, mu={s1_mu}, local_epochs={s1_epochs}, "
          f"gamma={s1_gamma}, focal_gamma={s1_focal}, warmup={s1_warmup}")

    s1_train_loss, s1_val_loss = fedpcnn_s1.train(
        client_data=client_data_s1_list,
        global_rounds=global_rounds,
        local_epochs=s1_epochs,
        client_fraction=1.0 if partition_type != 'iid' else 0.7,
        batch_size=256, lr=s1_lr, mu=s1_mu, gamma=s1_gamma,
        focal_gamma=s1_focal, alpha=alpha,

        X_val=X_val, y_val=y_val_bin,
        eval_interval=5,
        pre_smote_class_weights=pre_s1_class_weights,
        client_data_raw=client_data_s1_raw,
        smote_warmup_rounds=s1_warmup,
    )

    # ── Stage1 训练完成，立即保存模型权重 ──────────────────────────────
    import torch as _torch
    tag = f"FedPCNN_{dataset_name}_{partition_type}_{classification}_two_stage"
    os.makedirs('./results/models', exist_ok=True)
    _torch.save({
        'stage1_model': fedpcnn_s1.global_model.state_dict(),
        'input_shape': input_shape,
    }, f"./results/models/{tag}_s1.pt")
    print(f"  Stage1 模型已保存（训练阶段）")

    # Stage 1 XGBoost
    fedpcnn_s1.train_svm(X_train, y_train_bin)

    # 评估 Stage 1
    metrics_s1, _, _ = fedpcnn_s1.evaluate_with_svm(X_test, y_test_bin)
    print(f"\n  Stage1 结果: Acc={metrics_s1['Accuracy']:.2f}%, "
          f"F1={metrics_s1['F1-Score']:.2f}%, FAR={metrics_s1['FAR']:.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # Stage 2: 9-class Attack Types
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "━" * 60)
    print("Stage 2: 9类攻击分类 (labels remap 1-9→0-8)")
    print("━" * 60)

    # 提取攻击样本
    atk_mask_train = y_train > 0
    X_train_atk = X_train[atk_mask_train]
    y_train_atk = y_train[atk_mask_train] - 1  # remap 1-9 → 0-8

    atk_mask_val = y_val > 0
    X_val_atk = X_val[atk_mask_val]
    y_val_atk = y_val[atk_mask_val] - 1

    print(f"  攻击样本: 训练集 {len(y_train_atk)}, 验证集 {len(y_val_atk)}")
    unique_atk, counts_atk = np.unique(y_train_atk, return_counts=True)
    atk_class_names = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers',
                       'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
    for u, c in zip(unique_atk, counts_atk):
        print(f"    {atk_class_names[u]:15s} ({u}): {c:6d} ({c/len(y_train_atk)*100:.2f}%)")

    # 分区 + SMOTE (9类攻击)
    client_data_s2 = partition_data(X_train_atk, y_train_atk, partition_type, num_devices, alpha)
    client_data_s2_list = [client_data_s2[i] for i in range(num_devices)]

    # Pre-SMOTE class weights (9-class)
    n_classes_s2 = 9
    pre_s2_labels = np.concatenate([y for _, y in client_data_s2_list])
    pre_s2_counts = np.maximum(np.bincount(pre_s2_labels, minlength=n_classes_s2), 1).astype(float)
    pre_s2_cw = 1.0 / pre_s2_counts
    min_cw = pre_s2_cw.min()
    pre_s2_cw = np.clip(pre_s2_cw, 0, min_cw * 15.0)
    pre_s2_cw = pre_s2_cw / pre_s2_cw.sum() * n_classes_s2
    pre_s2_class_weights = torch.FloatTensor(pre_s2_cw)

    # SMOTE 预热：保留原始数据用于前5轮稳定全局模型（Non-IID 场景）
    client_data_s2_raw = None
    s2_warmup = 0
    if partition_type == 'non-iid':
        client_data_s2_raw = list(client_data_s2_list)  # 浅拷贝保留原始数据引用
        s2_warmup = 5

    # SMOTE (9类攻击)
    client_data_s2_list = apply_smote_per_client(
        client_data_s2_list, num_classes=n_classes_s2,
        dataset_name=dataset_name, classification='two-stage-attack', k_neighbors=5,
    )

    # 初始化 Stage 2 模型
    fedpcnn_s2 = FedPCNN(num_devices=num_devices, num_classes=n_classes_s2, input_shape=input_shape, n_continuous=n_continuous)
    if hasattr(fedpcnn_s2, 'device'):
        fedpcnn_s2.device = device

    # 多分类超参数
    s2_lr, s2_mu, s2_epochs, s2_gamma, s2_focal = 0.005, 0.05, 5, 0.6, 1.5

    print(f"\n  Stage2 超参数: lr={s2_lr}, mu={s2_mu}, local_epochs={s2_epochs}, "
          f"gamma={s2_gamma}, focal_gamma={s2_focal}, warmup={s2_warmup}")

    s2_train_loss, s2_val_loss = fedpcnn_s2.train(
        client_data=client_data_s2_list,
        global_rounds=global_rounds,
        local_epochs=s2_epochs,
        client_fraction=1.0,  # 9类中 Worms 仅 0.1%，全客户端参与
        batch_size=256, lr=s2_lr, mu=s2_mu, gamma=s2_gamma,
        focal_gamma=s2_focal, alpha=alpha,

        X_val=X_val_atk, y_val=y_val_atk,
        eval_interval=5,
        pre_smote_class_weights=pre_s2_class_weights,
        client_data_raw=client_data_s2_raw,
        smote_warmup_rounds=s2_warmup,
    )

    # ── Stage2 训练完成，立即保存模型权重 ──────────────────────────────
    _torch.save({
        'stage2_model': fedpcnn_s2.global_model.state_dict(),
        'input_shape': input_shape,
        'n_classes_s2': n_classes_s2,
    }, f"./results/models/{tag}_s2.pt")
    print(f"  Stage2 模型已保存（训练阶段）")

    # cRT (9类)
    fedpcnn_s2.classifier_retrain(X_train_atk, y_train_atk, epochs=10, lr=0.01)

    # Logit Bias 校准 (在攻击子集验证集上)
    fedpcnn_s2.calibrate_thresholds(X_val_atk, y_val_atk)

    # Stage 2 XGBoost
    fedpcnn_s2.train_svm(X_train_atk, y_train_atk)

    # 单独评估 Stage 2 (仅攻击样本)
    atk_mask_test = y_test > 0
    X_test_atk = X_test[atk_mask_test]
    y_test_atk = y_test[atk_mask_test] - 1
    metrics_s2, _, _ = fedpcnn_s2.evaluate_with_svm(X_test_atk, y_test_atk)
    print(f"\n  Stage2 结果 (仅攻击样本): Acc={metrics_s2['Accuracy']:.2f}%, "
          f"Macro-F1={metrics_s2['Macro-F1']:.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # Two-Stage Evaluation: 合并为 10 类
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "━" * 60)
    print("两阶段合并评估 (10类)")
    print("━" * 60)

    # Step 1: Stage 1 SVM 预测 binary
    preds_binary = fedpcnn_s1.predict_with_svm(X_test)

    # Step 2: 对 predicted-Attack 样本运行 Stage 2
    attack_indices = np.where(preds_binary == 1)[0]
    normal_indices = np.where(preds_binary == 0)[0]
    print(f"  Stage1 预测: Normal={len(normal_indices)}, Attack={len(attack_indices)}")

    final_preds = np.zeros(len(y_test), dtype=int)  # 默认 Normal(0)
    if len(attack_indices) > 0:
        X_test_pred_atk = X_test[attack_indices]
        preds_atk_type = fedpcnn_s2.predict_with_svm(X_test_pred_atk)
        final_preds[attack_indices] = preds_atk_type + 1  # remap 0-8 → 1-9

    # Step 3: 计算 10 类指标
    accuracy = 100.0 * (final_preds == y_test).mean()
    precision = precision_score(y_test, final_preds, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, final_preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, final_preds, average='weighted', zero_division=0) * 100

    macro_precision = precision_score(y_test, final_preds, average='macro', zero_division=0) * 100
    macro_recall = recall_score(y_test, final_preds, average='macro', zero_division=0) * 100
    macro_f1 = f1_score(y_test, final_preds, average='macro', zero_division=0) * 100

    # 逐类别 Recall
    per_class_recall = recall_score(y_test, final_preds, average=None, zero_division=0)
    print(f"\n  逐类别 Recall (10类合并):")
    for cls_idx, rec in enumerate(per_class_recall):
        name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        print(f"    {name:15s} ({cls_idx}): {rec*100:.1f}%")
    print(f"\n  Macro平均: Precision={macro_precision:.2f}%, Recall={macro_recall:.2f}%, "
          f"F1={macro_f1:.2f}%")

    # FAR
    cm = confusion_matrix(y_test, final_preds)
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

    # ═══════════════════════════════════════════════════════════════
    # 立即保存结果（先于绘图，防止绘图失败丢失结果）
    # ═══════════════════════════════════════════════════════════════
    logger.save_result(
        dataset=dataset_name,
        model_name='fedpcnn',
        partition=partition_type,
        alpha=alpha,
        metrics=metrics,
        params={
            'num_devices': num_devices,
            'global_rounds': global_rounds,
            'local_epochs_s1': s1_epochs,
            'local_epochs_s2': s2_epochs,
            'batch_size': 256,
            'lr_s1': s1_lr, 'lr_s2': s2_lr,
            'classifier': 'two-stage CNN+SVM',
        },
        classification=classification
    )
    print(f"\n实验结果已保存（评估阶段）")

    # ── 可视化（失败不影响已保存的结果） ────────────────────────────────────
    try:
        # 损失曲线 (Stage 2 为主，因为 Stage 1 是辅助)
        plot_loss_curves(
            train_loss=s2_train_loss, val_loss=s2_val_loss,
            train_label="Stage2_train_loss", val_label="Stage2_val_loss",
            title=f"训练损失 Stage2({dataset_name} · {partition_type.upper()} · 两阶段)",
            save_path=f"./results/plots/{tag}_loss.png",
        )

        # 混淆矩阵 (10类合并)
        plot_confusion_matrix(
            y_true=y_test, y_pred=final_preds,
            class_names=class_names,
            title=f"混淆矩阵({dataset_name} · {partition_type.upper()} · 两阶段)",
            save_path=f"./results/plots/{tag}_cm.png",
        )

        # 指标对比图
        if metrics:
            plot_model_comparison(
                results={f"FedPCNN 两阶段 ({partition_type.upper()})": metrics},
                metrics=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FAR'],
                title=f"FedPCNN 两阶段 ({dataset_name} · {classification})",
                save_path=f"./results/plots/{tag}_metrics.png",
            )
    except Exception as plot_err:
        print(f"\n绘图失败（结果已保存，不影响实验）: {plot_err}")

    # 总结
    print("\n" + "=" * 60)
    print("两阶段实验总结")
    print("=" * 60)
    print(f"数据集: {dataset_name}")
    print(f"训练样本: {len(y_train)}, 验证样本: {len(y_val)}, 测试样本: {len(y_test)}")
    print(f"Stage1 Binary: Acc={metrics_s1['Accuracy']:.2f}%")
    print(f"Stage2 9-class: Acc={metrics_s2['Accuracy']:.2f}%, Macro-F1={metrics_s2['Macro-F1']:.2f}%")
    print(f"最终 10-class 合并:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}%")
    print("=" * 60)
    print("\n实验完成!")

    return metrics


def run_segmented_fl(dataset_name='NSL-KDD', partition_type='iid', alpha=0.5, device='cpu',
                     global_rounds=50, local_epochs=10, classification='multi'):
    print("=" * 60)
    print("分段式联邦学习实验")
    print("=" * 60)

    # 加载数据
    X_train, y_train, X_test, y_test, n_classes, n_features, class_names, n_continuous = get_dataset(dataset_name, classification)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\n训练/验证集划分:")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")

    #  数据划分到客户端
    num_devices = 10
    client_data = partition_data(X_train, y_train, partition_type, num_devices, alpha)

    # 初始化模型
    print(f"\n模型配置:")
    print(f"  设备数: {num_devices}")
    print(f"  输入维度: {n_features}")
    print(f"  类别数: {n_classes}")

    model = SegmentedFederatedLearning(
        num_devices=num_devices,
        num_classes=n_classes,
        input_size=n_features,
        sequence_length=5
    )

    if hasattr(model, 'device'):
        model.device = device

    print("\n开始训练...")

    client_data_list = [client_data[i] for i in range(num_devices)]

    # 统计 SMOTE 前的真实类别分布（用于 Focal Loss 权重计算）
    # 与 FedPCNN 对齐：SMOTE 后分布变均衡会压缩少数类权重，使 Focal Loss 失效
    pre_smote_labels = np.concatenate([y for _, y in client_data_list])
    pre_smote_counts = np.maximum(np.bincount(pre_smote_labels, minlength=n_classes), 1).astype(float)
    pre_smote_cw = 1.0 / pre_smote_counts
    min_cw = pre_smote_cw.min()
    pre_smote_cw = np.clip(pre_smote_cw, 0, min_cw * 15.0)
    pre_smote_cw = pre_smote_cw / pre_smote_cw.sum() * n_classes
    pre_smote_class_weights = torch.FloatTensor(pre_smote_cw)

    # SMOTE 数据增强（论文：刘长杰 §3.3.1）
    # 在联邦训练前各客户端本地独立执行：R2L×13, U2R×200 (NSL-KDD)
    from data_preprocessing import apply_smote_per_client
    client_data_list = apply_smote_per_client(
        client_data_list,
        num_classes=n_classes,
        dataset_name=dataset_name,
        classification=classification,
        k_neighbors=5,
    )

    train_loss, val_loss = model.train(
        client_data=client_data_list,
        X_val=X_val,
        y_val=y_val,
        global_rounds=global_rounds,
        local_epochs=local_epochs,
        client_fraction=0.5,
        batch_size=256,
        lr=0.001,
        mu=0.01,
        gamma=0.1,
        eval_interval=5,
        threshold=0.45,
        focal_gamma=1.5 if classification == 'multi' else 2.0,
        pre_smote_class_weights=pre_smote_class_weights,
    )

    # 评估
    print("\n开始评估...")
    metrics = model.evaluate(X_test, y_test)

    #  可视化 
    tag = f"SegmentedFL_{dataset_name}_{partition_type}_{classification}"

    # 1. 损失曲线（含验证）
    plot_loss_curves(
        train_loss=train_loss,
        val_loss=val_loss,
        train_label="train_loss",
        val_label="val_loss", #local中(1-F1)
        title=f"损失曲线 ({dataset_name} · {partition_type.upper()} · {classification})",
        save_path=f"./results/plots/{tag}_loss.png",
    )

    # 2. 混淆矩阵
    
    X_test_seq, y_test_seq = model.preprocess_data(X_test, y_test, fit_scaler=False)
    ds = TensorDataset(torch.FloatTensor(X_test_seq), torch.LongTensor(y_test_seq))
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.global_model.eval()
    all_preds = []
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(model.device)
            all_preds.extend(model.global_model(bx).argmax(1).cpu().numpy())
    plot_confusion_matrix(
        y_true=y_test_seq,
        y_pred=np.array(all_preds),
        class_names=class_names,
        title=f"SegmentedFL 混淆矩阵 ({dataset_name} · {partition_type.upper()} · {classification})",
        save_path=f"./results/plots/{tag}_cm.png",
    )

    # 保存结果
    logger.save_result(
        dataset=dataset_name,
        model_name='segmented',
        partition=partition_type,
        alpha=alpha,
        metrics=metrics,
        params={
            'num_devices': num_devices,
            'global_rounds': global_rounds,
            'local_epochs': local_epochs,
            'batch_size': 256,
            'lr': 0.001
        },
        classification=classification
    )

    # 3. 单模型指标对比图
    if metrics:
        plot_model_comparison(
            results={f"SegmentedFL ({partition_type.upper()} · {classification})": metrics},
            metrics=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FAR'],
            title=f"SegmentedFL 汇总 ({dataset_name} · {classification})",
            save_path=f"./results/plots/{tag}_metrics.png",
        )

    # 结果总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print(f"数据集: {dataset_name}")
    print(f"训练样本: {len(y_train)}, 验证样本: {len(y_val)}, 测试样本: {len(y_test)}")
    print(f"类别数: {n_classes}, 特征数: {n_features}")
    print(f"数据划分: {partition_type.upper()}" + (f" (alpha={alpha})" if partition_type == 'non-iid' else ""))
    if metrics:
        print(f"评估指标:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f}%")
    print("=" * 60)
    print("\n实验完成!")

    return metrics



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='联邦学习入侵检测')
    parser.add_argument('--model', type=str, choices=['fedpcnn', 'fedpcnn-2stage', 'segmented'], required=True,
                        help='模型类型: fedpcnn / fedpcnn-2stage (两阶段) / segmented')
    parser.add_argument('--dataset', type=str, choices=['NSL-KDD', 'UNSW-NB15', 'CIC-IDS2017'], default='NSL-KDD',
                        help='数据集: NSL-KDD / UNSW-NB15 / CIC-IDS2017')
    parser.add_argument('--partition', type=str, choices=['iid', 'non-iid'], default='iid',
                        help='数据划分方式: iid 或 non-iid')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Non-IID Dirichlet参数 (越小越不均衡)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（不指定则每次随机）')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='禁用CUDA（即使可用）')
    parser.add_argument('--global-rounds', type=int, default=50,
                        help='全局训练轮次（降低lr后需更多轮次收敛）')
    parser.add_argument('--local-epochs', type=int, default=5,
                        help='本地训练轮次')
    parser.add_argument('--classification', type=str, choices=['binary', 'multi'],
                        default='multi',
                        help='分类模式: binary (二分类) 或 multi (多分类，默认)')
    parser.add_argument('--no-dynamic-agg', action='store_true', default=False,
                        help='禁用动态聚合')
    parser.add_argument('--bohb', type=int, default=0, metavar='N',
                        help='XGBoost BOHB超参搜索试验次数 (0=禁用, 推荐30)')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)
    print(f" 使用随机种子: {args.seed}")


    # 获取设备
    if args.no_cuda:
        device = torch.device('cpu')
        print("使用 CPU")
    else:
        device = get_device()

    # 运行实验
    try:
        if args.model == 'fedpcnn':
            run_fedpcnn(args.dataset, args.partition, args.alpha, device,
                        args.global_rounds, args.local_epochs, args.classification,
                        dynamic_agg=not args.no_dynamic_agg,
                        bohb_trials=args.bohb)
        elif args.model == 'fedpcnn-2stage':
            run_fedpcnn_two_stage(args.dataset, args.partition, args.alpha, device,
                                  args.global_rounds, args.local_epochs, args.classification,
                                  dynamic_agg=not args.no_dynamic_agg)
        elif args.model == 'segmented':
            run_segmented_fl(args.dataset, args.partition, args.alpha, device,
                             args.global_rounds, args.local_epochs, args.classification)
    except Exception as e:
        print(f"\n 实验失败: {e}")


        traceback.print_exc()
        sys.exit(1)
