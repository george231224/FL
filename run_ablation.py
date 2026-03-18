"""
消融实验: 验证动态聚合 & Focal Loss 组件有效性

实验组:
  1. 动态聚合 vs 标准FedAvg（固定 focal_gamma）
  2. Focal Loss γ 参数扫描（固定动态聚合）

对比指标: Accuracy, Macro-F1, 逐类别 Recall, FAR
输出: 终端表格 + results/ablation/ 下的 JSON 汇总
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from main import set_seed, get_device, get_dataset, partition_data


def run_fedpcnn_single(dataset_name, classification, partition_type, alpha,
                       device, global_rounds, dynamic_agg, focal_gamma,
                       seed=42):
    """运行单次 FedPCNN 实验，返回指标字典"""
    from models.fedpcnn import FedPCNN
    from data_preprocessing import apply_smote_per_client

    set_seed(seed)

    # [额外修复] get_dataset 现在返回 8 个值（含 n_continuous）
    X_train, y_train, X_test, y_test, n_classes, n_features, class_names, n_continuous = \
        get_dataset(dataset_name, classification)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    num_devices = 10
    client_data = partition_data(X_train, y_train, partition_type, num_devices, alpha)
    client_data_list = [client_data[i] for i in range(num_devices)]

    # pre-SMOTE class weights
    pre_smote_labels = np.concatenate([y for _, y in client_data_list])
    pre_smote_counts = np.maximum(
        np.bincount(pre_smote_labels, minlength=n_classes), 1
    ).astype(float)
    pre_smote_cw = 1.0 / pre_smote_counts
    min_cw = pre_smote_cw.min()
    cw_cap = 50.0 if n_classes > 5 else 15.0  # 与 main.py 对齐
    pre_smote_cw = np.clip(pre_smote_cw, 0, min_cw * cw_cap)
    pre_smote_cw = pre_smote_cw / pre_smote_cw.sum() * n_classes
    pre_smote_class_weights = torch.FloatTensor(pre_smote_cw)

    # SMOTE
    client_data_list = apply_smote_per_client(
        client_data_list, num_classes=n_classes,
        dataset_name=dataset_name, classification=classification,
        k_neighbors=5,
    )

    # [额外修复] 使用 1D 输入形状（与 main.py 对齐），传入 n_continuous
    input_shape = (1, n_features)

    fedpcnn = FedPCNN(
        num_devices=num_devices,
        num_classes=n_classes,
        input_shape=input_shape,
        n_continuous=n_continuous,
    )
    fedpcnn.device = device

    # 超参数（与 main.py 对齐）
    # 超参数与 main.py 严格对齐
    if dataset_name.upper().startswith('UNSW'):
        if n_classes == 2:
            hp_lr, hp_mu, hp_local_epochs, hp_gamma = 0.008, 0.08, 8, 0.5
        else:
            hp_lr, hp_mu, hp_local_epochs, hp_gamma = 0.005, 0.10, 5, 0.5
    elif dataset_name.upper().startswith('CIC'):
        if n_classes == 2:
            hp_lr, hp_mu, hp_local_epochs, hp_gamma = 0.008, 0.05, 8, 0.5
        else:
            hp_lr, hp_mu, hp_local_epochs, hp_gamma = 0.005, 0.05, 5, 0.5
    else:  # NSL-KDD
        if n_classes == 2:
            hp_lr, hp_mu, hp_local_epochs, hp_gamma = 0.008, 0.08, 8, 0.5
        else:
            hp_lr, hp_mu, hp_local_epochs, hp_gamma = 0.005, 0.05, 5, 0.6

    # [额外修复] 使用新的 dynamic_aggregation 参数（原来传 dynamic_aggregation 到旧接口会报错）
    train_loss, val_loss = fedpcnn.train(
        client_data=client_data_list,
        global_rounds=global_rounds,
        local_epochs=hp_local_epochs,
        client_fraction=1.0 if (partition_type != 'iid' or n_classes == 2) else 0.7,
        batch_size=256,  # 与 main.py 对齐
        lr=hp_lr, mu=hp_mu, gamma=hp_gamma,
        focal_gamma=focal_gamma,
        alpha=alpha,
        X_val=X_val, y_val=y_val,
        eval_interval=5,
        pre_smote_class_weights=pre_smote_class_weights,
        dynamic_aggregation=dynamic_agg,
    )

    # cRT
    if n_classes > 2:
        fedpcnn.classifier_retrain(X_train, y_train, epochs=10, lr=0.01)

    # logit bias
    logit_bias = fedpcnn.calibrate_thresholds(X_val, y_val)

    # CNN 评估
    metrics_cnn = fedpcnn.evaluate(X_test, y_test, logit_bias=logit_bias)

    # SVM 评估
    fedpcnn.train_svm(X_train, y_train, C=1.0, kernel='rbf')
    metrics_svm, _, _ = fedpcnn.evaluate_with_svm(X_test, y_test)

    # 选最优
    cnn_f1 = metrics_cnn.get('Macro-F1', 0)
    svm_f1 = metrics_svm.get('Macro-F1', 0)
    if cnn_f1 > svm_f1:
        metrics = metrics_cnn
        metrics['best_path'] = 'CNN'
    else:
        metrics = metrics_svm
        metrics['best_path'] = 'SVM'

    # 获取逐类别 recall
    from sklearn.metrics import recall_score
    # 重新获取预测用于逐类别 recall
    if metrics['best_path'] == 'CNN':
        X_proc, y_proc = fedpcnn.preprocess_data(X_test, y_test)
        ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_proc).to(device),
            torch.LongTensor(y_proc).to(device)
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
        bias_tensor = logit_bias.to(device) if logit_bias is not None else None
        fedpcnn.global_model.eval()
        preds = []
        with torch.no_grad():
            for bx, _ in loader:
                out = fedpcnn.global_model(bx)
                if bias_tensor is not None:
                    out = out + bias_tensor
                preds.extend(out.argmax(1).cpu().numpy())
        per_cls = recall_score(y_proc, preds, average=None, zero_division=0)
    else:
        features, labels = fedpcnn._extract_features_batch(X_test, y_test)
        features_scaled = fedpcnn.svm_scaler.transform(features)
        preds = fedpcnn.svm_classifier.predict(features_scaled)
        per_cls = recall_score(labels, preds, average=None, zero_division=0)

    metrics['per_class_recall'] = {int(i): round(float(v) * 100, 2)
                                    for i, v in enumerate(per_cls)}
    return metrics


def print_comparison_table(results, title):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

    # 表头
    keys_main = ['Accuracy', 'Macro-F1', 'F1-Score', 'FAR']
    header = f"{'配置':<30}"
    for k in keys_main:
        header += f"{k:>12}"
    header += f"{'路径':>6}"
    print(header)
    print("-" * 80)

    for label, m in results.items():
        row = f"{label:<30}"
        for k in keys_main:
            row += f"{m.get(k, 0):>11.2f}%"
        row += f"{m.get('best_path', ''):>6}"
        print(row)

    # 逐类别 Recall
    print("\n逐类别 Recall:")
    all_classes = set()
    for m in results.values():
        all_classes.update(m.get('per_class_recall', {}).keys())
    all_classes = sorted(all_classes)

    header2 = f"{'配置':<30}"
    for c in all_classes:
        header2 += f"{'类别'+str(c):>10}"
    print(header2)
    print("-" * 80)

    for label, m in results.items():
        row = f"{label:<30}"
        pcr = m.get('per_class_recall', {})
        for c in all_classes:
            row += f"{pcr.get(c, 0):>9.1f}%"
        print(row)

    print("=" * 80)


def run_aggregation_ablation(args):
    """实验 1: 动态聚合 vs FedAvg"""
    print("\n" + "#" * 80)
    print("# 实验 1: 动态聚合 vs 标准 FedAvg")
    print("#" * 80)

    results = {}

    # 动态聚合 (ON)
    print("\n>>> 运行: 动态聚合 ON <<<")
    m_on = run_fedpcnn_single(
        args.dataset, args.classification, args.partition, args.alpha,
        args.device, args.global_rounds, dynamic_agg=True,
        focal_gamma=args.focal_gamma, seed=args.seed,
    )
    results['动态聚合 (ON)'] = m_on

    # 标准 FedAvg (OFF)
    print("\n>>> 运行: 动态聚合 OFF (FedAvg) <<<")
    m_off = run_fedpcnn_single(
        args.dataset, args.classification, args.partition, args.alpha,
        args.device, args.global_rounds, dynamic_agg=False,
        focal_gamma=args.focal_gamma, seed=args.seed,
    )
    results['标准 FedAvg (OFF)'] = m_off

    print_comparison_table(results, f"动态聚合消融 ({args.dataset} · {args.partition} · {args.classification})")

    # 差值
    delta_f1 = m_on.get('Macro-F1', 0) - m_off.get('Macro-F1', 0)
    delta_acc = m_on.get('Accuracy', 0) - m_off.get('Accuracy', 0)
    delta_far = m_on.get('FAR', 0) - m_off.get('FAR', 0)
    print(f"\n差值 (ON - OFF): Accuracy={delta_acc:+.2f}%, Macro-F1={delta_f1:+.2f}%, FAR={delta_far:+.2f}%")
    if delta_f1 > 0:
        print("  → 动态聚合有正向作用")
    elif delta_f1 < -0.5:
        print("  → 动态聚合有负向作用，考虑禁用")
    else:
        print("  → 动态聚合影响不显著")

    return results


def run_focal_ablation(args):
    """实验 2: Focal Loss γ 参数扫描"""
    print("\n" + "#" * 80)
    print("# 实验 2: Focal Loss γ 参数扫描")
    print("#" * 80)

    gamma_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    results = {}

    for gamma in gamma_values:
        label = f"γ={gamma:.1f}" + (" (CE)" if gamma == 0 else "")
        print(f"\n>>> 运行: focal_gamma={gamma} <<<")
        m = run_fedpcnn_single(
            args.dataset, args.classification, args.partition, args.alpha,
            args.device, args.global_rounds, dynamic_agg=True,
            focal_gamma=gamma, seed=args.seed,
        )
        results[label] = m

    print_comparison_table(results, f"Focal Loss γ 消融 ({args.dataset} · {args.partition} · {args.classification})")

    # 找最优 gamma
    best_label = max(results, key=lambda k: results[k].get('Macro-F1', 0))
    print(f"\n最优配置: {best_label} (Macro-F1={results[best_label]['Macro-F1']:.2f}%)")

    # CE vs 最优 Focal 的差值
    ce_f1 = results.get('γ=0.0 (CE)', {}).get('Macro-F1', 0)
    best_f1 = results[best_label].get('Macro-F1', 0)
    print(f"CE→最优Focal: Macro-F1 {ce_f1:.2f}% → {best_f1:.2f}% ({best_f1-ce_f1:+.2f}%)")

    # 逐类别分析
    print("\n少数类改善分析:")
    ce_pcr = results.get('γ=0.0 (CE)', {}).get('per_class_recall', {})
    best_pcr = results[best_label].get('per_class_recall', {})
    for cls in sorted(set(ce_pcr.keys()) | set(best_pcr.keys())):
        ce_v = ce_pcr.get(cls, 0)
        best_v = best_pcr.get(cls, 0)
        delta = best_v - ce_v
        marker = " ←少数类改善" if delta > 2 else ""
        print(f"  类别 {cls}: {ce_v:.1f}% → {best_v:.1f}% ({delta:+.1f}%){marker}")

    return results


def main():
    parser = argparse.ArgumentParser(description='FedPCNN 组件消融实验')
    parser.add_argument('--experiment', type=str,
                        choices=['agg', 'focal', 'all'], default='all',
                        help='实验类型: agg=聚合对比, focal=Focal Loss扫描, all=全部')
    parser.add_argument('--dataset', type=str, default='NSL-KDD',
                        choices=['NSL-KDD', 'UNSW-NB15', 'CIC-IDS2017'])
    parser.add_argument('--classification', type=str, default='multi',
                        choices=['binary', 'multi'])
    parser.add_argument('--partition', type=str, default='iid',
                        choices=['iid', 'non-iid'])
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--global-rounds', type=int, default=50)
    parser.add_argument('--focal-gamma', type=float, default=1.0,
                        help='聚合对比实验中使用的 focal_gamma (默认1.0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cuda', action='store_true', default=False)

    args = parser.parse_args()

    if args.no_cuda:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"设备: {args.device}")
    print(f"数据集: {args.dataset}, 分类: {args.classification}")
    print(f"划分: {args.partition}, seed: {args.seed}")

    all_results = {}

    if args.experiment in ('agg', 'all'):
        all_results['aggregation'] = run_aggregation_ablation(args)

    if args.experiment in ('focal', 'all'):
        all_results['focal_loss'] = run_focal_ablation(args)

    # 保存结果
    out_dir = './results/ablation'
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.dataset}_{args.partition}_{args.classification}"
    out_path = os.path.join(out_dir, f"ablation_{tag}.json")

    # 序列化（处理 numpy 类型）
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"\n结果已保存至 {out_path}")


if __name__ == '__main__':
    main()
