"""
基线模型对比实验
对比 LIBSVM / CNN / DNN / DBN-EGWO-KELM（集中式训练）

用法:
    python run_baselines.py --dataset NSL-KDD --classification binary
    python run_baselines.py --dataset UNSW-NB15 --classification binary
    python run_baselines.py --dataset UNSW-NB15 --classification binary
    python run_baselines.py --dataset CIC-IDS2017
    python run_baselines.py --dataset NSL-KDD --epochs 100
    python run_baselines.py --dataset NSL-KDD --skip-dbn   # 跳过 DBN-EGWO-KELM
"""

import argparse
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from utils.result_logger import ResultLogger


# ── 工具函数 ────────────────────────────────────────────────

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_far(y_true, y_pred):
    """与 FedPCNN 一致的 FAR = (FPR + FNR) / 2，class 0 = Normal，支持多分类"""
    cm = confusion_matrix(y_true, y_pred)
    normal_total = cm[0, :].sum()
    attack_total = cm[1:, :].sum()
    FPR = cm[0, 1:].sum() / normal_total if normal_total > 0 else 0.0
    FNR = cm[1:, 0].sum() / attack_total if attack_total > 0 else 0.0
    return (FPR + FNR) / 2 * 100


def build_metrics(y_true, y_pred):
    """统一指标计算，与 FedPCNN 保持一致"""
    return {
        'Accuracy':  accuracy_score(y_true, y_pred) * 100,
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'Recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'F1-Score':  f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'FAR':       compute_far(y_true, y_pred),
    }


def pad_to_square(X):
    """将特征填充到 side×side（与 FedPCNN preprocess_data 一致）"""
    n_samples, n_features = X.shape
    side = int(np.ceil(np.sqrt(n_features)))
    pad_size = side * side - n_features
    if pad_size > 0:
        X = np.pad(X, ((0, 0), (0, pad_size)), mode='constant')
    return X, side


def print_metrics(model_name, metrics):
    print(f"\n{'='*50}")
    print(f"  {model_name} 评估结果")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.2f}%")


# ── 各基线模型 ───────────────────────────────────────────────

def run_libsvm(X_train, y_train, X_test, y_test):
    from models.baseline.traditional_models import LIBSVM

    print("\n" + "="*50)
    print("  运行 LIBSVM (RBF kernel)")
    print("="*50)

    model = LIBSVM(kernel='rbf', C=1.0, gamma='scale')
    print("  训练中")
    model.train(X_train, y_train)

    y_pred = model.model.predict(X_test)
    return build_metrics(y_test, y_pred)


def run_cnn(X_train, y_train, X_test, y_test, device, epochs):
    from models.baseline.deep_models import CNN, DeepModelTrainer

    print("\n" + "="*50)
    print("  运行 CNN（集中式）")
    print("="*50)

    # 填充特征到 side×side（CNN 内部 view 需要）
    X_train_pad, _ = pad_to_square(X_train)
    X_test_pad, _  = pad_to_square(X_test)

    input_size  = X_train_pad.shape[1]
    num_classes = len(np.unique(y_train))

    model   = CNN(input_size=input_size, num_classes=num_classes)
    trainer = DeepModelTrainer(model, device=device)

    print(f"  训练 {epochs} 轮...")
    trainer.train(X_train_pad, y_train, epochs=epochs, batch_size=64, lr=0.001)

    # 预测
    X_test_t = torch.FloatTensor(X_test_pad).to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).argmax(1).cpu().numpy()

    return build_metrics(y_test, y_pred)


def run_dnn(X_train, y_train, X_test, y_test, device, epochs):
    from models.baseline.deep_models import DNN, DeepModelTrainer

    print("\n" + "="*50)
    print("  运行 DNN（集中式）")
    print("="*50)

    input_size  = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model   = DNN(input_size=input_size, num_classes=num_classes)
    trainer = DeepModelTrainer(model, device=device)

    print(f"  训练 {epochs} 轮...")
    trainer.train(X_train, y_train, epochs=epochs, batch_size=64, lr=0.001)

    X_test_t = torch.FloatTensor(X_test).to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).argmax(1).cpu().numpy()

    return build_metrics(y_test, y_pred)


def run_dbn_egwo_kelm(X_train, y_train, X_test, y_test, device):
    from models.baseline.dbn_egwo_kelm import DBN_EGWO_KELM

    print("\n" + "="*50)
    print("  运行 DBN-EGWO-KELM")
    print("="*50)

    model = DBN_EGWO_KELM(device=device)
    print("  训练中（DBN预训练 + EGWO优化 + KELM求解）...")
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    return build_metrics(y_test, y_pred)


# ── 主流程 ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='基线模型对比实验')
    parser.add_argument('--dataset',  type=str, default='NSL-KDD',
                        choices=['NSL-KDD', 'UNSW-NB15', 'CIC-IDS2017'])
    parser.add_argument('--epochs',   type=int, default=50,
                        help='CNN/DNN 训练轮次（默认50）')
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--no-cuda',  action='store_true')
    parser.add_argument('--skip-svm', action='store_true',
                        help='跳过 LIBSVM（数据量大时训练慢）')
    parser.add_argument('--skip-dbn', action='store_true',
                        help='跳过 DBN-EGWO-KELM（优化搜索耗时较长）')
    parser.add_argument('--classification', type=str, choices=['binary', 'multi'],
                        default='binary',
                        help='分类模式: binary 或 multi ')
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda'

    print("="*60)
    print(f"  基线模型对比实验")
    print(f"  数据集: {args.dataset}  |  设备: {device}  |  Epochs: {args.epochs}")
    print("="*60)

    # ── 数据加载 ────────────────────────────────────────────
    print(f"\n加载数据集: {args.dataset}  分类模式: {args.classification}")
    if args.dataset == 'NSL-KDD':
        from data_preprocessing import NSLKDDPreprocessor
        preprocessor = NSLKDDPreprocessor(classification=args.classification)
    elif args.dataset == 'UNSW-NB15':
        from data_preprocessing import UNSWNB15Preprocessor
        preprocessor = UNSWNB15Preprocessor(classification=args.classification)
    elif args.dataset == 'CIC-IDS2017':
        from data_preprocessing import CICIDS2017Preprocessor
        preprocessor = CICIDS2017Preprocessor(
            classification=args.classification,
            sample_size=300000,
        )
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")

    X_train, y_train, X_test, y_test = preprocessor.load_and_preprocess()
    print(f"  训练集: {X_train.shape}，测试集: {X_test.shape}")
    print(f"  类别分布（训练）: { {c: int((y_train==c).sum()) for c in np.unique(y_train)} }")

    logger  = ResultLogger(results_dir='./results/baseline')
    results = {}

    # ── 运行各基线 ───────────────────────────────────────────
    if not args.skip_svm:
        results['LIBSVM'] = run_libsvm(X_train, y_train, X_test, y_test)
        print_metrics('LIBSVM', results['LIBSVM'])
        logger.save_result(
            dataset=args.dataset, model_name='libsvm',
            partition='centralized', alpha=0.0,
            metrics=results['LIBSVM'],
            params={'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
        )

    results['CNN'] = run_cnn(X_train, y_train, X_test, y_test, device, args.epochs)
    print_metrics('CNN', results['CNN'])
    logger.save_result(
        dataset=args.dataset, model_name='cnn',
        partition='centralized', alpha=0.0,
        metrics=results['CNN'],
        params={'epochs': args.epochs, 'batch_size': 64, 'lr': 0.001},
    )

    results['DNN'] = run_dnn(X_train, y_train, X_test, y_test, device, args.epochs)
    print_metrics('DNN', results['DNN'])
    logger.save_result(
        dataset=args.dataset, model_name='dnn',
        partition='centralized', alpha=0.0,
        metrics=results['DNN'],
        params={'epochs': args.epochs, 'batch_size': 64, 'lr': 0.001},
    )

    if not args.skip_dbn:
        results['DBN-EGWO-KELM'] = run_dbn_egwo_kelm(X_train, y_train, X_test, y_test, device)
        print_metrics('DBN-EGWO-KELM', results['DBN-EGWO-KELM'])
        logger.save_result(
            dataset=args.dataset, model_name='dbn_egwo_kelm',
            partition='centralized', alpha=0.0,
            metrics=results['DBN-EGWO-KELM'],
            params={
                'dbn_layers': [256, 128, 64],
                'dbn_epochs': 50,
                'egwo_wolves': 20,
                'egwo_iters': 30,
            },
        )

    # ── 汇总对比 ─────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"  {args.dataset} 基线模型对比汇总")
    print("="*70)
    header = f"{'模型':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'FAR':>8}"
    print(header)
    print("-"*70)
    for name, m in results.items():
        print(f"{name:<10} {m['Accuracy']:>9.2f}%  {m['Precision']:>9.2f}%  "
              f"{m['Recall']:>9.2f}%  {m['F1-Score']:>9.2f}%  {m['FAR']:>7.2f}%")
    print("="*70)
    print("\n结果已追加到 ./results/baseline/summary.csv")


if __name__ == '__main__':
    main()
