"""
从已保存的实验结果生成可视化图表
读取 results/summary.csv 和 results/*.json，输出对比图到 results/plots/
用法:
    python plot_results.py              # 生成所有图表
    python plot_results.py --dataset NSL-KDD     # 仅指定数据集
    python plot_results.py --model fedpcnn       # 仅指定模型
"""

import os
import json
import glob
import argparse
import pandas as pd
from utils.plot_utils import plot_model_comparison, plot_loss_curves

METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FAR']
PLOTS_DIR = './results/plots'
SUMMARY_CSV = './results/summary.csv'
RESULTS_DIR = './results'


def load_summary(filter_dataset=None, filter_model=None):
    """加载 summary.csv，返回 DataFrame"""
    if not os.path.exists(SUMMARY_CSV):
        print(f"[错误] 未找到 {SUMMARY_CSV}，请先运行实验（python run_experiments.py）")
        return None

    df = pd.read_csv(SUMMARY_CSV)

    # 去重：保留每个组合最新的一条
    key_cols = ['dataset', 'model', 'partition', 'alpha']
    key_cols = [c for c in key_cols if c in df.columns]
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').drop_duplicates(subset=key_cols, keep='last')

    if filter_dataset:
        df = df[df['dataset'] == filter_dataset]
    if filter_model:
        df = df[df['model'] == filter_model]

    return df


def build_label(row):
    """根据行数据生成展示标签"""
    model_map = {'fedpcnn': 'FedPCNN', 'segmented': 'SegmentedFL'}
    model = model_map.get(str(row.get('model', '')), str(row.get('model', '')))
    partition = str(row.get('partition', '')).upper()
    alpha = row.get('alpha', None)

    if partition == 'NON-IID' and pd.notna(alpha):
        return f"{model} ({partition}, α={alpha})"
    return f"{model} ({partition})"


def plot_by_dataset(df):
    """为每个数据集生成模型对比图"""
    for dataset in df['dataset'].unique():
        sub = df[df['dataset'] == dataset]
        results = {}
        for _, row in sub.iterrows():
            label = build_label(row)
            results[label] = {m: row[m] for m in METRICS if m in row and pd.notna(row[m])}

        if not results:
            continue

        save_path = os.path.join(PLOTS_DIR, f"comparison_{dataset.replace('-', '_')}.png")
        plot_model_comparison(
            results=results,
            metrics=METRICS,
            title=f"模型性能对比 ({dataset})",
            save_path=save_path,
        )


def plot_iid_vs_noniid(df):
    """为每个数据集×模型组合生成 IID vs Non-IID 对比图"""
    for dataset in df['dataset'].unique():
        for model in df['model'].unique():
            sub = df[(df['dataset'] == dataset) & (df['model'] == model)]
            if sub.empty:
                continue

            results = {}
            for _, row in sub.iterrows():
                label = build_label(row)
                results[label] = {m: row[m] for m in METRICS if m in row and pd.notna(row[m])}

            if len(results) < 2:
                continue

            model_label = 'FedPCNN' if model == 'fedpcnn' else 'SegmentedFL'
            save_path = os.path.join(
                PLOTS_DIR,
                f"iid_vs_noniid_{model}_{dataset.replace('-', '_')}.png"
            )
            plot_model_comparison(
                results=results,
                metrics=METRICS,
                title=f"{model_label} IID vs Non-IID ({dataset})",
                save_path=save_path,
            )


def plot_loss_from_json(filter_dataset=None, filter_model=None):
    """从 JSON 文件中提取训练损失并绘制损失曲线"""
    json_files = glob.glob(os.path.join(RESULTS_DIR, '*.json'))
    if not json_files:
        print("[提示] 未找到 JSON 结果文件，跳过损失曲线绘制")
        return

    for fpath in json_files:
        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        dataset = data.get('dataset', '')
        model   = data.get('model', '')
        partition = data.get('partition', '')
        alpha   = data.get('alpha', None)

        if filter_dataset and dataset != filter_dataset:
            continue
        if filter_model and model != filter_model:
            continue

        train_loss = data.get('train_loss', None)
        val_loss   = data.get('val_loss', None)

        if not train_loss:
            continue

        model_label = 'FedPCNN' if model == 'fedpcnn' else 'SegmentedFL'
        alpha_str = f"_a{alpha}" if partition == 'non-iid' and alpha is not None else ""
        tag = f"{model}_{dataset}_{partition}{alpha_str}"

        plot_loss_curves(
            train_loss=train_loss,
            val_loss=val_loss if val_loss else None,
            title=f"{model_label} 训练损失曲线 ({dataset} · {partition.upper()})",
            save_path=os.path.join(PLOTS_DIR, f"loss_{tag}.png"),
        )


def main():
    parser = argparse.ArgumentParser(description='生成联邦学习实验可视化图表')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['NSL-KDD', 'UNSW-NB15'],
                        help='仅绘制指定数据集的结果')
    parser.add_argument('--model', type=str, default=None,
                        choices=['fedpcnn', 'segmented'],
                        help='仅绘制指定模型的结果')
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_summary(filter_dataset=args.dataset, filter_model=args.model)
    if df is None or df.empty:
        print("[提示] 无结果数据可绘制")
        return

    print(f"\n共加载 {len(df)} 条实验记录")
    print(df[['dataset', 'model', 'partition', 'alpha'] + [m for m in METRICS if m in df.columns]].to_string(index=False))

    print("\n正在生成图表...")
    plot_by_dataset(df)
    plot_iid_vs_noniid(df)
    plot_loss_from_json(filter_dataset=args.dataset, filter_model=args.model)

    print(f"\n所有图表已保存到: {PLOTS_DIR}")


if __name__ == '__main__':
    main()
