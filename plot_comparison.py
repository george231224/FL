import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 1

# ============================================================
# 数据来源: results/*.json + summary.csv (最新最优结果)
# ============================================================

# --- 二分类 ---
binary_models = [
    'NSL-KDD\nIID',
    'NSL-KDD\nnon-IID',
    'UNSW-NB15\nIID',
    'UNSW-NB15\nnon-IID',
]
binary_metrics = {
    'Accuracy':  [94.30, 94.70, 90.61, 89.88],
    'Precision': [94.30, 94.71, 91.26, 90.86],
    'Recall':    [94.30, 94.70, 90.61, 89.88],
    'F1-Score':  [94.30, 94.70, 90.72, 90.02],
    'FAR':       [5.71,  5.29,  8.72,  9.11],
}

# --- 多分类 ---
multi_models = [
    'NSL-KDD\nIID',
    'NSL-KDD\nnon-IID',
    'UNSW-NB15\nIID',
    'UNSW-NB15\nnon-IID',
]
multi_metrics = {
    'Accuracy':       [94.89, 94.58, 73.04, 73.54],
    'Precision':      [96.21, 95.85, 81.50, 81.66],
    'Recall':         [94.89, 94.58, 73.04, 73.54],
    'F1-Score':       [95.31, 94.98, 75.97, 76.29],
    'Macro-F1':       [75.44, 76.07, 48.70, 49.44],
    'FAR':            [4.19,  4.59,  12.48, 12.59],
}


def plot_grouped_bar(models, metrics, title, save_path, figsize=(10, 5)):
    """绘制分组柱状图"""
    metric_names = list(metrics.keys())
    x = np.arange(len(models))
    n = len(metric_names)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4']

    for i, (name, values) in enumerate(metrics.items()):
        offset = (i - n / 2) * width + width / 2
        bars = ax.bar(x + offset, values, width=width, label=name, color=colors[i % len(colors)])
        # 在柱子上方标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Value (%)')
    ax.set_title(title, fontsize=13, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[保存] {save_path}")
    plt.close()


def plot_multi_separate(models, metrics, save_path, figsize=(12, 5)):
    """多分类: 主指标 + Macro-F1/FAR 分开画 (y轴范围不同)"""
    # 子图1: Accuracy, Precision, Recall, F1
    main_keys = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    main = {k: metrics[k] for k in main_keys}

    # 子图2: Macro-F1, FAR
    aux_keys = ['Macro-F1', 'FAR']
    aux = {k: metrics[k] for k in aux_keys}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})

    x = np.arange(len(models))
    colors1 = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    colors2 = ['#9C27B0', '#E91E63']

    # --- 子图1: 主指标 ---
    n1 = len(main_keys)
    w1 = 0.8 / n1
    for i, (name, values) in enumerate(main.items()):
        offset = (i - n1 / 2) * w1 + w1 / 2
        bars = ax1.bar(x + offset, values, width=w1, label=name, color=colors1[i])
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('Value (%)')
    ax1.set_title('Multi-class: Main Metrics', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.legend(frameon=False, fontsize=8)

    # --- 子图2: Macro-F1 & FAR ---
    n2 = len(aux_keys)
    w2 = 0.8 / n2
    for i, (name, values) in enumerate(aux.items()):
        offset = (i - n2 / 2) * w2 + w2 / 2
        bars = ax2.bar(x + offset, values, width=w2, label=name, color=colors2[i])
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylim(0, 90)
    ax2.set_ylabel('Value (%)')
    ax2.set_title('Multi-class: Macro-F1 & FAR', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[保存] {save_path}")
    plt.close()


if __name__ == '__main__':
    # 1. 二分类对比图
    plot_grouped_bar(
        binary_models, binary_metrics,
        title='FedPCNN Binary Classification Comparison',
        save_path='./results/plots/comparison_binary.png',
    )

    # 2. 多分类对比图 (分两部分)
    plot_multi_separate(
        multi_models, multi_metrics,
        save_path='./results/plots/comparison_multi.png',
    )

    # 3. 多分类全指标合并图
    plot_grouped_bar(
        multi_models, multi_metrics,
        title='FedPCNN Multi-class Classification Comparison',
        save_path='./results/plots/comparison_multi_all.png',
        figsize=(12, 5),
    )

    print("\n所有对比图已生成!")
