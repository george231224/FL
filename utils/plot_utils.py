"""
可视化工具模块
包含：
  - plot_loss_curves       训练 & 验证损失曲线
  - plot_feature_importance 特征重要性条形图
  - plot_confusion_matrix   混淆矩阵热力图
  - plot_model_comparison   多模型指标对比柱状图
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')           # 非交互后端，兼容无显示器环境
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

# ─── 全局字体设置（支持中文）─────────────────────────────────────────────────
rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 训练 & 验证损失曲线
# ══════════════════════════════════════════════════════════════════════════════
def plot_loss_curves(
    train_loss: list,
    val_loss: list = None,
    title: str = "训练损失曲线",
    xlabel: str = "全局轮次",
    ylabel: str = "平均损失",
    train_label: str = "训练损失",
    val_label: str = "验证损失",
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    绘制训练（和可选的验证）损失曲线。

    参数:
        train_loss  -- 训练损失列表，每个元素对应一个全局轮次
        val_loss    -- 验证损失列表（可选）；长度可与 train_loss 不同
                       （例如每 eval_interval 轮才有一个验证值）
        title       -- 图表标题
        xlabel      -- x 轴标签
        ylabel      -- y 轴标签
        save_path   -- 保存路径（None 则自动存入 ./results/plots/）
        show        -- 是否调用 plt.show()

    返回:
        保存的文件路径
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    rounds = list(range(1, len(train_loss) + 1))
    ax.plot(rounds, train_loss, 'b-o', markersize=4,
            linewidth=1.8, label=train_label, alpha=0.9)

    if val_loss and len(val_loss) > 0:
        # val_loss 可能每隔 eval_interval 轮才有一个点
        interval = max(1, len(train_loss) // len(val_loss))
        val_rounds = [interval * (i + 1) for i in range(len(val_loss))]
        ax.plot(val_rounds, val_loss, 'r--s', markersize=5,
                linewidth=1.8, label=val_label, alpha=0.9)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()

    if save_path is None:
        _ensure_dir('./results/plots')
        safe_title = title.replace(' ', '_').replace('/', '-')
        save_path = f'./results/plots/{safe_title}.png'

    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"[图表] 损失曲线已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 2. 特征重要性
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    title: str = "特征重要性",
    top_n: int = 20,
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    绘制特征重要性水平条形图（取 top_n 个特征）。

    参数:
        feature_names -- 特征名称列表
        importances   -- 对应的重要性分数（与 feature_names 等长）
        title         -- 图表标题
        top_n         -- 展示前 N 个最重要的特征
        save_path     -- 保存路径
        show          -- 是否调用 plt.show()

    返回:
        保存的文件路径
    """
    importances = np.asarray(importances)
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]
    bars = ax.barh(range(top_n), top_vals[::-1], color=colors[::-1],
                   edgecolor='white', height=0.72)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("重要性分数", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(bar.get_width() + max(top_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_xlim(0, max(top_vals) * 1.15)
    fig.tight_layout()

    if save_path is None:
        _ensure_dir('./results/plots')
        safe_title = title.replace(' ', '_').replace('/', '-')
        save_path = f'./results/plots/{safe_title}.png'

    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"[图表] 特征重要性已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 3. 混淆矩阵
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    title: str = "混淆矩阵",
    normalize: bool = False,
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    绘制混淆矩阵热力图。

    参数:
        y_true      -- 真实标签
        y_pred      -- 预测标签
        class_names -- 类别名称列表（None 则用数字标注）
        title       -- 图表标题
        normalize   -- True=格子显示行归一化比例；False=格子显示原始计数
                       颜色始终使用行归一化，避免不均衡类别使色阶失真
        save_path   -- 保存路径
        show        -- 是否调用 plt.show()

    返回:
        保存的文件路径
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    cm = sk_cm(y_true, y_pred)
    n_classes = cm.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    # 行归一化：每行除以该行真实样本总数，值域 [0,1]
    # 无论 normalize 参数如何，颜色始终基于行归一化，避免大类主导色阶
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / (row_sums + 1e-9)

    if normalize:
        # 格子内显示归一化比例（如 0.96）
        cm_text = cm_norm
        fmt = '.2f'
    else:
        # 格子内显示原始计数，颜色仍用行归一化
        cm_text = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(max(6, n_classes * 1.1), max(5, n_classes * 0.95)))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label('召回率 (行归一化)', fontsize=9)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=40, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("预测标签", fontsize=11)
    ax.set_ylabel("真实标签", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

    thresh = 0.5  # 颜色阈值（行归一化空间）
    for i in range(n_classes):
        for j in range(n_classes):
            color_val = cm_norm[i, j]   # 用于判断字体颜色
            text_val  = cm_text[i, j]   # 实际显示的数值
            text = f'{int(text_val)}' if fmt == 'd' else f'{text_val:.2f}'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=9,
                    color='white' if color_val > thresh else 'black')

    fig.tight_layout()

    if save_path is None:
        _ensure_dir('./results/plots')
        safe_title = title.replace(' ', '_').replace('/', '-')
        save_path = f'./results/plots/{safe_title}.png'

    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"[图表] 混淆矩阵已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 4. 多模型指标对比柱状图
# ══════════════════════════════════════════════════════════════════════════════
def plot_model_comparison(
    results: dict,
    metrics: list = None,
    title: str = "不同模型在 NSL-KDD 数据集上的指标对比",
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    绘制多模型、多指标对比分组柱状图。

    参数:
        results -- dict，格式如下:
                   {
                     'FedPCNN (IID)':     {'Accuracy': 95.1, 'F1-Score': 94.8, ...},
                     'SegmentedFL (IID)': {'Accuracy': 93.4, 'F1-Score': 92.0, ...},
                     ...
                   }
        metrics -- 要展示的指标列表（None 则取第一个模型的所有 key）
        title   -- 图表标题
        save_path -- 保存路径
        show      -- 是否调用 plt.show()

    返回:
        保存的文件路径
    """
    model_names = list(results.keys())
    if metrics is None:
        metrics = list(next(iter(results.values())).keys())

    n_models = len(model_names)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.75 / n_models          # 每个模型柱宽
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    # 配色：从 tab10 调色板取色
    palette = plt.cm.tab10(np.linspace(0, 0.9, n_models))

    fig, ax = plt.subplots(figsize=(max(9, n_metrics * 1.8), 6))

    for idx, (model_name, color, offset) in enumerate(zip(model_names, palette, offsets)):
        vals = [results[model_name].get(m, 0.0) for m in metrics]
        bars = ax.bar(x + offset, vals, width,
                      label=model_name, color=color,
                      edgecolor='white', linewidth=0.6, alpha=0.88)
        # 在柱顶标注数值
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7.5,
                    rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("指标值 (%)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.legend(loc='upper right', fontsize=9, framealpha=0.85)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    if save_path is None:
        _ensure_dir('./results/plots')
        safe_title = title.replace(' ', '_').replace('/', '-')
        save_path = f'./results/plots/{safe_title}.png'

    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"[图表] 模型对比已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 5. 逐类别指标水平条形图（论文风格）
# ══════════════════════════════════════════════════════════════════════════════
def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    title: str = "逐类别性能指标",
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    绘制每个类别的 Accuracy / Precision / Recall / F1-Score 水平条形图。
    类似论文中按类别展示的横向柱状图。

    参数:
        y_true      -- 真实标签 (N,)
        y_pred      -- 预测标签 (N,)
        class_names -- 类别名称列表（None 则用数字）
        title       -- 图表标题
        save_path   -- 保存路径
        show        -- 是否调用 plt.show()

    返回:
        保存的文件路径
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix as sk_cm

    labels = sorted(set(y_true) | set(y_pred))
    n_classes = len(labels)
    if class_names is None:
        class_names = [str(i) for i in labels]

    # 计算每类 Precision / Recall / F1
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0) * 100
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0) * 100

    # 每类 Accuracy = (TP + TN) / Total
    cm = sk_cm(y_true, y_pred, labels=labels)
    total = cm.sum()
    per_class_acc = np.zeros(n_classes)
    for i in range(n_classes):
        tp = cm[i, i]
        tn = total - cm[i, :].sum() - cm[:, i].sum() + tp
        per_class_acc[i] = (tp + tn) / total * 100

    # 绘图：竖向分组柱状图
    metric_names = ['F1-Score', 'Recall', 'Precision', 'Accuracy']
    metric_data = [f1, rec, prec, per_class_acc]
    colors = ['#F4C542', '#A8D8EA', '#F4A460', '#87CEEB']

    fig, ax = plt.subplots(figsize=(max(10, n_classes * 1.5), 6))

    bar_width = 0.18
    x_pos = np.arange(n_classes)

    for idx, (mname, mdata, color) in enumerate(zip(metric_names, metric_data, colors)):
        offset = (idx - 1.5) * bar_width
        ax.bar(x_pos + offset, mdata, bar_width,
               label=mname, color=color, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, fontsize=9, rotation=30, ha='right')
    ax.set_ylabel('%', fontsize=11)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g%%'))
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.85)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    if save_path is None:
        _ensure_dir('./results/plots')
        safe_title = title.replace(' ', '_').replace('/', '-')
        save_path = f'./results/plots/{safe_title}.png'

    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"[图表] 逐类别指标已保存: {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 6. 多模型指标对比水平条形图（论文风格）
# ══════════════════════════════════════════════════════════════════════════════
def plot_model_comparison_horizontal(
    results: dict,
    metrics: list = None,
    title: str = "模型性能对比",
    save_path: str = None,
    show: bool = False,
) -> str:
    """
    绘制多模型多指标水平条形图（论文风格：指标在 y 轴，模型用颜色区分）。

    参数:
        results -- dict，格式:
                   {
                     'Our Model': {'Accuracy': 95.1, 'Precision': 94.8, ...},
                     'FedAVG':    {'Accuracy': 93.4, 'Precision': 92.0, ...},
                     ...
                   }
        metrics -- 要展示的指标列表（None 则取第一个模型的所有 key）
        title   -- 图表标题
        save_path -- 保存路径
        show      -- 是否调用 plt.show()

    返回:
        保存的文件路径
    """
    model_names = list(results.keys())
    if metrics is None:
        metrics = list(next(iter(results.values())).keys())

    n_models = len(model_names)
    n_metrics = len(metrics)
    colors = ['#F4C542', '#C0C0C0', '#F4A460', '#87CEEB'][:n_models]

    x_pos = np.arange(n_metrics)
    bar_width = 0.75 / n_models

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 2.2), 6))

    for idx, (model_name, color) in enumerate(zip(model_names, colors)):
        offset = (idx - (n_models - 1) / 2) * bar_width
        vals = [results[model_name].get(m, 0.0) for m in metrics]
        bars = ax.bar(x_pos + offset, vals, bar_width,
                      label=model_name, color=color, edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel('%', fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g%%'))
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.85)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    if save_path is None:
        _ensure_dir('./results/plots')
        safe_title = title.replace(' ', '_').replace('/', '-')
        save_path = f'./results/plots/{safe_title}.png'

    _ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"[图表] 模型对比(水平)已保存: {save_path}")
    return save_path
