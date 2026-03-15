"""
纯 XGBoost 基线测试 —— 不经过 CNN，直接在原始特征上训练
目的：确立数据集的真实上限，判断 CNN 是否在破坏特征
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import torch

from data_preprocessing import UNSWNB15Preprocessor

def _xgb_tree_method():
    return 'gpu_hist' if torch.cuda.is_available() else 'hist'

# ── 1. 加载数据（与 main.py 完全一致的预处理流程）──
print("=" * 60)
print("纯 XGBoost 基线测试 (无 CNN)")
print("=" * 60)

preprocessor = UNSWNB15Preprocessor(classification='multi')
X_train_full, y_train_full, X_test, y_test = preprocessor.load_and_preprocess()

# 与 main.py 一致：从训练集切出 20% 做验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

class_names = [c.capitalize() for c in preprocessor.label_encoder.classes_]
n_classes = len(np.unique(y_train))

print(f"\n训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
print(f"特征维度: {X_train.shape[1]}, 类别数: {n_classes}")

# ── 2. 训练 XGBoost ──
print("\n" + "=" * 60)
print("训练 XGBoost (原始特征, balanced sample_weight)")
print("=" * 60)

sample_weights = compute_sample_weight('balanced', y_train)

model = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.7,
    min_child_weight=3,
    gamma=0.5,
    reg_alpha=0.001,
    reg_lambda=10,
    objective='multi:softprob',
    num_class=n_classes,
    tree_method=_xgb_tree_method(),
    eval_metric='mlogloss',
    random_state=42,
    verbosity=1,
)

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

# ── 3. 评估 ──
print("\n" + "=" * 60)
print("测试集评估")
print("=" * 60)

y_pred = model.predict(X_test)

print("\n逐类别 Recall:")
report = classification_report(y_test, y_pred, target_names=class_names, digits=4, output_dict=True)
for i, name in enumerate(class_names):
    r = report[name]
    print(f"  类别 {i} ({name:15s}): Recall={r['recall']*100:.1f}%  Precision={r['precision']*100:.1f}%  F1={r['f1-score']*100:.1f}%")

macro = report['macro avg']
weighted = report['weighted avg']
acc = report['accuracy']

print(f"\n  Accuracy:        {acc*100:.2f}%")
print(f"  Macro-Precision: {macro['precision']*100:.2f}%")
print(f"  Macro-Recall:    {macro['recall']*100:.2f}%")
print(f"  Macro-F1:        {macro['f1-score']*100:.2f}%")
print(f"  Weighted-F1:     {weighted['f1-score']*100:.2f}%")

# FAR = FP_normal / (FP_normal + TN_normal)
cm = confusion_matrix(y_test, y_pred)
normal_idx = 0
fp_normal = cm[:, normal_idx].sum() - cm[normal_idx, normal_idx]
tn_normal = cm.sum() - cm[normal_idx, :].sum() - cm[:, normal_idx].sum() + cm[normal_idx, normal_idx]
far = fp_normal / (fp_normal + tn_normal) if (fp_normal + tn_normal) > 0 else 0
print(f"  FAR:             {far*100:.2f}%")

print("\n" + "=" * 60)
print("对比参考: CNN+XGBoost 基线 Macro-F1 = 53.62%")
print("=" * 60)
