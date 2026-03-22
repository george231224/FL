# Experiment Summary

- Archived at: 2026-03-22 20:18:12
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_base60_bohb_thr0270.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-22 20:16:19

## Metrics

- Accuracy: 80.7335
- Precision: 85.1824
- Recall: 80.7335
- F1-Score: 81.8906
- Macro-Precision: 59.9845
- Macro-Recall: 69.4796
- Macro-F1: 62.3569
- FAR: 6.1931

## Params

- num_devices: 10
- global_rounds: 60
- local_epochs: 5
- batch_size: 256
- lr: 0.005
- classifier: CNN+XGBoost(门限=0.27)
- normal_threshold: 0.27
- threshold_start: 0.27
- threshold_end: 0.27
- threshold_step: 0.005
- threshold_lambda: 5.0
- threshold_selector: far_cap
- threshold_far_cap: 100.0
- bohb_cv_folds: 3
- feature_order: mrmr
- exp_tag: base60_bohb_thr0270
- pretrained_model_path: ./results/models/FedPCNN_UNSW-NB15_non-iid_multi_base60_model.pt
- bohb_best_params: {'n_estimators': 145, 'max_depth': 8, 'learning_rate': 0.1446504066107025, 'subsample': 0.8054549617551995, 'colsample_bytree': 0.7039918830643546, 'min_child_weight': 6, 'gamma': 0.6257083288885382, 'reg_alpha': 0.5819468919452753, 'reg_lambda': 0.024915807012636793}
- bohb_best_cv_f1: 60.91

## Notes

RTX4090 60轮主干复用 + BOHB 30 trials + 单点门限验证0.270; 在0.275和0.265之间补齐缺失点，检验平衡性

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0270_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0270_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0270_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0270_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0270_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0270_model.pt
