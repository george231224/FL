# Experiment Summary

- Archived at: 2026-03-22 19:57:33
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_base60_bohb_farcap630.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-22 19:56:31

## Metrics

- Accuracy: 80.7335
- Precision: 85.1457
- Recall: 80.7335
- F1-Score: 81.8713
- Macro-Precision: 59.9972
- Macro-Recall: 69.4136
- Macro-F1: 62.3358
- FAR: 6.1590

## Params

- num_devices: 10
- global_rounds: 60
- local_epochs: 5
- batch_size: 256
- lr: 0.005
- classifier: CNN+XGBoost(门限=0.265)
- normal_threshold: 0.265
- threshold_start: 0.26
- threshold_end: 0.34
- threshold_step: 0.005
- threshold_lambda: 5.0
- threshold_selector: far_cap
- threshold_far_cap: 6.3
- bohb_cv_folds: 3
- feature_order: mrmr
- exp_tag: base60_bohb_farcap630
- pretrained_model_path: ./results/models/FedPCNN_UNSW-NB15_non-iid_multi_base60_model.pt
- bohb_best_params: {'n_estimators': 145, 'max_depth': 8, 'learning_rate': 0.1446504066107025, 'subsample': 0.8054549617551995, 'colsample_bytree': 0.7039918830643546, 'min_child_weight': 6, 'gamma': 0.6257083288885382, 'reg_alpha': 0.5819468919452753, 'reg_lambda': 0.024915807012636793}
- bohb_best_cv_f1: 60.91

## Notes

RTX4090 60轮主干复用 + BOHB 30 trials + selector_v2 far_cap=6.30; 自动选中门限0.265，进一步压低 FAR

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_farcap630_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_farcap630_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_farcap630_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_farcap630_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_farcap630_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_farcap630_model.pt
