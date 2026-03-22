# Experiment Summary

- Archived at: 2026-03-22 14:22:53
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_base60_bohb_thr0280.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-22 14:22:26

## Metrics

- Accuracy: 80.6908
- Precision: 85.2069
- Recall: 80.6908
- F1-Score: 81.8816
- Macro-Precision: 59.9445
- Macro-Recall: 69.5603
- Macro-F1: 62.3738
- FAR: 6.3073

## Params

- num_devices: 10
- global_rounds: 60
- local_epochs: 5
- batch_size: 256
- lr: 0.005
- classifier: CNN+XGBoost(门限=0.28)
- normal_threshold: 0.28
- threshold_start: 0.28
- threshold_end: 0.28
- threshold_step: 0.005
- threshold_lambda: 5.0
- bohb_cv_folds: 3
- exp_tag: base60_bohb_thr0280
- pretrained_model_path: ./results/models/FedPCNN_UNSW-NB15_non-iid_multi_base60_model.pt
- bohb_best_params: {'n_estimators': 145, 'max_depth': 8, 'learning_rate': 0.1446504066107025, 'subsample': 0.8054549617551995, 'colsample_bytree': 0.7039918830643546, 'min_child_weight': 6, 'gamma': 0.6257083288885382, 'reg_alpha': 0.5819468919452753, 'reg_lambda': 0.024915807012636793}
- bohb_best_cv_f1: 60.91

## Notes

RTX4090 60轮主干复用 + BOHB 30 trials + 显式门限0.280; 结论：Macro-F1基本持平，FAR显著下降

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0280_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0280_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0280_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0280_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0280_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0280_model.pt
