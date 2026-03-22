# Experiment Summary

- Archived at: 2026-03-22 15:28:58
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_base60_bohb_thr0275.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-22 15:28:26

## Metrics

- Accuracy: 80.7044
- Precision: 85.1887
- Recall: 80.7044
- F1-Score: 81.8795
- Macro-Precision: 59.9551
- Macro-Recall: 69.5112
- Macro-F1: 62.3579
- FAR: 6.2608

## Params

- num_devices: 10
- global_rounds: 60
- local_epochs: 5
- batch_size: 256
- lr: 0.005
- classifier: CNN+XGBoost(门限=0.28)
- normal_threshold: 0.275
- threshold_start: 0.275
- threshold_end: 0.275
- threshold_step: 0.005
- threshold_lambda: 5.0
- bohb_cv_folds: 3
- exp_tag: base60_bohb_thr0275
- pretrained_model_path: ./results/models/FedPCNN_UNSW-NB15_non-iid_multi_base60_model.pt
- bohb_best_params: {'n_estimators': 145, 'max_depth': 8, 'learning_rate': 0.1446504066107025, 'subsample': 0.8054549617551995, 'colsample_bytree': 0.7039918830643546, 'min_child_weight': 6, 'gamma': 0.6257083288885382, 'reg_alpha': 0.5819468919452753, 'reg_lambda': 0.024915807012636793}
- bohb_best_cv_f1: 60.91

## Notes

RTX4090 60轮主干复用 + BOHB 30 trials + 显式门限0.275; 结论：Accuracy/FAR更优，Macro-F1小幅回落

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0275_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0275_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0275_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0275_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0275_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_thr0275_model.pt
