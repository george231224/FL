# Experiment Summary

- Archived at: 2026-03-22 11:23:32
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_base60_bohb_fine.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-22 11:23:12

## Metrics

- Accuracy: 80.5879
- Precision: 85.2842
- Recall: 80.5879
- F1-Score: 81.8592
- Macro-Precision: 59.8510
- Macro-Recall: 69.7387
- Macro-F1: 62.3941
- FAR: 6.5745

## Params

- num_devices: 10
- global_rounds: 60
- local_epochs: 5
- batch_size: 256
- lr: 0.005
- classifier: CNN+XGBoost(门限=0.30)
- normal_threshold: 0.3
- threshold_start: 0.26
- threshold_end: 0.34
- threshold_step: 0.005
- threshold_lambda: 5.0
- exp_tag: base60_bohb_fine
- pretrained_model_path: ./results/models/FedPCNN_UNSW-NB15_non-iid_multi_base60_model.pt
- bohb_best_params: {'n_estimators': 145, 'max_depth': 8, 'learning_rate': 0.1446504066107025, 'subsample': 0.8054549617551995, 'colsample_bytree': 0.7039918830643546, 'min_child_weight': 6, 'gamma': 0.6257083288885382, 'reg_alpha': 0.5819468919452753, 'reg_lambda': 0.024915807012636793}
- bohb_best_cv_f1: 60.91

## Notes

RTX4090 base60 backbone with BOHB and fine threshold search 0.26-0.34 step 0.005

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_fine_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_fine_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_fine_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_fine_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_fine_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb_fine_model.pt
