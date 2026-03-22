# Experiment Summary

- Archived at: 2026-03-22 11:00:13
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_bohb50_fine.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-22 11:00:02

## Metrics

- Accuracy: 80.3667
- Precision: 85.4544
- Recall: 80.3667
- F1-Score: 81.7181
- Macro-Precision: 59.2303
- Macro-Recall: 70.0004
- Macro-F1: 61.6719
- FAR: 6.5155

## Params

- num_devices: 10
- global_rounds: 50
- local_epochs: 5
- batch_size: 256
- lr: 0.005
- classifier: CNN+XGBoost(门限=0.28)
- normal_threshold: 0.275
- threshold_start: 0.26
- threshold_end: 0.34
- threshold_step: 0.005
- threshold_lambda: 5.0
- exp_tag: bohb50_fine
- pretrained_model_path: ./results/models/FedPCNN_UNSW-NB15_non-iid_multi_bohb50_model.pt
- bohb_best_params: {'n_estimators': 113, 'max_depth': 8, 'learning_rate': 0.1374213587196571, 'subsample': 0.7245042188206406, 'colsample_bytree': 0.5277193751821014, 'min_child_weight': 4, 'gamma': 0.6326598847663321, 'reg_alpha': 1.8444788537125651, 'reg_lambda': 0.019433174506300516}
- bohb_best_cv_f1: 60.95

## Notes

RTX4090 fine threshold search from bohb50 backbone; threshold range 0.26-0.34 step 0.005

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_bohb50_fine_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_bohb50_fine_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_bohb50_fine_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_bohb50_fine_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_bohb50_fine_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_bohb50_fine_model.pt
