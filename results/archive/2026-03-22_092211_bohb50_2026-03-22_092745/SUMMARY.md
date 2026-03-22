# Experiment Summary

- Archived at: 2026-03-22 09:27:45
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_bohb50.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-22 09:22:11

## Metrics

- Accuracy: 80.1979
- Precision: 85.5267
- Recall: 80.1979
- F1-Score: 81.6530
- Macro-Precision: 59.0923
- Macro-Recall: 70.1686
- Macro-F1: 61.6548
- FAR: 6.8922

## Params

- num_devices: 10
- global_rounds: 50
- local_epochs: 5
- batch_size: 256
- lr: 0.005
- classifier: CNN+XGBoost(门限=0.30)
- exp_tag: bohb50
- bohb_best_params: {'n_estimators': 113, 'max_depth': 8, 'learning_rate': 0.1374213587196571, 'subsample': 0.7245042188206406, 'colsample_bytree': 0.5277193751821014, 'min_child_weight': 4, 'gamma': 0.6326598847663321, 'reg_alpha': 1.8444788537125651, 'reg_lambda': 0.019433174506300516}
- bohb_best_cv_f1: 60.95

## Notes

RTX4090 remote bohb50, resumed + xgboost3 fix

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_bohb50_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_bohb50_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_bohb50_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_bohb50_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_bohb50_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_bohb50_model.pt
