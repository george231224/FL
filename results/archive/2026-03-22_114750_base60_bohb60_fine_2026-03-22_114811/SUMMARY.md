# Experiment Summary

- Archived at: 2026-03-22 11:48:11
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_base60_bohb60_fine.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-22 11:47:50

## Metrics

- Accuracy: 80.5802
- Precision: 85.1258
- Recall: 80.5802
- F1-Score: 81.8142
- Macro-Precision: 60.1161
- Macro-Recall: 68.2044
- Macro-F1: 61.9166
- FAR: 6.3252

## Params

- num_devices: 10
- global_rounds: 60
- local_epochs: 5
- batch_size: 256
- lr: 0.005
- classifier: CNN+XGBoost(门限=0.29)
- normal_threshold: 0.285
- threshold_start: 0.26
- threshold_end: 0.34
- threshold_step: 0.005
- threshold_lambda: 5.0
- exp_tag: base60_bohb60_fine
- pretrained_model_path: ./results/models/FedPCNN_UNSW-NB15_non-iid_multi_base60_model.pt
- bohb_best_params: {'n_estimators': 189, 'max_depth': 8, 'learning_rate': 0.09605266805798604, 'subsample': 0.6708475831209393, 'colsample_bytree': 0.5057993146070927, 'min_child_weight': 1, 'gamma': 0.021039532319094396, 'reg_alpha': 0.06304951484758202, 'reg_lambda': 0.0012761913870533977}
- bohb_best_cv_f1: 61.38

## Notes

RTX4090 base60 backbone with BOHB 60 trials and fine threshold search 0.26-0.34 step 0.005

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb60_fine_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb60_fine_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb60_fine_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb60_fine_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb60_fine_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb60_fine_model.pt
