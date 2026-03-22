# Experiment Summary

- Archived at: 2026-03-22 13:19:57
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_base60_bohb5cv_fine.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-22 13:17:39

## Metrics

- Accuracy: 80.5336
- Precision: 84.7277
- Recall: 80.5336
- F1-Score: 81.7085
- Macro-Precision: 59.0584
- Macro-Recall: 68.7916
- Macro-F1: 61.7564
- FAR: 6.7491

## Params

- num_devices: 10
- global_rounds: 60
- local_epochs: 5
- batch_size: 256
- lr: 0.005
- classifier: CNN+XGBoost(门限=0.30)
- normal_threshold: 0.295
- threshold_start: 0.26
- threshold_end: 0.34
- threshold_step: 0.005
- threshold_lambda: 5.0
- bohb_cv_folds: 5
- exp_tag: base60_bohb5cv_fine
- pretrained_model_path: ./results/models/FedPCNN_UNSW-NB15_non-iid_multi_base60_model.pt
- bohb_best_params: {'n_estimators': 121, 'max_depth': 8, 'learning_rate': 0.18883910720881103, 'subsample': 0.6026168324121943, 'colsample_bytree': 0.5094917905649298, 'min_child_weight': 5, 'gamma': 0.553617064887679, 'reg_alpha': 1.8319741835711656, 'reg_lambda': 0.03261078452675138}
- bohb_best_cv_f1: 61.41

## Notes

RTX4090 60轮主干复用 + BOHB 30 trials + 5-fold Meta-CV + 门限搜索0.26-0.34 step0.005; 结论：较3-fold BOHB无提升

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb5cv_fine_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb5cv_fine_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb5cv_fine_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb5cv_fine_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb5cv_fine_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_base60_bohb5cv_fine_model.pt
