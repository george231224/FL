# Experiment Summary

- Archived at: 2026-03-22 18:03:38
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_feature20_mrmr.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-22 18:03:13

## Metrics

- Accuracy: 78.7504
- Precision: 86.1620
- Recall: 78.7504
- F1-Score: 80.9949
- Macro-Precision: 57.8394
- Macro-Recall: 74.0210
- Macro-F1: 60.3320
- FAR: 7.6400

## Params

- num_devices: 10
- global_rounds: 20
- local_epochs: 5
- batch_size: 256
- lr: 0.005
- classifier: CNN+XGBoost(门限=0.3)
- normal_threshold: 0.3
- threshold_start: 0.3
- threshold_end: 0.75
- threshold_step: 0.025
- threshold_lambda: 5.0
- threshold_selector: baseline_penalty
- threshold_far_cap: None
- bohb_cv_folds: 3
- feature_order: mrmr
- exp_tag: feature20_mrmr

## Notes

RTX4090 20轮 smoke 控制组; feature_order=mrmr; 作为 corr_greedy / semantic_group 的同轮次对照基线

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_feature20_mrmr_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_feature20_mrmr_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_feature20_mrmr_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_feature20_mrmr_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_feature20_mrmr_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_feature20_mrmr_model.pt
