# Experiment Summary

- Archived at: 2026-03-22 19:02:36
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_feature20_corr_greedy.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-22 18:30:54

## Metrics

- Accuracy: 79.0240
- Precision: 86.1141
- Recall: 79.0240
- F1-Score: 81.1545
- Macro-Precision: 57.4221
- Macro-Recall: 73.5335
- Macro-F1: 59.9468
- FAR: 7.3667

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
- feature_order: corr_greedy
- exp_tag: feature20_corr_greedy

## Notes

RTX4090 20轮 smoke; feature_order=corr_greedy; 与 feature20_mrmr 做同轮次对照

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_feature20_corr_greedy_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_feature20_corr_greedy_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_feature20_corr_greedy_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_feature20_corr_greedy_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_feature20_corr_greedy_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_feature20_corr_greedy_model.pt
