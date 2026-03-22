# Experiment Summary

- Archived at: 2026-03-22 19:31:03
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_feature20_semantic_group.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-22 19:30:03

## Metrics

- Accuracy: 78.8978
- Precision: 86.1937
- Recall: 78.8978
- F1-Score: 81.1259
- Macro-Precision: 57.3291
- Macro-Recall: 73.2198
- Macro-F1: 59.6587
- FAR: 7.4355

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
- feature_order: semantic_group
- exp_tag: feature20_semantic_group

## Notes

RTX4090 20轮 smoke; feature_order=semantic_group; 与 mrmr / corr_greedy 做同轮次对照

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_feature20_semantic_group_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_feature20_semantic_group_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_feature20_semantic_group_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_feature20_semantic_group_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_feature20_semantic_group_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_feature20_semantic_group_model.pt
