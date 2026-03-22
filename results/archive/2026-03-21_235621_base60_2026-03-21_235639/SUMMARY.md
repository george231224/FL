# Experiment Summary

- Archived at: 2026-03-21 23:56:39
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_base60.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-21 23:56:21

## Metrics

- Accuracy: 79.0084
- Precision: 86.2281
- Recall: 79.0084
- F1-Score: 81.1155
- Macro-Precision: 58.1400
- Macro-Recall: 74.0285
- Macro-F1: 60.6420
- FAR: 7.4127

## Params

- num_devices: 10
- global_rounds: 60
- local_epochs: 5
- batch_size: 256
- lr: 0.005
- classifier: CNN+XGBoost(门限=0.30)
- exp_tag: base60

## Notes

RTX4090 remote base60, 60 rounds

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_base60_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_base60_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_base60_model.pt
