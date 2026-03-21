# Experiment Summary

- Archived at: 2026-03-21 23:02:34
- Result JSON: UNSW-NB15_fedpcnn_non-iid_multi_alpha0.5_smoke.json
- Dataset: UNSW-NB15
- Model: fedpcnn
- Partition: non-iid
- Classification: multi
- Timestamp: 2026-03-21 23:02:01

## Metrics

- Accuracy: 78.4787
- Precision: 86.2033
- Recall: 78.4787
- F1-Score: 80.8392
- Macro-Precision: 57.3563
- Macro-Recall: 73.2579
- Macro-F1: 59.6444
- FAR: 7.9670

## Params

- num_devices: 10
- global_rounds: 5
- local_epochs: 5
- batch_size: 256
- lr: 0.005
- classifier: CNN+XGBoost(门限=0.33)
- exp_tag: smoke

## Notes

RTX4090 remote smoke validation, 5 rounds

## Charts

- FedPCNN_UNSW-NB15_non-iid_multi_smoke_cm.png
- FedPCNN_UNSW-NB15_non-iid_multi_smoke_comparison.png
- FedPCNN_UNSW-NB15_non-iid_multi_smoke_loss.png
- FedPCNN_UNSW-NB15_non-iid_multi_smoke_metrics.png
- FedPCNN_UNSW-NB15_non-iid_multi_smoke_per_class.png

## Model Files

- FedPCNN_UNSW-NB15_non-iid_multi_smoke_model.pt
