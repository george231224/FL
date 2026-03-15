import json
import os
from datetime import datetime
import pandas as pd

class ResultLogger:
    def __init__(self, results_dir='./results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_result(self, dataset, model_name, partition, alpha, metrics, params,
                    classification='multi'):
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': dataset,
            'model': model_name,
            'partition': partition,
            'alpha': alpha if partition == 'non-iid' else None,
            'classification': classification,
            'metrics': metrics,
            'params': params
        }

        filename = f"{dataset}_{model_name}_{partition}_{classification}_alpha{alpha}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"\n结果已保存: {filepath}")
        self._append_to_summary(result)

    def _append_to_summary(self, result):
        summary_file = os.path.join(self.results_dir, 'summary.csv')

        flat_result = {
            'timestamp': result['timestamp'],
            'dataset': result['dataset'],
            'model': result['model'],
            'partition': result['partition'],
            'alpha': result['alpha'],
            'classification': result.get('classification', 'multi'),
            **result['metrics'],
            **{f'param_{k}': v for k, v in result['params'].items()}
        }

        df = pd.DataFrame([flat_result])

        if os.path.exists(summary_file):
            df.to_csv(summary_file, mode='a', header=False, index=False)
        else:
            df.to_csv(summary_file, index=False)
