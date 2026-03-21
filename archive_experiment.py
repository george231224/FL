import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def build_result_filename(dataset, model, partition, classification, alpha, exp_tag=None):
    alpha_str = str(alpha)
    base = f"{dataset}_{model}_{partition}_{classification}_alpha{alpha_str}"
    if exp_tag:
        base = f"{base}_{exp_tag}"
    return f"{base}.json"


def build_prefixes(dataset, model, partition, classification, exp_tag=None, classifier_name=None):
    suffix = f"_{exp_tag}" if exp_tag else ""
    prefixes = []
    if model == "fedpcnn":
        prefixes.append(f"FedPCNN_{dataset}_{partition}_{classification}{suffix}")
        if classifier_name and "two-stage" in str(classifier_name).lower():
            prefixes.append(f"FedPCNN_{dataset}_{partition}_{classification}_two_stage{suffix}")
    elif model == "segmented":
        prefixes.append(f"SegmentedFL_{dataset}_{partition}_{classification}{suffix}")
    else:
        prefixes.append(f"{model}_{dataset}_{partition}_{classification}{suffix}")
    return prefixes


def collect_artifacts(results_dir, prefixes):
    plot_dir = results_dir / "plots"
    model_dir = results_dir / "models"

    plots = []
    models = []

    for prefix in prefixes:
        is_two_stage_prefix = "_two_stage" in prefix
        if plot_dir.exists():
            for path in sorted(plot_dir.glob(f"{prefix}_*")):
                if not is_two_stage_prefix and "_two_stage_" in path.name:
                    continue
                plots.append(path)
        if model_dir.exists():
            for path in sorted(model_dir.glob(f"{prefix}*")):
                if not is_two_stage_prefix and "_two_stage_" in path.name:
                    continue
                models.append(path)

    return sorted(set(plots)), sorted(set(models))


def copy_files(files, destination):
    copied = []
    for src in files:
        dst = destination / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def write_summary(summary_path, result, copied_json, copied_plots, copied_models, notes=None):
    metrics = result.get("metrics", {})
    params = result.get("params", {})

    lines = [
        "# Experiment Summary",
        "",
        f"- Archived at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Result JSON: {copied_json.name}",
        f"- Dataset: {result.get('dataset')}",
        f"- Model: {result.get('model')}",
        f"- Partition: {result.get('partition')}",
        f"- Classification: {result.get('classification')}",
        f"- Timestamp: {result.get('timestamp')}",
        "",
        "## Metrics",
        "",
    ]

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")

    lines.extend(["", "## Params", ""])
    for key, value in params.items():
        lines.append(f"- {key}: {value}")

    if notes:
        lines.extend(["", "## Notes", "", notes])

    lines.extend(["", "## Charts", ""])
    if copied_plots:
        for path in copied_plots:
            lines.append(f"- {path.name}")
    else:
        lines.append("- None found")

    lines.extend(["", "## Model Files", ""])
    if copied_models:
        for path in copied_models:
            lines.append(f"- {path.name}")
    else:
        lines.append("- None found")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Archive one experiment's result json, charts, and model files.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, choices=["fedpcnn", "segmented"])
    parser.add_argument("--partition", required=True, choices=["iid", "non-iid"])
    parser.add_argument("--classification", required=True, choices=["binary", "multi"])
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--exp-tag", default="")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    exp_tag = args.exp_tag.strip().replace(" ", "_") if args.exp_tag else ""
    results_dir = Path(args.results_dir)
    result_json = results_dir / build_result_filename(
        args.dataset,
        args.model,
        args.partition,
        args.classification,
        args.alpha,
        exp_tag or None,
    )

    if not result_json.exists():
        raise FileNotFoundError(f"Result json not found: {result_json}")

    result = json.loads(result_json.read_text(encoding="utf-8"))
    prefixes = build_prefixes(
        args.dataset,
        args.model,
        args.partition,
        args.classification,
        exp_tag or None,
        result.get("params", {}).get("classifier"),
    )
    plots, models = collect_artifacts(results_dir, prefixes)

    archive_root = results_dir / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    experiment_ts = result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")).replace(":", "").replace(" ", "_")
    archived_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder_bits = [experiment_ts]
    if exp_tag:
        folder_bits.append(exp_tag)
    folder_bits.append(archived_ts)
    archive_dir = archive_root / "_".join(folder_bits)
    archive_dir.mkdir(parents=True, exist_ok=True)

    copied_json = archive_dir / result_json.name
    shutil.copy2(result_json, copied_json)
    copied_plots = copy_files(plots, archive_dir)
    copied_models = copy_files(models, archive_dir)

    manifest = {
        "source_result": str(result_json),
        "plots": [str(p) for p in plots],
        "models": [str(m) for m in models],
        "archived_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(archive_dir / "SUMMARY.md", result, copied_json, copied_plots, copied_models, notes=args.notes or None)

    print(f"Archived experiment to: {archive_dir}")
    print(f"Result json: {copied_json.name}")
    print(f"Charts: {len(copied_plots)}")
    print(f"Model files: {len(copied_models)}")


if __name__ == "__main__":
    main()
