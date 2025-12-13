"""Script to tune LSTM hyperparameters by testing different configurations on a single fold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.train_deep import DeepTrainConfig, run_deep_training_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Tune LSTM hyperparameters by testing configurations on a single fold"
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="fold1",
        help="Fold to use for tuning (default: fold1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/lstm_tuning"),
        help="Base output directory for tuning experiments",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/train.csv"),
        help="Path to training data",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing fold splits",
    )
    args = parser.parse_args()

    # Define configurations to test
    configs = [
        {
            "name": "config1_small",
            "description": "Smaller model: hidden_dim=64, max_length=128, num_layers=1",
            "params": {
                "hidden_dim": 64,
                "max_length": 128,
                "num_layers": 1,
                "dropout": 0.3,
                "embedding_dim": 128,
                "bidirectional": True,
                "vocab_size": 10000,
                "batch_size": 32,
                "epochs": args.epochs,
                "learning_rate": 0.001,
            },
        },
        {
            "name": "config2_large",
            "description": "Larger model: hidden_dim=128, max_length=200, num_layers=2",
            "params": {
                "hidden_dim": 128,
                "max_length": 200,
                "num_layers": 2,
                "dropout": 0.3,
                "embedding_dim": 128,
                "bidirectional": True,
                "vocab_size": 10000,
                "batch_size": 32,
                "epochs": args.epochs,
                "learning_rate": 0.001,
            },
        },
    ]

    print("=" * 70)
    print("LSTM Hyperparameter Tuning")
    print("=" * 70)
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of configurations: {len(configs)}")
    print("=" * 70)

    results_summary = {}

    for i, config_info in enumerate(configs, 1):
        print(f"\n{'=' * 70}")
        print(f"Configuration {i}/{len(configs)}: {config_info['name']}")
        print(f"Description: {config_info['description']}")
        print(f"{'=' * 70}")

        config = DeepTrainConfig(
            model_type="lstm",
            fold=args.fold,
            output_dir=args.output_dir / config_info["name"],
            data_path=args.data_path,
            splits_dir=args.splits_dir,
            lstm_params=config_info["params"],
            lstm_config_path=None,  # Prevent YAML override when using custom params
        )

        try:
            results = run_deep_training_pipeline(config)
            fold_results = results.get(args.fold, {})
            overall_metrics = fold_results.get("overall_metrics", {})
            results_summary[config_info["name"]] = {
                "description": config_info["description"],
                "params": config_info["params"],
                "metrics": overall_metrics,
            }

            print(f"\n✅ Configuration {config_info['name']} completed")
            print(f"   F1 Macro: {overall_metrics.get('f1_macro', 0):.4f}")
            print(f"   F1 Micro: {overall_metrics.get('f1_micro', 0):.4f}")
            print(f"   ROC-AUC: {overall_metrics.get('roc_auc', 0):.4f}")

        except Exception as e:
            print(f"\n❌ Configuration {config_info['name']} failed: {e}")
            results_summary[config_info["name"]] = {
                "description": config_info["description"],
                "params": config_info["params"],
                "error": str(e),
            }

    # Print comparison summary
    print(f"\n{'=' * 70}")
    print("TUNING RESULTS SUMMARY")
    print(f"{'=' * 70}")

    for name, result in results_summary.items():
        print(f"\n{name}:")
        print(f"  Description: {result['description']}")
        if "metrics" in result:
            metrics = result["metrics"]
            print(f"  F1 Macro:    {metrics.get('f1_macro', 0):.4f}")
            print(f"  F1 Micro:    {metrics.get('f1_micro', 0):.4f}")
            print(f"  ROC-AUC:     {metrics.get('roc_auc', 0):.4f}")
            print(f"  PR-AUC:      {metrics.get('pr_auc', 0):.4f}")
            print(f"  Hamming Loss: {metrics.get('hamming_loss', 0):.4f}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")

    # Save summary to JSON
    summary_path = args.output_dir / "tuning_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "fold": args.fold,
        "epochs": args.epochs,
        "configurations": results_summary,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'=' * 70}")

    # Determine best configuration
    valid_results = {
        k: v for k, v in results_summary.items() if "metrics" in v
    }
    if valid_results:
        best_config = max(
            valid_results.items(),
            key=lambda x: x[1]["metrics"].get("f1_macro", 0),
        )
        print(f"\n🏆 Best Configuration: {best_config[0]}")
        print(f"   F1 Macro: {best_config[1]['metrics'].get('f1_macro', 0):.4f}")
        print(f"   Description: {best_config[1]['description']}")


if __name__ == "__main__":
    main()

