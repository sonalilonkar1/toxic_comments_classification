"""Comprehensive LSTM hyperparameter tuning focused on improving PR-AUC."""

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
        description="Comprehensive LSTM hyperparameter tuning focused on PR-AUC"
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
        default=Path("experiments/lstm_tuning_comprehensive"),
        help="Base output directory for tuning experiments",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
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
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tuning with fewer configurations",
    )
    args = parser.parse_args()

    # Define comprehensive configurations to test
    # Focus on parameters that most impact PR-AUC
    if args.quick:
        # Quick tuning: test key variations
        configs = [
            {
                "name": "baseline",
                "description": "Baseline: hidden_dim=128, max_length=200, num_layers=2, lr=0.001, dropout=0.3",
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
            {
                "name": "higher_lr",
                "description": "Higher learning rate: lr=0.002",
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
                    "learning_rate": 0.002,
                },
            },
            {
                "name": "lower_dropout",
                "description": "Lower dropout: dropout=0.2",
                "params": {
                    "hidden_dim": 128,
                    "max_length": 200,
                    "num_layers": 2,
                    "dropout": 0.2,
                    "embedding_dim": 128,
                    "bidirectional": True,
                    "vocab_size": 10000,
                    "batch_size": 32,
                    "epochs": args.epochs,
                    "learning_rate": 0.001,
                },
            },
            {
                "name": "larger_hidden",
                "description": "Larger hidden dimension: hidden_dim=256",
                "params": {
                    "hidden_dim": 256,
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
            {
                "name": "longer_sequences",
                "description": "Longer sequences: max_length=300",
                "params": {
                    "hidden_dim": 128,
                    "max_length": 300,
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
    else:
        # Comprehensive tuning: test multiple combinations
        configs = []
        
        # Base configurations with different architectures
        base_configs = [
            {"hidden_dim": 128, "max_length": 200, "num_layers": 2},
            {"hidden_dim": 256, "max_length": 200, "num_layers": 2},
            {"hidden_dim": 128, "max_length": 300, "num_layers": 2},
            {"hidden_dim": 256, "max_length": 300, "num_layers": 3},
        ]
        
        # Learning rates to test
        learning_rates = [0.0005, 0.001, 0.002]
        
        # Dropout rates to test
        dropouts = [0.2, 0.3, 0.4]
        
        # Generate configurations
        config_idx = 0
        for base in base_configs:
            for lr in learning_rates:
                for dropout in dropouts:
                    configs.append({
                        "name": f"config_{config_idx:03d}",
                        "description": f"hidden_dim={base['hidden_dim']}, max_length={base['max_length']}, "
                                     f"num_layers={base['num_layers']}, lr={lr}, dropout={dropout}",
                        "params": {
                            "hidden_dim": base["hidden_dim"],
                            "max_length": base["max_length"],
                            "num_layers": base["num_layers"],
                            "dropout": dropout,
                            "embedding_dim": 128,
                            "bidirectional": True,
                            "vocab_size": 10000,
                            "batch_size": 32,
                            "epochs": args.epochs,
                            "learning_rate": lr,
                        },
                    })
                    config_idx += 1

    print("=" * 70)
    print("COMPREHENSIVE LSTM HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of configurations: {len(configs)}")
    print(f"Optimization metric: PR-AUC (Macro)")
    print("=" * 70)

    results_summary = {}
    best_pr_auc = -1.0
    best_config_name = ""

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

        # Check if this configuration already has results
        config_dir = args.output_dir / config_info["name"]
        summary_file = config_dir / "summary_metrics.json"
        
        if summary_file.exists():
            print(f"⏭️  Configuration {config_info['name']} already completed, loading existing results...")
            try:
                with open(summary_file, "r") as f:
                    existing_results = json.load(f)
                    fold_results = existing_results.get(args.fold, {})
                    if fold_results:
                        pr_auc = fold_results.get("macro_pr_auc", fold_results.get("pr_auc", 0))
                        results_summary[config_info["name"]] = {
                            "description": config_info["description"],
                            "params": config_info["params"],
                            "metrics": fold_results,
                        }
                        print(f"   PR-AUC (Macro): {pr_auc:.4f}")
                        print(f"   F1 Macro:       {fold_results.get('macro_f1', 0):.4f}")
                        print(f"   F1 Micro:       {fold_results.get('micro_f1', 0):.4f}")
                        if pr_auc > best_pr_auc:
                            best_pr_auc = pr_auc
                            best_config_name = config_info["name"]
                        continue
            except Exception as e:
                print(f"⚠️  Could not load existing results, will rerun: {e}")

        try:
            results = run_deep_training_pipeline(config)
            fold_results = results.get(args.fold, {})
            overall_metrics = fold_results.get("overall_metrics", {})
            
            pr_auc = overall_metrics.get("pr_auc", overall_metrics.get("macro_pr_auc", 0))
            
            results_summary[config_info["name"]] = {
                "description": config_info["description"],
                "params": config_info["params"],
                "metrics": overall_metrics,
            }

            print(f"\n✅ Configuration {config_info['name']} completed")
            print(f"   PR-AUC (Macro): {pr_auc:.4f}")
            print(f"   F1 Macro:       {overall_metrics.get('f1_macro', 0):.4f}")
            print(f"   F1 Micro:       {overall_metrics.get('f1_micro', 0):.4f}")
            print(f"   ROC-AUC:        {overall_metrics.get('roc_auc', 0):.4f}")

            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_config_name = config_info["name"]

        except Exception as e:
            print(f"\n❌ Configuration {config_info['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            results_summary[config_info["name"]] = {
                "description": config_info["description"],
                "params": config_info["params"],
                "error": str(e),
            }

    # Print comparison summary
    print(f"\n{'=' * 70}")
    print("TUNING RESULTS SUMMARY (Sorted by PR-AUC)")
    print(f"{'=' * 70}")

    # Sort by PR-AUC
    valid_results = {
        k: v for k, v in results_summary.items() if "metrics" in v
    }
    
    sorted_results = sorted(
        valid_results.items(),
        key=lambda x: x[1]["metrics"].get("pr_auc", x[1]["metrics"].get("macro_pr_auc", 0)),
        reverse=True
    )

    for name, result in sorted_results[:10]:  # Show top 10
        metrics = result["metrics"]
        pr_auc = metrics.get("pr_auc", metrics.get("macro_pr_auc", 0))
        print(f"\n{name}:")
        print(f"  Description: {result['description']}")
        print(f"  PR-AUC:     {pr_auc:.4f}")
        print(f"  F1 Macro:   {metrics.get('f1_macro', 0):.4f}")
        print(f"  F1 Micro:   {metrics.get('f1_micro', 0):.4f}")

    # Save summary to JSON
    summary_path = args.output_dir / "tuning_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "fold": args.fold,
        "epochs": args.epochs,
        "optimization_metric": "PR-AUC (Macro)",
        "configurations": results_summary,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'=' * 70}")

    # Determine best configuration
    if valid_results:
        best_config = sorted_results[0]
        print(f"\n🏆 Best Configuration: {best_config[0]}")
        best_metrics = best_config[1]["metrics"]
        pr_auc = best_metrics.get("pr_auc", best_metrics.get("macro_pr_auc", 0))
        print(f"   PR-AUC:     {pr_auc:.4f}")
        print(f"   F1 Macro:   {best_metrics.get('f1_macro', 0):.4f}")
        print(f"   F1 Micro:   {best_metrics.get('f1_micro', 0):.4f}")
        print(f"   Description: {best_config[1]['description']}")
        print(f"\n💡 To use this configuration, update configs/lstm.yaml with:")
        print(f"   {json.dumps(best_config[1]['params'], indent=2)}")


if __name__ == "__main__":
    main()
