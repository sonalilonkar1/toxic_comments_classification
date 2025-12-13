"""Train config_011 (best LSTM config) with early stopping and class weighting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.train_deep import DeepTrainConfig, run_deep_training_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train best LSTM config (config_011) with early stopping and class weighting"
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="fold1",
        help="Fold to use (default: fold1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs (default: 20, early stopping may stop earlier)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5 epochs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/lstm_config011_improved"),
        help="Output directory for experiment",
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

    # Best configuration from tuning (config_011) with improvements
    config = {
        "name": "config011_improved",
        "description": "Best config (011) with early stopping and class weighting: hidden_dim=256, max_length=200, num_layers=2, lr=0.0005, dropout=0.4",
        "params": {
            "hidden_dim": 256,
            "max_length": 200,
            "num_layers": 2,
            "dropout": 0.4,
            "embedding_dim": 128,
            "bidirectional": True,
            "vocab_size": 10000,
            "batch_size": 32,
            "epochs": args.epochs,
            "learning_rate": 0.0005,
            "use_class_weights": True,  # NEW: Class weighting for imbalanced labels
            "early_stopping_patience": args.patience,  # NEW: Early stopping
            "early_stopping_metric": "pr_auc",  # NEW: Monitor PR-AUC for early stopping
        },
    }

    print("=" * 70)
    print("TRAINING BEST LSTM CONFIG (config_011) WITH IMPROVEMENTS")
    print("=" * 70)
    print(f"Configuration: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Fold: {args.fold}")
    print(f"Max Epochs: {args.epochs} (early stopping with patience={args.patience})")
    print(f"Improvements:")
    print(f"  ✅ Class weighting: ENABLED (handles imbalanced labels)")
    print(f"  ✅ Early stopping: ENABLED (monitors PR-AUC, patience={args.patience})")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    train_config = DeepTrainConfig(
        model_type="lstm",
        fold=args.fold,
        output_dir=args.output_dir / config["name"],
        data_path=args.data_path,
        splits_dir=args.splits_dir,
        lstm_params=config["params"],
        lstm_config_path=None,
    )

    try:
        print("\n🚀 Starting training...")
        results = run_deep_training_pipeline(train_config)
        fold_results = results.get(args.fold, {})
        overall_metrics = fold_results.get("overall_metrics", {})
        
        pr_auc = overall_metrics.get("pr_auc", overall_metrics.get("macro_pr_auc", 0))
        
        print(f"\n{'=' * 70}")
        print("✅ TRAINING COMPLETED")
        print(f"{'=' * 70}")
        print(f"\n📊 Results:")
        print(f"   PR-AUC (Macro):     {pr_auc:.6f}")
        print(f"   F1 Macro:           {overall_metrics.get('f1_macro', 0):.4f}")
        print(f"   F1 Micro:           {overall_metrics.get('f1_micro', 0):.4f}")
        print(f"   ROC-AUC:            {overall_metrics.get('roc_auc', 0):.4f}")
        print(f"   Hamming Loss:      {overall_metrics.get('hamming_loss', 0):.4f}")
        
        # Compare with previous result
        previous_pr_auc = 0.570228  # config_011 with 15 epochs, no improvements
        improvement = pr_auc - previous_pr_auc
        print(f"\n📈 Comparison with config_011 baseline (PR-AUC={previous_pr_auc:.6f}):")
        if improvement > 0:
            print(f"   ✅ Improvement: +{improvement:.6f} ({improvement/previous_pr_auc*100:.2f}%)")
        else:
            print(f"   ⚠️  Change: {improvement:.6f} ({improvement/previous_pr_auc*100:.2f}%)")
        
        # Save results
        summary_path = args.output_dir / "result.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary_data = {
            "configuration": config,
            "fold": args.fold,
            "metrics": overall_metrics,
            "previous_pr_auc_baseline": previous_pr_auc,
            "improvement": improvement,
            "improvements_used": {
                "class_weights": True,
                "early_stopping": True,
                "early_stopping_patience": args.patience,
            },
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {summary_path}")
        
        if pr_auc > 0.6:
            print(f"\n🎉 SUCCESS! Reached target PR-AUC of 0.6!")
        elif pr_auc > previous_pr_auc:
            print(f"\n💡 Improvements helped! Consider training on all folds.")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

