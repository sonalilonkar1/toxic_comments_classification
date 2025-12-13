"""Test a single promising LSTM configuration to improve PR-AUC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.train_deep import DeepTrainConfig, run_deep_training_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Test a single promising LSTM configuration"
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="fold1",
        help="Fold to use for testing (default: fold1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/lstm_tuning_single"),
        help="Output directory for experiment",
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
    args = parser.parse_args()

    # Promising configuration based on:
    # 1. config2_large was better than config1_small
    # 2. Larger hidden_dim (256) often helps with complex patterns
    # 3. Lower dropout (0.2) allows more learning
    # 4. More epochs (15) for better convergence
    # 5. Slightly higher learning rate (0.0015) for faster learning
    config = {
        "name": "optimized_large",
        "description": "Optimized: hidden_dim=256, max_length=200, num_layers=2, lr=0.0015, dropout=0.2, epochs=15",
        "params": {
            "hidden_dim": 256,      # Increased from 128 for more capacity
            "max_length": 200,       # Keep from config2_large
            "num_layers": 2,        # Keep from config2_large
            "dropout": 0.2,         # Lower from 0.3 to allow more learning
            "embedding_dim": 128,
            "bidirectional": True,
            "vocab_size": 10000,
            "batch_size": 32,
            "epochs": args.epochs,
            "learning_rate": 0.0015,  # Slightly higher for faster learning
        },
    }

    print("=" * 70)
    print("TESTING OPTIMIZED LSTM CONFIGURATION")
    print("=" * 70)
    print(f"Configuration: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    train_config = DeepTrainConfig(
        model_type="lstm",
        fold=args.fold,
        output_dir=args.output_dir / config["name"],
        data_path=args.data_path,
        splits_dir=args.splits_dir,
        lstm_params=config["params"],
        lstm_config_path=None,  # Prevent YAML override
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
        print(f"   PR-AUC (Macro):     {pr_auc:.4f}")
        print(f"   F1 Macro:           {overall_metrics.get('f1_macro', 0):.4f}")
        print(f"   F1 Micro:           {overall_metrics.get('f1_micro', 0):.4f}")
        print(f"   ROC-AUC:            {overall_metrics.get('roc_auc', 0):.4f}")
        print(f"   Hamming Loss:      {overall_metrics.get('hamming_loss', 0):.4f}")
        
        # Compare with previous best
        print(f"\n📈 Comparison with previous LSTM (avg PR-AUC ~0.512):")
        improvement = pr_auc - 0.512
        if improvement > 0:
            print(f"   ✅ Improvement: +{improvement:.4f} ({improvement/0.512*100:.1f}%)")
        else:
            print(f"   ⚠️  Change: {improvement:.4f} ({improvement/0.512*100:.1f}%)")
        
        # Save results
        summary_path = args.output_dir / "result.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary_data = {
            "configuration": config,
            "fold": args.fold,
            "metrics": overall_metrics,
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {summary_path}")
        
        if pr_auc > 0.55:  # If we get above 0.55, that's a good improvement
            print(f"\n💡 This configuration shows promise! Consider updating configs/lstm.yaml")
            print(f"   with these parameters if you want to train across all folds.")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()




