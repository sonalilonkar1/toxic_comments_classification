"""Train config_011 (best LSTM config) with extended epochs (25-30)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.train_deep import DeepTrainConfig, run_deep_training_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train best LSTM config (config_011) with extended epochs"
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
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/lstm_config011_extended"),
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

    # Check for existing checkpoints to resume from
    output_base = args.output_dir / "config011_extended"
    resume_from = None
    
    # Find the latest checkpoint from ANY training directory
    if output_base.exists():
        checkpoint_dirs = list(output_base.glob("fold1-*"))
        if checkpoint_dirs:
            # Find the latest checkpoint across all directories
            all_checkpoints = []
            for dir_path in checkpoint_dirs:
                checkpoints_dir = dir_path / "checkpoints"
                if checkpoints_dir.exists():
                    checkpoint_files = list(checkpoints_dir.glob("checkpoint_epoch_*.pt"))
                    all_checkpoints.extend(checkpoint_files)
            
            if all_checkpoints:
                # Sort by epoch number (extract from filename)
                all_checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
                resume_from = str(all_checkpoints[-1])
                epoch_num = all_checkpoints[-1].stem.split("_")[-1]
                print(f"📂 Found existing checkpoint: {resume_from}")
                print(f"   Will resume from epoch {epoch_num}")
                print(f"   Training will continue from epoch {int(epoch_num) + 1} to {args.epochs}")

    # Best configuration from tuning (config_011)
    config = {
        "name": "config011_extended",
        "description": "Best config (011) with extended epochs: hidden_dim=256, max_length=200, num_layers=2, lr=0.0005, dropout=0.4",
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
            "resume_from": resume_from,  # Add resume_from parameter
        },
    }

    print("=" * 70)
    print("TRAINING BEST LSTM CONFIG (config_011) WITH EXTENDED EPOCHS")
    print("=" * 70)
    print(f"Configuration: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs} (increased from 15)")
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
        previous_pr_auc = 0.570228  # config_011 with 15 epochs
        improvement = pr_auc - previous_pr_auc
        print(f"\n📈 Comparison with config_011 (15 epochs, PR-AUC={previous_pr_auc:.6f}):")
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
            "previous_pr_auc_15_epochs": previous_pr_auc,
            "improvement": improvement,
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {summary_path}")
        
        if pr_auc > 0.6:
            print(f"\n🎉 SUCCESS! Reached target PR-AUC of 0.6!")
        elif pr_auc > previous_pr_auc:
            print(f"\n💡 Extended training improved results! Consider training on all folds.")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


