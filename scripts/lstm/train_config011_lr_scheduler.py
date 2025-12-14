"""Train config_011 (best LSTM config) with learning rate scheduling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.train_deep import DeepTrainConfig, run_deep_training_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train best LSTM config (config_011) with learning rate scheduling"
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
        default=15,
        help="Maximum number of training epochs (default: 15, early stopping may stop earlier)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine", "plateau"],
        default="cosine",
        help="Learning rate scheduler type: 'cosine' or 'plateau' (default: cosine)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/lstm_config011_lr_scheduler"),
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
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("data/embeddings/glove.6B.100d.txt"),
        help="Path to pre-trained embeddings (default: GloVe 100d)",
    )
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Freeze embedding layer during training (default: False, embeddings will be fine-tuned)",
    )
    args = parser.parse_args()

    # Best configuration from tuning (config_011)
    config = {
        "name": "config011_lr_scheduler",
        "description": f"Best config (011) with LR scheduling ({args.scheduler}) + GloVe: hidden_dim=256, max_length=200, num_layers=2, lr=0.0005, dropout=0.4",
        "params": {
            "hidden_dim": 256,
            "max_length": 200,
            "num_layers": 2,
            "dropout": 0.4,
            "embedding_dim": 100,  # Updated to match GloVe 100d
            "bidirectional": True,
            "vocab_size": 10000,
            "batch_size": 32,
            "epochs": args.epochs,
            "learning_rate": 0.0005,
            "embedding_path": str(args.embeddings_path),  # Add pre-trained embeddings
            "freeze_embeddings": args.freeze_embeddings,  # Allow fine-tuning by default
            "use_class_weights": True,  # Keep class weighting from improved version
            "early_stopping_patience": args.patience,
            "early_stopping_metric": "pr_auc",
            "lr_scheduler_type": args.scheduler,
        },
    }

    # Set scheduler-specific parameters
    if args.scheduler == "cosine":
        config["params"]["lr_scheduler_params"] = {
            "T_max": args.epochs,  # Total epochs for cosine annealing
            "eta_min": 1e-6,  # Minimum learning rate
        }
        # Note: Cosine annealing will decay LR smoothly over T_max epochs
    elif args.scheduler == "plateau":
        config["params"]["lr_scheduler_params"] = {
            "factor": 0.5,  # Reduce LR by this factor
            "patience": 3,  # Wait this many epochs before reducing
            "min_lr": 1e-6,  # Minimum learning rate
        }

    print("=" * 70)
    print("TRAINING BEST LSTM CONFIG (config_011) WITH LR SCHEDULING")
    print("=" * 70)
    print(f"Configuration: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Fold: {args.fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Scheduler params: {config['params']['lr_scheduler_params']}")
    print(f"Pre-trained embeddings: {args.embeddings_path}")
    print(f"Freeze embeddings: {args.freeze_embeddings}")
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
        
        # Compare with previous results
        baseline_pr_auc = 0.570228  # config_011 baseline (15 epochs)
        improved_pr_auc = 0.579760  # config_011 with class weights + early stopping
        
        print(f"\n📈 Comparison:")
        print(f"   Baseline (15 epochs):           {baseline_pr_auc:.6f}")
        print(f"   With class weights (14 epochs): {improved_pr_auc:.6f}")
        print(f"   With LR scheduling:             {pr_auc:.6f}")
        
        improvement_vs_baseline = pr_auc - baseline_pr_auc
        improvement_vs_improved = pr_auc - improved_pr_auc
        
        if improvement_vs_baseline > 0:
            print(f"\n   ✅ vs Baseline: +{improvement_vs_baseline:.6f} ({improvement_vs_baseline/baseline_pr_auc*100:.2f}%)")
        else:
            print(f"\n   ⚠️  vs Baseline: {improvement_vs_baseline:.6f} ({improvement_vs_baseline/baseline_pr_auc*100:.2f}%)")
        
        if improvement_vs_improved > 0:
            print(f"   ✅ vs Improved: +{improvement_vs_improved:.6f} ({improvement_vs_improved/improved_pr_auc*100:.2f}%)")
        else:
            print(f"   ⚠️  vs Improved: {improvement_vs_improved:.6f} ({improvement_vs_improved/improved_pr_auc*100:.2f}%)")
        
        # Save results
        summary_path = args.output_dir / "result.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary_data = {
            "configuration": config,
            "fold": args.fold,
            "metrics": overall_metrics,
            "comparison": {
                "baseline_pr_auc_15_epochs": baseline_pr_auc,
                "improved_pr_auc_14_epochs": improved_pr_auc,
                "lr_scheduler_pr_auc": pr_auc,
                "improvement_vs_baseline": improvement_vs_baseline,
                "improvement_vs_improved": improvement_vs_improved,
            },
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {summary_path}")
        
        if pr_auc > 0.6:
            print(f"\n🎉 SUCCESS! Reached target PR-AUC of 0.6!")
        elif pr_auc > improved_pr_auc:
            print(f"\n💡 LR scheduling improved results!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

