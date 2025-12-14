"""Monitor LSTM training progress and notify when complete."""

import json
import sys
import time
from pathlib import Path
from subprocess import check_output

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def check_training_process():
    """Check if training process is still running."""
    try:
        result = check_output(
            ["ps", "aux"],
            text=True
        )
        return "tune_lstm_single" in result or "train_deep" in result
    except:
        return False


def check_results_ready(results_dir: Path):
    """Check if training results are ready."""
    if not results_dir.exists():
        return False, None
    
    # Look for overall_metrics.json in any subdirectory
    for subdir in results_dir.rglob("overall_metrics.json"):
        try:
            with open(subdir, "r") as f:
                metrics = json.load(f)
                return True, metrics
        except:
            continue
    
    # Also check for result.json in the root
    result_file = results_dir / "result.json"
    if result_file.exists():
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
                if "metrics" in data:
                    return True, data["metrics"]
        except:
            pass
    
    return False, None


def check_checkpoints(results_dir: Path):
    """Check for checkpoint files to estimate progress."""
    if not results_dir.exists():
        return 0, 0
    
    checkpoint_dirs = list(results_dir.rglob("checkpoints"))
    if not checkpoint_dirs:
        return 0, 0
    
    checkpoint_dir = checkpoint_dirs[0]
    if not checkpoint_dir.exists():
        return 0, 0
    
    # Count checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    return len(checkpoint_files), 15  # Assuming 15 epochs max


def notify_complete(message: str):
    """Notify user that training is complete."""
    print("\n" + "=" * 70)
    print("🔔 TRAINING COMPLETE!")
    print("=" * 70)
    print(message)
    print("=" * 70)
    
    # Try to play a system sound (macOS)
    try:
        import os
        os.system('afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || echo "🔔"')
    except:
        pass


def main():
    results_dir = Path("experiments/lstm_tuning_single")
    check_interval = 30  # Check every 30 seconds
    
    print("=" * 70)
    print("🔍 MONITORING LSTM TRAINING")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print(f"Check interval: {check_interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 70)
    
    start_time = time.time()
    last_checkpoint_count = 0
    
    try:
        while True:
            # Check if process is running
            process_running = check_training_process()
            
            # Check for results
            results_ready, metrics = check_results_ready(results_dir)
            
            # Check checkpoints for progress
            checkpoint_count, max_epochs = check_checkpoints(results_dir)
            
            elapsed = int(time.time() - start_time)
            elapsed_str = f"{elapsed // 60}m {elapsed % 60}s"
            
            if results_ready and metrics:
                # Training complete!
                notify_complete(
                    f"✅ Training completed in {elapsed_str}!\n\n"
                    f"📊 Results:\n"
                    f"   PR-AUC (Macro): {metrics.get('pr_auc', metrics.get('macro_pr_auc', 'N/A')):.4f}\n"
                    f"   F1 Macro:       {metrics.get('f1_macro', 'N/A'):.4f}\n"
                    f"   F1 Micro:       {metrics.get('f1_micro', 'N/A'):.4f}\n"
                    f"   ROC-AUC:        {metrics.get('roc_auc', 'N/A'):.4f}\n"
                )
                break
            
            # Show progress
            if process_running:
                if checkpoint_count > last_checkpoint_count:
                    progress = (checkpoint_count / max_epochs * 100) if max_epochs > 0 else 0
                    print(f"\r⏳ Training in progress... [{elapsed_str}] "
                          f"Epochs: {checkpoint_count}/{max_epochs} ({progress:.1f}%)", end="", flush=True)
                    last_checkpoint_count = checkpoint_count
                else:
                    print(f"\r⏳ Training in progress... [{elapsed_str}]", end="", flush=True)
            else:
                # Process not running but no results yet - might have failed or just finished
                if checkpoint_count > 0:
                    print(f"\n⚠️  Process stopped but checking for final results...")
                    time.sleep(5)  # Wait a bit for files to be written
                    results_ready, metrics = check_results_ready(results_dir)
                    if results_ready and metrics:
                        notify_complete(
                            f"✅ Training completed!\n\n"
                            f"📊 Results:\n"
                            f"   PR-AUC (Macro): {metrics.get('pr_auc', metrics.get('macro_pr_auc', 'N/A')):.4f}\n"
                            f"   F1 Macro:       {metrics.get('f1_macro', 'N/A'):.4f}\n"
                            f"   F1 Micro:       {metrics.get('f1_micro', 'N/A'):.4f}\n"
                        )
                        break
                    else:
                        print(f"❌ Training may have failed. Check logs.")
                        break
                else:
                    print(f"\n❌ Training process not found and no checkpoints. May have failed.")
                    break
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
        if results_ready and metrics:
            print(f"\n📊 Latest results found:")
            print(f"   PR-AUC: {metrics.get('pr_auc', metrics.get('macro_pr_auc', 'N/A'))}")


if __name__ == "__main__":
    main()




