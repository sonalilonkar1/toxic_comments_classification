"""Monitor config_030 completion, stop comprehensive tuning, and run config_011 with 20 epochs."""

import time
import subprocess
import sys
from pathlib import Path

def check_config_completed(config_num):
    """Check if a config has completed by looking for summary_metrics.json"""
    config_dir = Path(f"experiments/lstm_tuning_comprehensive/config_{config_num:03d}")
    summary_file = config_dir / "summary_metrics.json"
    return summary_file.exists()

def find_tuning_process():
    """Find the PID of the running tune_lstm_comprehensive process"""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if 'tune_lstm_comprehensive' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) > 1:
                    return int(parts[1])  # PID is second column
    except Exception as e:
        print(f"Error finding process: {e}")
    return None

def stop_tuning_process(pid):
    """Stop the tuning process gracefully"""
    try:
        print(f"Stopping tuning process (PID {pid})...")
        subprocess.run(["kill", str(pid)])
        time.sleep(2)  # Wait a moment for process to stop
        # Check if still running, force kill if needed
        result = subprocess.run(["ps", "-p", str(pid)], capture_output=True)
        if result.returncode == 0:
            print(f"Process still running, force killing...")
            subprocess.run(["kill", "-9", str(pid)])
        print("✅ Tuning process stopped")
        return True
    except Exception as e:
        print(f"Error stopping process: {e}")
        return False

def run_config011_extended():
    """Run config_011 with 20 epochs"""
    print("\n" + "=" * 70)
    print("Starting config_011 extended training (20 epochs)")
    print("=" * 70)
    
    script_path = Path("scripts/train_config011_extended.py")
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--epochs", "20", "--fold", "fold1"],
            cwd=Path.cwd()
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return False

def main():
    print("=" * 70)
    print("Monitoring config_030 completion")
    print("=" * 70)
    print("Waiting for config_030 to complete...")
    print("(This will check every 60 seconds)")
    print("=" * 70)
    
    check_interval = 60  # Check every 60 seconds
    max_checks = 1440  # Check for up to 24 hours (1440 minutes)
    
    for check_num in range(max_checks):
        if check_config_completed(30):
            print(f"\n✅ config_030 has completed!")
            
            # Find and stop the tuning process
            pid = find_tuning_process()
            if pid:
                if stop_tuning_process(pid):
                    time.sleep(5)  # Wait a bit more for cleanup
                else:
                    print("⚠️  Could not stop process, but continuing anyway...")
            else:
                print("⚠️  Could not find tuning process (may have already stopped)")
            
            # Run config_011 extended training
            print("\n" + "=" * 70)
            print("Switching to config_011 extended training")
            print("=" * 70)
            success = run_config011_extended()
            
            if success:
                print("\n✅ config_011 extended training started successfully!")
            else:
                print("\n❌ Failed to start config_011 extended training")
            
            return
        
        # Progress update every 10 checks (10 minutes)
        if check_num > 0 and check_num % 10 == 0:
            elapsed_minutes = check_num * check_interval / 60
            print(f"[{elapsed_minutes:.0f} min] Still waiting for config_030...")
        
        time.sleep(check_interval)
    
    print("\n⚠️  Timeout: config_030 did not complete within 24 hours")
    print("You may want to check the process manually")

if __name__ == "__main__":
    main()


