

import os
import time
from pathlib import Path
import json

def check_training_progress():
    """Check if training is running and show progress"""
    print("=" * 60)
    print("Training Progress Monitor")
    print("=" * 60)
    
    # Check if model file exists (indicates training has started)
    model_path = Path('models/age_mood_model.h5')
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Model file found: {size_mb:.2f} MB")
        print("  Training has started and checkpoint saved!")
    else:
        print("\n✗ Model file not found yet")
        print("  Training may not have started or no checkpoint saved yet")
    
    # Check training history
    history_path = Path('models/training_history/model1_info.json')
    if history_path.exists():
        print("\n✓ Training history found!")
        try:
            with open(history_path, 'r') as f:
                info = json.load(f)
                print(f"\n  Epochs completed: {info.get('epochs_trained', 'N/A')}")
                metrics = info.get('final_metrics', {})
                if metrics:
                    print(f"\n  Latest Metrics:")
                    print(f"    Age MAE - Train: {metrics.get('age_mae_train', 'N/A'):.3f}")
                    print(f"    Age MAE - Val: {metrics.get('age_mae_val', 'N/A'):.3f}")
                    print(f"    Mood Accuracy - Train: {metrics.get('mood_acc_train', 'N/A'):.2%}")
                    print(f"    Mood Accuracy - Val: {metrics.get('mood_acc_val', 'N/A'):.2%}")
        except Exception as e:
            print(f"  Could not read history: {e}")
    else:
        print("\n✗ Training history not found yet")
        print("  History will be saved after training completes")
    
    # Check if graphs are being generated
    graph_path = Path('models/training_history/model1_training_history.png')
    if graph_path.exists():
        print(f"\n✓ Training graphs available!")
        print(f"  Location: {graph_path}")
    else:
        print("\n✗ Training graphs not generated yet")
        print("  Graphs will be created after training completes")
    
    print("\n" + "=" * 60)
    print("How to Monitor Training:")
    print("=" * 60)
    print("1. Watch the terminal where training is running")
    print("   - You'll see epoch progress with loss/accuracy")
    print("   - Example: 'Epoch 10/50 - loss: 0.5 - val_loss: 0.6'")
    print("\n2. Check Task Manager > Performance > GPU")
    print("   - GPU usage should be high (80-100%)")
    print("   - Memory usage should be active")
    print("\n3. Run this script again:")
    print("   python monitor_training.py")
    print("\n4. Check training logs (if using TensorBoard)")
    print("   tensorboard --logdir=models/training_history")
    print("=" * 60)

if __name__ == '__main__':
    check_training_progress()

