"""
Quick script to check training progress while Model 1 is training
Run this in a separate terminal while training is happening
"""

import csv
import os
from pathlib import Path
from datetime import datetime

def check_progress():
    """Check current training progress"""
    print("=" * 60)
    print("Training Progress Check")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check CSV log file
    csv_path = Path('models/training_history/model1_training_progress.csv')
    
    if csv_path.exists():
        print("\nOK: Training log found!")
        print(f"  Reading: {csv_path}\n")
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            if rows:
                latest = rows[-1]
                print("Latest Training Metrics:")
                print("-" * 60)
                print(f"Epoch: {latest.get('epoch', 'N/A')}")
                print(f"\nAge Metrics:")
                age_loss = latest.get('age_output_loss', 'N/A')
                age_mae = latest.get('age_output_mae', 'N/A')
                age_val_loss = latest.get('val_age_output_loss', 'N/A')
                age_val_mae = latest.get('val_age_output_mae', 'N/A')
                print(f"  Loss: {age_loss}")
                print(f"  MAE: {age_mae}")
                print(f"  Val Loss: {age_val_loss}")
                print(f"  Val MAE: {age_val_mae}")
                print(f"\nMood Metrics:")
                mood_loss = latest.get('mood_output_loss', 'N/A')
                mood_acc = latest.get('mood_output_accuracy', 'N/A')
                mood_val_loss = latest.get('val_mood_output_loss', 'N/A')
                mood_val_acc = latest.get('val_mood_output_accuracy', 'N/A')
                print(f"  Loss: {mood_loss}")
                print(f"  Accuracy: {mood_acc}")
                print(f"  Val Loss: {mood_val_loss}")
                print(f"  Val Accuracy: {mood_val_acc}")
                print(f"\nTotal Epochs Completed: {len(rows)}")
                
                # Show trend (last 3 epochs)
                if len(rows) >= 3:
                    print("\nRecent Trend (last 3 epochs):")
                    for i, row in enumerate(rows[-3:], 1):
                        epoch = row.get('epoch', '?')
                        val_loss = row.get('val_loss', '?')
                        val_acc = row.get('val_mood_output_accuracy', '?')
                        print(f"  Epoch {epoch}: Val Loss={val_loss}, Val Acc={val_acc}")
            else:
                print("  No data yet - training just started")
                
        except Exception as e:
            print(f"  Error reading log: {e}")
    else:
        print("\nTraining log not found yet")
        print("  It will be created after first epoch completes")
    
    # Check model file
    model_path = Path('models/age_mood_model.h5')
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        print(f"\nOK: Model checkpoint exists")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Last updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  (Updated {datetime.now() - mtime} ago)")
    
    print("\n" + "=" * 60)
    print("Tips:")
    print("  - Run this script again to see updated progress")
    print("  - Check Task Manager > GPU for GPU usage")
    print("  - Training terminal shows real-time progress")
    print("=" * 60)

if __name__ == '__main__':
    check_progress()

