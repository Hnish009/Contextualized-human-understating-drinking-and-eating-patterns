"""
Training script for Model 1: Age and Mood Detection
Trains CNN on FER2013 (emotions) and UTKFace (age) datasets
Saves model, training history, and generates graphs for PPT
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
sys.path.append('.')
from src.model1_age_mood import create_age_mood_model, compile_model
from src.preprocess_data import load_fer2013, load_utk, combine_fer_utk


def plot_training_history(history, save_dir='models/training_history'):
    """
    Plot and save training history graphs for PPT presentation
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model 1: Age & Mood Detection - Training History', fontsize=16, fontweight='bold')
    
    # Age Loss (MAE)
    axes[0, 0].plot(history['age_output_loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history['val_age_output_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Age Prediction Loss (MAE)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mood Accuracy
    axes[0, 1].plot(history['mood_output_accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history['val_mood_output_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Mood Classification Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mood Loss
    axes[1, 0].plot(history['mood_output_loss'], label='Training Loss', linewidth=2)
    axes[1, 0].plot(history['val_mood_output_loss'], label='Validation Loss', linewidth=2)
    axes[1, 0].set_title('Mood Classification Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Crossentropy Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined Loss
    total_train_loss = np.array(history['age_output_loss']) + np.array(history['mood_output_loss'])
    total_val_loss = np.array(history['val_age_output_loss']) + np.array(history['val_mood_output_loss'])
    axes[1, 1].plot(total_train_loss, label='Training Total Loss', linewidth=2)
    axes[1, 1].plot(total_val_loss, label='Validation Total Loss', linewidth=2)
    axes[1, 1].set_title('Combined Loss (Age + Mood)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Total Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'model1_training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Training graphs saved to {save_path / 'model1_training_history.png'}")
    plt.close()


def save_training_info(model, history, epochs, save_dir='models/training_history'):
    """Save model summary, training info, and metrics for PPT"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model summary
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    with open(save_path / 'model1_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_list))
    
    # Save training info
    info = {
        'epochs_trained': epochs,
        'total_params': model.count_params(),
        'final_metrics': {
            'age_mae_train': float(history['age_output_mae'][-1]),
            'age_mae_val': float(history['val_age_output_mae'][-1]),
            'mood_acc_train': float(history['mood_output_accuracy'][-1]),
            'mood_acc_val': float(history['val_mood_output_accuracy'][-1]),
            'mood_loss_train': float(history['mood_output_loss'][-1]),
            'mood_loss_val': float(history['val_mood_output_loss'][-1])
        },
        'architecture': {
            'input_shape': model.input_shape,
            'outputs': ['age_output', 'mood_output']
        }
    }
    
    with open(save_path / 'model1_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✓ Training info saved to {save_path / 'model1_info.json'}")
    print(f"\n📊 Final Metrics:")
    print(f"   Age MAE - Train: {info['final_metrics']['age_mae_train']:.2f}, Val: {info['final_metrics']['age_mae_val']:.2f}")
    print(f"   Mood Accuracy - Train: {info['final_metrics']['mood_acc_train']:.2%}, Val: {info['final_metrics']['mood_acc_val']:.2%}")


def train_model1(epochs=50, batch_size=32, learning_rate=0.001):
    """
    Main training function for Model 1
    """
    print("=" * 60)
    print("Training Model 1: Age & Mood Detection CNN")
    print("=" * 60)
    
    # Check GPU/CPU
    print("\nChecking GPU/CPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
        print("  Training will use GPU (faster: 30-60 min)")
    else:
        print("✗ No GPU detected - using CPU")
        print("  Training will use CPU (slower: 2-4 hours)")
    print()
    
    # Load datasets
    fer_images, fer_emotions = load_fer2013(max_samples=20000)  # Limit for faster training
    utk_images, utk_ages = load_utk(max_samples=10000)  # Limit for faster training
    
    if fer_images is None or utk_images is None:
        print("❌ Failed to load datasets. Please check data paths.")
        return
    
    # Prepare data
    # Strategy: Use FER for mood, UTK for age
    # We'll need to combine them or train separately
    # For simplicity, let's use FER images with synthetic ages (or use UTK with synthetic emotions)
    # Better approach: Use both datasets, train on FER for mood, UTK for age
    
    # Option 1: Train on FER (mood known, age unknown) - use average age
    # Option 2: Train on UTK (age known, mood unknown) - use neutral mood
    # Option 3: Use transfer learning or data augmentation
    
    # For now, let's combine: use FER images with estimated ages (random 10-70 range)
    # and UTK images with estimated emotions (neutral or predicted)
    
    print("\nPreparing combined dataset...")
    
    # Use FER for mood training (assign random ages for now, or use average)
    fer_with_ages = np.full(len(fer_images), 35.0)  # Average age placeholder
    
    # Use UTK for age training (assign neutral mood)
    neutral_mood = np.zeros((len(utk_images), 7))
    neutral_mood[:, 6] = 1.0  # All neutral
    
    # Combine datasets
    all_images = np.concatenate([fer_images, utk_images])
    all_ages = np.concatenate([fer_with_ages, utk_ages])
    all_emotions = np.concatenate([fer_emotions, neutral_mood])
    
    # Reshape images for model input
    all_images = all_images.reshape(-1, 64, 64, 1)
    
    # Shuffle
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_ages = all_ages[indices]
    all_emotions = all_emotions[indices]
    
    # Split train/val
    split_idx = int(0.8 * len(all_images))
    X_train = all_images[:split_idx]
    X_val = all_images[split_idx:]
    y_age_train = all_ages[:split_idx]
    y_age_val = all_ages[split_idx:]
    y_mood_train = all_emotions[:split_idx]
    y_mood_val = all_emotions[split_idx:]
    
    print(f"✓ Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create and compile model
    print("\nCreating model...")
    model = create_age_mood_model()
    model = compile_model(model, learning_rate=learning_rate)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('models/age_mood_model.h5', save_best_only=True, monitor='val_loss'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        # Save progress after each epoch
        keras.callbacks.CSVLogger('models/training_history/model1_training_progress.csv', append=False)
    ]
    
    # Train
    print(f"\n🚀 Starting training for {epochs} epochs...")
    history = model.fit(
        X_train,
        {'age_output': y_age_train, 'mood_output': y_mood_train},
        validation_data=(X_val, {'age_output': y_age_val, 'mood_output': y_mood_val}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model.save('models/age_mood_model.h5')
    print("✓ Model saved to models/age_mood_model.h5")
    
    # Generate graphs and save info
    plot_training_history(history.history)
    save_training_info(model, history.history, epochs)
    
    print("\n✅ Training completed!")
    return model, history


if __name__ == '__main__':
    # Train with default parameters (adjust as needed)
    train_model1(epochs=50, batch_size=32, learning_rate=0.001)

