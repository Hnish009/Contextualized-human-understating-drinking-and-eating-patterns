"""
Training script for Model 2: Drink Suggestion MLP
Trains on custom drink dataset with feedback learning capability
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
from src.model2_drink import create_drink_model, compile_model
from src.preprocess_data import load_drink_data


def plot_training_history(history, save_dir='models/training_history'):
    """Plot and save training history graphs for PPT presentation"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Model 2: Drink Suggestion - Training History', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Loss (MSE)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'model2_training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Training graphs saved to {save_path / 'model2_training_history.png'}")
    plt.close()


def save_training_info(model, history, epochs, save_dir='models/training_history'):
    """Save model summary, training info, and metrics for PPT"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model summary
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    with open(save_path / 'model2_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_list))
    
    # Save training info
    info = {
        'epochs_trained': epochs,
        'total_params': model.count_params(),
        'final_metrics': {
            'loss_train': float(history['loss'][-1]),
            'loss_val': float(history['val_loss'][-1]),
            'mae_train': float(history['mae'][-1]),
            'mae_val': float(history['val_mae'][-1])
        },
        'architecture': {
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'bottles': ['Sugar', 'Salt', 'Lemon', 'Soda', 'Mint', 'Ginger', 'Masala']
        }
    }
    
    with open(save_path / 'model2_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✓ Training info saved to {save_path / 'model2_info.json'}")
    print(f"\n📊 Final Metrics:")
    print(f"   Loss (MSE) - Train: {info['final_metrics']['loss_train']:.4f}, Val: {info['final_metrics']['loss_val']:.4f}")
    print(f"   MAE - Train: {info['final_metrics']['mae_train']:.4f}, Val: {info['final_metrics']['mae_val']:.4f}")


def train_model2(epochs=100, batch_size=16, learning_rate=0.001):
    """
    Main training function for Model 2
    """
    print("=" * 60)
    print("Training Model 2: Drink Suggestion MLP")
    print("=" * 60)
    
    # Check GPU/CPU
    print("\nChecking GPU/CPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
        print("  Training will use GPU (faster)")
    else:
        print("✗ No GPU detected - using CPU")
        print("  Training will use CPU (slower)")
    print()
    
    # Load custom drink dataset
    data = load_drink_data('data/drinks.csv')
    
    if data is None:
        print("❌ Failed to load drink dataset.")
        print("Please create data/drinks.csv with columns: gender, age, mood, drink")
        return
    
    X, y = data
    
    # Split train/val
    split_idx = int(0.8 * len(X))
    X_train = X[:split_idx]
    X_val = X[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    print(f"✓ Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create and compile model
    print("\nCreating model...")
    model = create_drink_model()
    model = compile_model(model, learning_rate=learning_rate)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('models/drink_model.h5', save_best_only=True, monitor='val_loss'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
        # Save progress after each epoch
        keras.callbacks.CSVLogger('models/training_history/model2_training_progress.csv', append=False)
    ]
    
    # Train
    print(f"\n🚀 Starting training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model.save('models/drink_model.h5')
    print("✓ Model saved to models/drink_model.h5")
    
    # Generate graphs and save info
    plot_training_history(history.history)
    save_training_info(model, history.history, epochs)
    
    print("\n✅ Training completed!")
    return model, history


if __name__ == '__main__':
    train_model2(epochs=100, batch_size=16, learning_rate=0.001)

