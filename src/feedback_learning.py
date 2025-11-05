"""
Feedback learning module for Model 2
Fine-tunes the drink suggestion model based on user ratings
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from src.model2_drink import create_drink_model, compile_model, prepare_input_features
from src.utils import encode_mood


def load_feedback_data(feedback_file='data/feedback.csv'):
    """Load feedback data from CSV"""
    feedback_path = Path(feedback_file)
    
    if not feedback_path.exists():
        return None
    
    try:
        df = pd.read_csv(feedback_file)
        return df
    except Exception as e:
        print(f"Error loading feedback: {e}")
        return None


def prepare_feedback_training_data(df, learning_rate_weight=0.1):
    """
    Prepare training data from feedback
    Strategy: Adjust target ratios based on ratings
    - High ratings (8-10): Keep ratios as-is (they're good)
    - Medium ratings (5-7): Slight adjustment toward average
    - Low ratings (1-4): More significant adjustment
    
    Args:
        df: DataFrame with feedback data
        learning_rate_weight: How much to adjust ratios (0.0 to 1.0)
    
    Returns:
        X: Input features (gender, age, mood)
        y: Target ratios (adjusted based on ratings)
    """
    X_features = []
    y_ratios = []
    
    # Average drink ratios (balanced baseline)
    avg_ratios = np.array([1/7] * 7)
    
    for idx, row in df.iterrows():
        gender = row['gender']
        age = row['age']
        mood_idx = int(row['mood'])
        rating = row['rating']
        
        # Current predicted ratios
        current_ratios = np.array([
            row['sugar'], row['salt'], row['lemon'], row['soda'],
            row['mint'], row['ginger'], row['masala']
        ])
        
        # Adjust target ratios based on rating
        if rating >= 8:
            # High rating: Keep ratios similar (slight reinforcement)
            target_ratios = current_ratios * (1.0 + learning_rate_weight * 0.1)
        elif rating >= 5:
            # Medium rating: Blend with average (moderate adjustment)
            blend_factor = (rating - 5) / 3.0  # 0.0 to 1.0
            target_ratios = current_ratios * (1.0 - learning_rate_weight * 0.2) + avg_ratios * learning_rate_weight * 0.2
        else:
            # Low rating: Move more toward average (significant adjustment)
            adjustment = learning_rate_weight * (5 - rating) / 4.0  # 0.0 to 0.25
            target_ratios = current_ratios * (1.0 - adjustment) + avg_ratios * adjustment
        
        # Normalize to sum to 1.0
        target_ratios = target_ratios / np.sum(target_ratios)
        
        # Prepare input features
        mood_one_hot = np.zeros(7)
        mood_one_hot[mood_idx] = 1.0
        
        features = prepare_input_features(gender, age, mood_one_hot)
        X_features.append(features[0])  # Remove batch dimension
        y_ratios.append(target_ratios)
    
    return np.array(X_features), np.array(y_ratios)


def fine_tune_model_with_feedback(model, feedback_file='data/feedback.csv', 
                                   epochs=10, batch_size=8, learning_rate=0.0001,
                                   min_samples=10):
    """
    Fine-tune Model 2 with collected feedback
    
    Args:
        model: The drink suggestion model to fine-tune
        feedback_file: Path to feedback CSV
        epochs: Number of fine-tuning epochs
        batch_size: Batch size for fine-tuning
        learning_rate: Learning rate (lower than original training)
        min_samples: Minimum samples needed to fine-tune
    
    Returns:
        True if fine-tuning was successful, False otherwise
    """
    print("\n" + "=" * 60)
    print("Fine-tuning Model 2 with Feedback")
    print("=" * 60)
    
    # Load feedback data
    df = load_feedback_data(feedback_file)
    if df is None:
        print("No feedback data found")
        return False
    
    if len(df) < min_samples:
        print(f"Not enough feedback samples ({len(df)} < {min_samples})")
        print(f"Need at least {min_samples} ratings to fine-tune")
        return False
    
    print(f"Loaded {len(df)} feedback samples")
    print(f"Average rating: {df['rating'].mean():.2f}/10")
    
    # Prepare training data
    print("Preparing training data from feedback...")
    X_train, y_train = prepare_feedback_training_data(df, learning_rate_weight=0.15)
    
    print(f"Training samples: {len(X_train)}")
    
    # Use lower learning rate for fine-tuning
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Fine-tune with feedback
    print(f"\nFine-tuning for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.2
    )
    
    # Save updated model
    model.save('models/drink_model.h5')
    print("\nOK: Model fine-tuned and saved!")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Final MAE: {history.history['mae'][-1]:.4f}")
    
    return True


def update_model_with_feedback(model, feedback_file='data/feedback.csv', 
                               min_samples=10, auto_finetune=True):
    """
    Check feedback and update model if enough samples collected
    
    Args:
        model: The drink suggestion model
        feedback_file: Path to feedback CSV
        min_samples: Minimum samples needed
        auto_finetune: Whether to automatically fine-tune
    
    Returns:
        True if model was updated, False otherwise
    """
    feedback_path = Path(feedback_file)
    
    if not feedback_path.exists():
        return False
    
    try:
        df = pd.read_csv(feedback_file)
        
        num_samples = len(df)
        print(f"\nFeedback collected: {num_samples} samples")
        
        if num_samples < min_samples:
            remaining = min_samples - num_samples
            print(f"Need {remaining} more samples to fine-tune (minimum: {min_samples})")
            return False
        
        if auto_finetune:
            # Fine-tune the model
            return fine_tune_model_with_feedback(
                model, 
                feedback_file=feedback_file,
                epochs=10,  # Fewer epochs for fine-tuning
                batch_size=8,
                learning_rate=0.0001,  # Lower learning rate
                min_samples=min_samples
            )
        else:
            print(f"Enough samples ({num_samples}) collected. Ready for fine-tuning.")
            return True
    
    except Exception as e:
        print(f"Error updating model: {e}")
        return False

