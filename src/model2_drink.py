"""
Model 2: Drink Suggestion MLP
Takes age, mood, and gender as input, outputs 7 bottle ratios
Includes feedback learning mechanism
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_drink_model():
    """
    Create MLP model for drink suggestion
    Input: [gender (1), age (1), mood_one_hot (7)] = 9 features
    Output: 7 bottle ratios (softmax to ensure sum to 1.0)
    """
    
    # Input layer: gender (1) + age (1) + mood one-hot (7) = 9 features
    inputs = keras.Input(shape=(9,), name='user_features')
    
    # Hidden layers
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    
    # Output layer: 7 bottle ratios (softmax ensures they sum to 1.0)
    # Order: [Sugar, Salt, Lemon, Soda, Mint, Ginger, Masala]
    outputs = layers.Dense(7, activation='softmax', name='bottle_ratios')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='drink_suggester')
    
    return model


def compile_model(model, learning_rate=0.001):
    """Compile model with appropriate loss and metrics"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',  # Mean Squared Error for ratio prediction
        metrics=['mae', 'cosine_similarity']  # Additional metrics
    )
    
    return model


def prepare_input_features(gender, age, mood_one_hot):
    """
    Prepare input features for drink model
    Inputs:
    - gender: 0 (male) or 1 (female)
    - age: integer (will be normalized)
    - mood_one_hot: numpy array of 7 values (one-hot encoded mood)
    
    Returns: numpy array of shape (9,)
    """
    # Normalize age to 0-1 range (assuming age range 0-100)
    age_normalized = age / 100.0
    
    # Combine: [gender, age_normalized, mood_one_hot]
    features = np.concatenate([[gender], [age_normalized], mood_one_hot])
    
    return features.reshape(1, -1)  # Return as batch of 1


def get_model_summary(model):
    """Print and return model summary"""
    model.summary()
    return model

