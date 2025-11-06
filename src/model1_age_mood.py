"""
Model 1: Age and Mood Detection CNN
Dual-output CNN: one head for age regression, one head for mood classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np



def create_age_mood_model(input_shape=(64, 64, 1)):
    """
    Create CNN model with dual outputs:
    - Age: Regression output (single value)
    - Mood: Classification output (7 classes: angry, disgust, fear, happy, sad, surprise, neutral)
    
    Architecture:
    - Shared CNN backbone (feature extraction)
    - Two separate heads (age regression + mood classification)
    """
    
    # Input layer
    inputs = keras.Input(shape=input_shape, name='face_image')
    
    # Shared CNN Backbone
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Flatten for dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Shared features
    shared_features = layers.Dense(256, activation='relu', name='shared_features')(x)
    
    # Age Regression Head
    age_branch = layers.Dense(128, activation='relu')(shared_features)
    age_branch = layers.Dropout(0.3)(age_branch)
    age_branch = layers.Dense(64, activation='relu')(age_branch)
    age_output = layers.Dense(1, activation='linear', name='age_output')(age_branch)
    
    # Mood Classification Head
    mood_branch = layers.Dense(128, activation='relu')(shared_features)
    mood_branch = layers.Dropout(0.3)(mood_branch)
    mood_branch = layers.Dense(64, activation='relu')(mood_branch)
    mood_output = layers.Dense(7, activation='softmax', name='mood_output')(mood_branch)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=[age_output, mood_output], name='age_mood_detector')
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile model with appropriate losses and metrics
    - Age: MAE (Mean Absolute Error) loss
    - Mood: Categorical Crossentropy loss
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'age_output': 'mae',  # Mean Absolute Error for age regression
            'mood_output': 'categorical_crossentropy'  # For mood classification
        },
        loss_weights={
            'age_output': 1.0,
            'mood_output': 1.0
        },
        metrics={
            'age_output': ['mae', 'mse'],
            'mood_output': ['accuracy']
        }
    )
    
    return model


def get_model_summary(model):
    """Print and return model summary"""
    model.summary()
    return model

