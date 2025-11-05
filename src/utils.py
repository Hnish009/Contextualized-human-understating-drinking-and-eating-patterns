"""
Utility functions for Chudai project
Handles data loading, preprocessing, and helper functions
"""

import json
import numpy as np
import cv2
from pathlib import Path
import pandas as pd


def load_recipes():
    """Load drink recipes from JSON file"""
    with open('data/recipes.json', 'r') as f:
        return json.load(f)


def normalize_ratios(ratios_dict):
    """
    Normalize drink ratios to sum to 1.0
    Input: dict with ingredient names as keys and values 0-100
    Output: numpy array of 7 normalized ratios [Sugar, Salt, Lemon, Soda, Mint, Ginger, Masala]
    """
    recipes = load_recipes()
    bottle_order = recipes['bottle_order']
    
    # Convert to list in correct order
    ratios = [ratios_dict.get(bottle, 0) for bottle in bottle_order]
    
    # Normalize to sum to 1.0
    total = sum(ratios)
    if total == 0:
        # If all zeros, return equal distribution
        return np.array([1/7] * 7)
    
    return np.array(ratios) / total


def get_drink_ratio(drink_name):
    """
    Get normalized ratio array for a specific drink
    Returns: numpy array of 7 ratios
    """
    recipes = load_recipes()
    if drink_name in recipes['drinks']:
        return normalize_ratios(recipes['drinks'][drink_name])
    else:
        # Return default balanced ratio if drink not found
        return np.array([1/7] * 7)


def encode_gender(gender_str):
    """Encode gender: male=0, female=1"""
    gender_str = gender_str.lower().strip()
    return 0 if gender_str == 'male' else 1


def encode_mood(mood_str):
    """
    Encode mood to one-hot vector
    FER2013 emotions: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    """
    mood_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }
    mood_str = mood_str.lower().strip()
    mood_idx = mood_map.get(mood_str, 6)  # Default to neutral
    
    # Return one-hot encoded vector (7 dimensions)
    one_hot = np.zeros(7)
    one_hot[mood_idx] = 1
    return one_hot, mood_idx


def decode_mood(mood_idx):
    """Decode mood index to string"""
    moods = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return moods[mood_idx] if 0 <= mood_idx < 7 else 'Neutral'


def detect_face(image):
    """
    Detect face in image using OpenCV Haar Cascade
    Returns: numpy array (x, y, w, h) bounding box or None
    """
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        # Return the largest face as numpy array
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        return largest_face
    return None


def crop_face(image, face_rect):
    """
    Crop and resize face to 64x64 for model input
    Returns: 64x64 grayscale image (numpy array)
    """
    x, y, w, h = face_rect
    face = image[y:y+h, x:x+w]
    
    # Convert to grayscale if needed
    if len(face.shape) == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Resize to 64x64
    face_resized = cv2.resize(face, (64, 64))
    
    # Normalize to 0-1 range
    face_normalized = face_resized.astype(np.float32) / 255.0
    
    return face_normalized


def format_arduino_output(ratios):
    """
    Format ratios for Arduino serial communication
    Input: numpy array of 7 ratios
    Output: string like "B1=0.20;B2=0.30;B3=0.10;B4=0.10;B5=0.10;B6=0.10;B7=0.10\n"
    """
    ratios_str = ';'.join([f"B{i+1}={ratios[i]:.2f}" for i in range(7)])
    return ratios_str + '\n'


def save_feedback(gender, age, mood_idx, predicted_ratios, user_rating, feedback_file='data/feedback.csv'):
    """
    Save user feedback to CSV file for continuous learning
    """
    feedback_path = Path(feedback_file)
    
    # Create DataFrame with feedback data
    feedback_data = {
        'gender': [gender],
        'age': [age],
        'mood': [mood_idx],
        'sugar': [predicted_ratios[0]],
        'salt': [predicted_ratios[1]],
        'lemon': [predicted_ratios[2]],
        'soda': [predicted_ratios[3]],
        'mint': [predicted_ratios[4]],
        'ginger': [predicted_ratios[5]],
        'masala': [predicted_ratios[6]],
        'rating': [user_rating]
    }
    
    df = pd.DataFrame(feedback_data)
    
    # Append to CSV (create if doesn't exist)
    if feedback_path.exists():
        df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        df.to_csv(feedback_file, mode='w', header=True, index=False)
    
    # Count total feedback
    try:
        total_feedback = len(pd.read_csv(feedback_file))
        print(f"OK: Feedback saved: Rating {user_rating}/10 (Total: {total_feedback} ratings)")
    except:
        print(f"OK: Feedback saved: Rating {user_rating}/10")

