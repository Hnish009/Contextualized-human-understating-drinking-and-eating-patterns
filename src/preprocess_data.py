"""
Data preprocessing for Chudai project
Handles FER2013, UTK, and custom drink dataset loading and preprocessing
"""


import numpy as np
import pandas as pd
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import sys
sys.path.append('.')
from src.utils import encode_gender, encode_mood, get_drink_ratio


def load_fer2013(data_path='data/fer2013', max_samples=None):
    """
    Load FER2013 dataset for emotion detection
    Expected structure: 
    - data/fer2013/fer2013.csv (CSV format)
    - OR data/fer2013/train/ and data/fer2013/test/ with emotion folders
    """
    print("Loading FER2013 dataset...")
    
    fer_path = Path(data_path)
    fer_csv = fer_path / 'fer2013.csv'
    
    # Try CSV format first
    if fer_csv.exists():
        df = pd.read_csv(fer_csv)
        
        if max_samples:
            df = df.head(max_samples)
        
        images = []
        emotions = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing FER2013"):
            # FER2013 format: pixels are space-separated string
            pixels = np.array(row['pixels'].split(), dtype=np.uint8)
            img = pixels.reshape(48, 48)
            
            # Resize to 64x64
            img = cv2.resize(img, (64, 64))
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            emotions.append(row['emotion'])  # 0-6: angry, disgust, fear, happy, sad, surprise, neutral
        
        images = np.array(images)
        emotions = np.array(emotions)
        
        # Convert to one-hot
        emotions_one_hot = tf.keras.utils.to_categorical(emotions, num_classes=7)
        
        print(f"✓ Loaded {len(images)} FER2013 images from CSV")
        return images, emotions_one_hot
    
    # Try folder structure (train/test with emotion folders)
    train_path = fer_path / 'train'
    test_path = fer_path / 'test'
    
    if train_path.exists() and test_path.exists():
        emotion_map = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'sad': 4,
            'surprise': 5,
            'neutral': 6
        }
        
        images = []
        emotions = []
        
        # Load from train and test folders
        for split in ['train', 'test']:
            split_path = fer_path / split
            for emotion_name, emotion_idx in emotion_map.items():
                emotion_folder = split_path / emotion_name
                if emotion_folder.exists():
                    image_files = list(emotion_folder.glob('*.jpg')) + list(emotion_folder.glob('*.png'))
                    
                    if max_samples:
                        # Limit samples per emotion
                        per_emotion_limit = max_samples // 7
                        image_files = image_files[:per_emotion_limit]
                    
                    for img_path in tqdm(image_files, desc=f"Loading {split}/{emotion_name}"):
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        
                        img = cv2.resize(img, (64, 64))
                        img = img.astype(np.float32) / 255.0
                        
                        images.append(img)
                        emotions.append(emotion_idx)
        
        images = np.array(images)
        emotions = np.array(emotions)
        
        # Convert to one-hot
        emotions_one_hot = tf.keras.utils.to_categorical(emotions, num_classes=7)
        
        print(f"✓ Loaded {len(images)} FER2013 images from folders")
        return images, emotions_one_hot
    
    else:
        print(f"⚠ FER2013 not found at {fer_path}")
        print("Expected: fer2013.csv OR train/test folders with emotion subfolders")
        return None, None


def load_utk(data_path='data/utk', max_samples=None):
    """
    Load UTKFace dataset for age estimation
    Expected structure: data/utk/*.jpg files with naming: [age]_[gender]_[race]_[date&time].jpg
    """
    print("Loading UTKFace dataset...")
    
    utk_path = Path(data_path)
    image_files = list(utk_path.glob('*.jpg')) + list(utk_path.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"⚠ No images found in {data_path}")
        print("Please ensure UTKFace images are in data/utk/")
        return None, None
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    images = []
    ages = []
    
    for img_path in tqdm(image_files, desc="Processing UTKFace"):
        try:
            # Parse filename: [age]_[gender]_[race]_[date&time].jpg
            filename = img_path.stem
            parts = filename.split('_')
            
            if len(parts) >= 1:
                age = int(parts[0])
                
                # Load and preprocess image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (64, 64))
                gray = gray.astype(np.float32) / 255.0
                
                images.append(gray)
                ages.append(age)
        
        except (ValueError, IndexError) as e:
            continue
    
    images = np.array(images)
    ages = np.array(ages)
    
    print(f"✓ Loaded {len(images)} UTKFace images")
    return images, ages


def load_drink_data(data_path='data/drinks.csv'):
    """
    Load custom drink dataset
    Expected format: CSV with columns: gender, age, mood, drink
    Or JSON array with format: {"gender: male, age: 15, mood: angry, drink: Mint Lemon Cooler"}
    """
    print("Loading custom drink dataset...")
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"⚠ Drink data not found at {data_path}")
        print("Please create drinks.csv or drinks.json in data/ folder")
        return None
    
    # Try CSV first
    if data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
        
        # Expected columns: gender, age, mood, drink
        required_cols = ['gender', 'age', 'mood', 'drink']
        
        if not all(col in df.columns for col in required_cols):
            print(f"⚠ CSV must have columns: {required_cols}")
            return None
        
        X_features = []
        y_ratios = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing drink data"):
            gender = encode_gender(row['gender'])
            age = int(row['age'])
            mood_one_hot, _ = encode_mood(row['mood'])
            
            # Get drink ratio
            drink_name = row['drink']
            ratio = get_drink_ratio(drink_name)
            
            # Prepare input features: [gender, age_normalized, mood_one_hot]
            features = np.concatenate([[gender], [age/100.0], mood_one_hot])
            
            X_features.append(features)
            y_ratios.append(ratio)
        
        X_features = np.array(X_features)
        y_ratios = np.array(y_ratios)
        
        print(f"✓ Loaded {len(X_features)} drink data samples")
        return X_features, y_ratios
    
    # Try JSON
    elif data_path.suffix == '.json':
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        X_features = []
        y_ratios = []
        
        for item in tqdm(data, desc="Processing drink data"):
            # Parse format: {"gender: male, age: 15, mood: angry, drink: Mint Lemon Cooler"}
            if isinstance(item, str):
                # Parse string format
                parts = item.strip('{}').split(',')
                gender_str = parts[0].split(':')[1].strip()
                age = int(parts[1].split(':')[1].strip())
                mood_str = parts[2].split(':')[1].strip()
                drink_str = parts[3].split(':')[1].strip()
            else:
                # Already dict
                gender_str = item['gender']
                age = item['age']
                mood_str = item['mood']
                drink_str = item['drink']
            
            gender = encode_gender(gender_str)
            mood_one_hot, _ = encode_mood(mood_str)
            
            ratio = get_drink_ratio(drink_str)
            
            features = np.concatenate([[gender], [age/100.0], mood_one_hot])
            X_features.append(features)
            y_ratios.append(ratio)
        
        X_features = np.array(X_features)
        y_ratios = np.array(y_ratios)
        
        print(f"✓ Loaded {len(X_features)} drink data samples")
        return X_features, y_ratios
    
    else:
        print(f"⚠ Unsupported file format: {data_path.suffix}")
        return None


def combine_fer_utk(fer_images, fer_emotions, utk_images, utk_ages):
    """
    Combine FER2013 and UTK datasets for training Model 1
    Since FER has emotions but no age, and UTK has age but no emotions,
    we'll train on both separately or use a combined approach
    """
    print("Combining FER2013 and UTK datasets...")
    
    # Strategy: Use FER for mood training, UTK for age training
    # For combined training, we'll use UTK images and estimate emotions
    # or use FER images and estimate ages (simpler approach)
    
    # For now, return both separately - training script will handle it
    return {
        'fer_images': fer_images,
        'fer_emotions': fer_emotions,
        'utk_images': utk_images,
        'utk_ages': utk_ages
    }

