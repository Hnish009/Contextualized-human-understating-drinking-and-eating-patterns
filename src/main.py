

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
sys.path.append('.')
from src.utils import (
    detect_face, crop_face, decode_mood, format_arduino_output,
    save_feedback, encode_gender
)
from src.model2_drink import prepare_input_features
from src.arduino_comm import send_to_arduino  # Placeholder for now
from src.feedback_learning import update_model_with_feedback


def load_models():
    """Load trained models"""
    print("Loading models...")
    
    # Load Model 1: Age & Mood
    try:
        # Try loading with custom_objects to handle version compatibility
        model1 = tf.keras.models.load_model(
            'models/age_mood_model.h5',
            compile=False  # Don't compile, we'll compile manually if needed
        )
        # Recompile with compatible metrics
        model1.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'age_output': 'mae',
                'mood_output': 'categorical_crossentropy'
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
        print("OK: Model 1 (Age & Mood) loaded")
    except Exception as e:
        print(f"Error: Failed to load Model 1: {e}")
        print("Trying alternative loading method...")
        try:
            # Alternative: Load weights only
            from src.model1_age_mood import create_age_mood_model, compile_model
            model1 = create_age_mood_model()
            model1.load_weights('models/age_mood_model.h5')
            model1 = compile_model(model1)
            print("OK: Model 1 loaded using weights-only method")
        except Exception as e2:
            print(f"Error: Alternative loading also failed: {e2}")
            print("Please retrain Model 1 using: python src/train_model1.py")
            return None, None
    
    # Load Model 2: Drink Suggestion
    try:
        model2 = tf.keras.models.load_model(
            'models/drink_model.h5',
            compile=False
        )
        # Recompile
        model2.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'cosine_similarity']
        )
        print("OK: Model 2 (Drink Suggestion) loaded")
    except Exception as e:
        print(f"Error: Failed to load Model 2: {e}")
        print("Trying alternative loading method...")
        try:
            from src.model2_drink import create_drink_model, compile_model
            model2 = create_drink_model()
            model2.load_weights('models/drink_model.h5')
            model2 = compile_model(model2)
            print("OK: Model 2 loaded using weights-only method")
        except Exception as e2:
            print(f"Error: Alternative loading also failed: {e2}")
            print("Please retrain Model 2 using: python src/train_model2.py")
            return None, None
    
    return model1, model2


def predict_age_mood(model, face_image):
    """
    Predict age and mood from face image
    Returns: (age, mood_idx, mood_one_hot)
    """
    # Reshape for model input: (1, 64, 64, 1)
    face_batch = face_image.reshape(1, 64, 64, 1)
    
    # Predict
    age_pred, mood_pred = model.predict(face_batch, verbose=0)
    
    # Extract values
    age = float(age_pred[0][0])
    mood_idx = int(np.argmax(mood_pred[0]))
    mood_one_hot = mood_pred[0]
    
    return age, mood_idx, mood_one_hot


def predict_drink(model, gender, age, mood_one_hot):
    """
    Predict drink ratios from user features
    Returns: numpy array of 7 ratios
    """
    # Prepare input features
    features = prepare_input_features(gender, age, mood_one_hot)
    
    # Predict
    ratios = model.predict(features, verbose=0)[0]
    
    return ratios




def main():
    """Main application loop"""
    print("=" * 60)
    print(" : Contextualized Human Understanding for Drinks")
    print("=" * 60)
    
    # Load models
    model1, model2 = load_models()
    if model1 is None or model2 is None:
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    
    print("\n📷 Camera opened. Press 'c' to capture, 'q' to quit")
    print("💡 Instructions:")
    print("   1. Position face in front of camera")
    print("   2. Press 'c' to capture → Get drink suggestion → Rate it → Done")
    print("   3. Press 'q' to quit\n")
    
    # State variables
    current_face = None
    current_age = None
    current_mood = None
    current_mood_idx = None
    current_ratios = None
    current_gender = 0  # Default to male, can be toggled with 'g'
    capture_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect face
        face_rect = detect_face(frame)
        
        # Draw face rectangle
        if face_rect is not None and len(face_rect) == 4:
            x, y, w, h = face_rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Crop and process face
            face_img = crop_face(frame, face_rect)
            current_face = face_img
        
        # Display info
        if current_age is not None:
            mood_str = decode_mood(current_mood_idx)
            cv2.putText(frame, f"Age: {current_age:.0f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Mood: {mood_str}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Gender: {'Female' if current_gender else 'Male'}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if current_ratios is not None:
            # Display ratios
            bottles = ['Sugar', 'Salt', 'Lemon', 'Soda', 'Mint', 'Ginger', 'Masala']
            y_offset = 120
            for i, (bottle, ratio) in enumerate(zip(bottles, current_ratios)):
                cv2.putText(frame, f"{bottle}: {ratio:.2f}", (10, y_offset + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'c' to capture | 'g' to toggle gender | 'q' to quit",
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow('  - Drink Suggestion System', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('c'):
            # Capture and predict
            if current_face is not None:
                print("\n🔍 Analyzing face...")
                age, mood_idx, mood_one_hot = predict_age_mood(model1, current_face)
                
                current_age = age
                current_mood_idx = mood_idx
                current_mood = mood_one_hot
                
                print(f"✓ Age: {age:.0f} years")
                print(f"✓ Mood: {decode_mood(mood_idx)}")
                
                # Predict drink
                ratios = predict_drink(model2, current_gender, age, mood_one_hot)
                current_ratios = ratios
                
                print(f"\n🥤 Suggested Drink Ratios:")
                bottles = ['Sugar', 'Salt', 'Lemon', 'Soda', 'Mint', 'Ginger', 'Masala']
                for bottle, ratio in zip(bottles, ratios):
                    print(f"   {bottle}: {ratio:.2%}")
                
                # Format for Arduino (placeholder)
                arduino_str = format_arduino_output(ratios)
                print(f"\n📤 Arduino command: {arduino_str.strip()}")
                # send_to_arduino(arduino_str)  # Uncomment when Arduino is ready
                
                # Automatically ask for feedback
                print("\n" + "=" * 60)
                print("⭐ Please rate this drink suggestion (1-10): ", end='')
                try:
                    rating = int(input().strip())
                    if 1 <= rating <= 10:
                        # Save feedback
                        save_feedback(
                            current_gender, int(current_age), current_mood_idx,
                            current_ratios, rating
                        )
                        
                        # Update model with feedback (fine-tune every 10 samples)
                        try:
                            import pandas as pd
                            feedback_count = len(pd.read_csv('data/feedback.csv'))
                            if feedback_count % 10 == 0 and feedback_count >= 10:
                                print(f"\n🎯 {feedback_count} feedback samples collected! Fine-tuning model...")
                                update_model_with_feedback(model2, min_samples=10, auto_finetune=True)
                            else:
                                remaining = 10 - (feedback_count % 10)
                                print(f"  ({remaining} more ratings needed for next fine-tuning)")
                        except Exception as e:
                            print(f"  Note: Could not check feedback status: {e}")
                        
                        print(f"\n✓ Thank you for your feedback! Rating: {rating}/10")
                        print("=" * 60)
                    else:
                        print("❌ Rating must be between 1 and 10")
                except ValueError:
                    print("❌ Invalid input. Please enter a number between 1 and 10")
                except KeyboardInterrupt:
                    print("\n\nRating cancelled.")
                
                # Close camera and exit after feedback
                print("\n👋 Closing camera...")
                cap.release()
                cv2.destroyAllWindows()
                break
        
        elif key == ord('g'):
            # Toggle gender
            current_gender = 1 - current_gender
            gender_str = "Female" if current_gender else "Male"
            print(f"✓ Gender changed to: {gender_str}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Application closed. Thank you!")


if __name__ == '__main__':
    main()

