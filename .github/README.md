# Contextualized Human Understanding for Drinks and Eating Pattern

An AI-powered drink suggestion system that detects a person's age and mood from their face, then suggests a personalized drink ratio.

## 🎯 Project Overview

**Goal**: Just like your mom cooks food by seeing your face and your mood lightens up, our AI tells you a drink to drink!

**Working Flow**:
1. 📷 Image capture → Detect person's mood and age
2. 🥤 Suggest user a drink ratio (7 flavored liquids)
3. 🔧 Machine with 7 pumps mixes the drink
4. ⭐ User rates (1-10) → System learns and improves

## 📁 Project Structure

```
Project/
├── data/
│   ├── fer2013/          # FER2013 emotion dataset (download separately)
│   ├── utk/              # UTKFace age dataset (download separately)
│   ├── drinks.csv        # Custom drink dataset (400 samples)
│   └── recipes.json      # Predefined drink recipes
├── models/
│   └── training_history/  # Training graphs and metrics (generated)
├── src/
│   ├── model1_age_mood.py       # CNN architecture for age + mood
│   ├── model2_drink.py          # MLP architecture for drink suggestion
│   ├── train_model1.py          # Training script for Model 1
│   ├── train_model2.py          # Training script for Model 2
│   ├── preprocess_data.py       # Data loading and preprocessing
│   ├── main.py                  # Main camera application
│   ├── utils.py                 # Helper functions
│   ├── feedback_learning.py     # Feedback learning mechanism
│   └── arduino_comm.py          # Arduino communication
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets

- **FER2013**: Download from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) → Place in `data/fer2013/`
- **UTKFace**: Download from [UTKFace](https://susanqq.github.io/UTKFace/) → Place in `data/utk/`
- **Drinks Dataset**: Create `data/drinks.csv` with columns: `gender,age,mood,drink`

### 3. Train Models

```bash
# Train Model 1 (Age + Mood Detection)
python src/train_model1.py

# Train Model 2 (Drink Suggestion)
python src/train_model2.py
```

### 4. Run Application

```bash
python run_project.py
```

## 🎮 Usage

1. Position your face in front of camera
2. Press `c` to capture → System detects age and mood
3. System suggests drink ratios (7 bottles)
4. Rate the drink (1-10) → Feedback saved automatically
5. System fine-tunes after every 10 ratings

## 📊 Models

- **Model 1**: CNN with dual outputs (age regression + mood classification)
- **Model 2**: MLP for drink suggestion with feedback learning

## 📝 License

Metalicense


