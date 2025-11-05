# Chudai: Contextualized Human Understanding for Drinks and Eating Pattern

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
chudai/
├── data/
│   ├── fer2013/          # FER2013 emotion dataset (you need to download)
│   ├── utk/              # UTKFace age dataset (you need to download)
│   ├── drinks.csv        # Your 400 custom drink data (create this)
│   ├── recipes.json      # Predefined drink recipes
│   └── feedback.csv      # User feedback (auto-generated)
├── models/
│   ├── age_mood_model.h5        # Trained Model 1 (age + mood)
│   ├── drink_model.h5           # Trained Model 2 (drink suggestion)
│   └── training_history/        # Training graphs and metrics
├── src/
│   ├── model1_age_mood.py       # CNN architecture for age + mood
│   ├── model2_drink.py          # MLP architecture for drink suggestion
│   ├── train_model1.py          # Training script for Model 1
│   ├── train_model2.py          # Training script for Model 2
│   ├── preprocess_data.py       # Data loading and preprocessing
│   ├── main.py                  # Main camera application
│   ├── utils.py                 # Helper functions
│   └── arduino_comm.py          # Arduino communication
├── requirements.txt
└── README.md
```

## 🚀 Setup Instructions

### Step 1: Install Python and Dependencies

1. **Python Version**: Make sure you have Python 3.8-3.10 installed
   ```bash
   python --version
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: If you have CUDA/GPU support, install TensorFlow GPU:
   ```bash
   pip install tensorflow-gpu==2.13.0
   ```

### Step 2: Prepare Datasets

#### Download FER2013 Dataset:
- Go to: https://www.kaggle.com/datasets/msambare/fer2013
- Download `fer2013.csv`
- Place it in `data/fer2013/fer2013.csv`

#### Download UTKFace Dataset:
- Go to: https://susanqq.github.io/UTKFace/
- Download images (or use a subset for faster training)
- Place all images in `data/utk/` folder
- Format: `[age]_[gender]_[race]_[date&time].jpg`

#### Create Your Custom Drink Dataset:
Create `data/drinks.csv` with your 400 data points. Format:

```csv
gender,age,mood,drink
male,15,angry,Mint Lemon Cooler
female,25,happy,Classic Lemonade
male,30,neutral,Ginger Lemon Sparkler
...
```

**Columns**:
- `gender`: male or female
- `age`: integer (0-100)
- `mood`: angry, disgust, fear, happy, sad, surprise, or neutral
- `drink`: One of the 10 predefined drinks (see recipes.json)

**Or use JSON format** (`data/drinks.json`):
```json
[
  {"gender": "male", "age": 15, "mood": "angry", "drink": "Mint Lemon Cooler"},
  {"gender": "female", "age": 25, "mood": "happy", "drink": "Classic Lemonade"}
]
```

## 🏋️ Training the Models

### Train Model 1 (Age + Mood Detection):

```bash
python src/train_model1.py
```

**What it does**:
- Loads FER2013 and UTKFace datasets
- Trains CNN with dual outputs (age regression + mood classification)
- Saves model to `models/age_mood_model.h5`
- Generates training graphs in `models/training_history/`

**Expected output**:
- Training progress with epochs, loss, accuracy
- Final metrics (Age MAE, Mood Accuracy)
- Graphs saved as PNG files for PPT presentation

### Train Model 2 (Drink Suggestion):

```bash
python src/train_model2.py
```

**What it does**:
- Loads your custom drink dataset
- Trains MLP to predict 7 bottle ratios
- Saves model to `models/drink_model.h5`
- Generates training graphs

**Expected output**:
- Training progress
- Final metrics (Loss, MAE)
- Graphs for PPT

## 🎮 Running the Main Application

Once both models are trained:

```bash
python src/main.py
```

**Controls**:
- `c` - Capture face and get drink suggestion
- `g` - Toggle gender (male/female)
- `r` - Rate the drink (1-10)
- `q` - Quit application

**Workflow**:
1. Position your face in front of camera
2. Press `c` to capture → System detects age and mood
3. System suggests drink ratios (7 bottles)
4. Press `r` to rate → Feedback saved for learning
5. After 100-200 uses, system improves automatically!

## 📊 Model Details for PPT

All training information is saved in `models/training_history/`:

- **Model 1 Summary**: `model1_summary.txt`
- **Model 1 Info**: `model1_info.json` (epochs, parameters, metrics)
- **Model 1 Graphs**: `model1_training_history.png`
- **Model 2 Summary**: `model2_summary.txt`
- **Model 2 Info**: `model2_info.json`
- **Model 2 Graphs**: `model2_training_history.png`

### Model 1 Architecture:
- **Input**: 64x64 grayscale face image
- **Backbone**: 4 Conv layers + BatchNorm + Dropout
- **Outputs**: 
  - Age: Regression (MAE loss)
  - Mood: 7-class classification (Crossentropy loss)
- **Total Parameters**: ~2-3 million

### Model 2 Architecture:
- **Input**: 9 features (gender, age, mood one-hot)
- **Hidden Layers**: 4 Dense layers (128 → 256 → 128 → 64)
- **Output**: 7 bottle ratios (softmax, sums to 1.0)
- **Total Parameters**: ~50-100k

## 🔧 Arduino Integration (Future)

When ready to connect Arduino:

1. **Update port** in `src/arduino_comm.py`:
   ```python
   send_to_arduino(command, port='COM3')  # Change COM3 to your port
   ```

2. **Uncomment** in `src/main.py`:
   ```python
   send_to_arduino(arduino_str)
   ```

3. **Arduino Code** (you'll write this):
   - Receive command: `B1=0.20;B2=0.30;...`
   - Parse ratios
   - Control 7 pumps accordingly

## 📈 Feedback Learning

Feedback is automatically saved to `data/feedback.csv`. The system collects:
- User features (gender, age, mood)
- Predicted ratios
- User rating (1-10)

After collecting enough feedback, the model can be fine-tuned for better predictions.

## 🐛 Troubleshooting

### Camera not opening:
- Check if another application is using the camera
- Try changing camera index: `cv2.VideoCapture(1)`

### Models not loading:
- Make sure you've trained both models first
- Check that model files exist in `models/` folder

### Dataset loading errors:
- Verify dataset paths are correct
- Check file formats (CSV/JSON)
- Ensure required columns are present

### GPU not being used:
- Install TensorFlow GPU version
- Check CUDA/cuDNN installation
- Verify GPU is detected: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

## 📝 Notes

- **Simple Architecture**: Models are kept simple for easy understanding and fast training
- **Accuracy**: Good enough for demonstration with proper training
- **Feedback Learning**: System improves with more user interactions
- **7 Bottles**: Sugar, Salt, Lemon, Soda, Mint, Ginger, Masala

## 🎓 For Presentation

Use the graphs and metrics from `models/training_history/`:
- Training curves (loss, accuracy)
- Model architecture summaries
- Final performance metrics
- Epochs trained, parameters count

## 📧 Questions?

If you encounter any issues:
1. Check error messages carefully
2. Verify all datasets are in correct format
3. Ensure Python version is compatible (3.8-3.10)
4. Make sure all dependencies are installed

---

**Good luck with your college project! 🎉**

