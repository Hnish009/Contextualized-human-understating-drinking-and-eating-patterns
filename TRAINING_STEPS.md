# Training Steps - Quick Guide

## Step 1: Convert Your Drink Data
 
Since you already have the data in `data/drinks_raw.txt`, you need to convert it to CSV format.

**Option A: Manual conversion (Easiest)**
1. Open `data/drinks_raw.txt` 
2. Copy all the entries
3. Open Excel or Google Sheets
4. Create columns: gender, age, mood, drink
5. Paste and format data
6. Save as `data/drinks.csv`

**Option B: Use Python script** (if you have pandas installed)
```bash
python src/convert_drinks.py
```

The CSV should look like:
```csv
gender,age,mood,drink
male,30,happy,Cool Fusion Lemon Mix
female,30,sad,Mint Salt Cooler
...
```

**Important**: 
- Mood values must be: angry, disgust, fear, happy, sad, surprise, neutral
- Drink names must match exactly the 10 drinks in recipes.json

---

## Step 2: Install Dependencies (If Not Done)

```bash
pip install -r requirements.txt
```

**Note**: If you get errors, try installing packages one by one:
```bash
pip install opencv-python numpy tensorflow matplotlib pandas scikit-learn pillow tqdm
```

---

## Step 3: Train Model 1 (Age + Mood Detection)

```bash
python src/train_model1.py
```

**What it does**:
- Loads FER2013 (emotions) and UTKFace (ages) datasets
- Trains CNN for ~50 epochs
- Takes 30-60 min on GPU, 2-4 hours on CPU
- Saves model to `models/age_mood_model.h5`
- Creates graphs in `models/training_history/`

**Expected output**:
- Training progress with loss/accuracy
- Final metrics printed
- Graphs saved automatically

---

## Step 4: Train Model 2 (Drink Suggestion)

```bash
python src/train_model2.py
```

**What it does**:
- Loads your `data/drinks.csv`
- Trains MLP for ~100 epochs  
- Takes 5-15 minutes
- Saves model to `models/drink_model.h5`
- Creates graphs

**Expected output**:
- Training progress
- Final metrics
- Graphs saved

---

## Step 5: Run the Application!

```bash
python run_project.py
```

Or:
```bash
python src/main.py
```

**Controls**:
- `c` - Capture face and get drink suggestion
- `g` - Toggle gender
- `r` - Rate drink (1-10)
- `q` - Quit

---

## Troubleshooting

### "ModuleNotFoundError"
- Install dependencies: `pip install -r requirements.txt`

### "Dataset not found"
- Check FER2013 is in `data/fer2013/train/` and `data/fer2013/test/`
- Check UTK images are in `data/utk/`
- Check drinks.csv exists in `data/`

### "Models not found"
- Make sure you trained both models first!

### Training too slow
- Reduce epochs in training scripts
- Use fewer samples (limit UTK to 5000-10000 images)

---

## For PPT Presentation

After training, check `models/training_history/`:
- `model1_training_history.png` - Age & Mood training curves
- `model1_info.json` - Final metrics and architecture
- `model2_training_history.png` - Drink model curves
- `model2_info.json` - Final metrics

