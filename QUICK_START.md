# 🚀 Quick Start Guide -  Project

## Step-by-Step Setup (For Beginners)

### ✅ Step 1: Check Python Installation

Open PowerShell/Command Prompt and type:
```bash
python --version
```
Should show Python 3.8, 3.9, or 3.10. If not, download from python.org

---

### ✅ Step 2: Install Dependencies

Navigate to your project folder:
```bash
cd E:\
```

Install all required packages:
```bash
pip install -r requirements.txt
```

**This will take 5-10 minutes** - it's downloading TensorFlow and other libraries.

---

### ✅ Step 3: Prepare Your Datasets

#### A. FER2013 Dataset (for mood detection):
1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Download `fer2013.csv`
3. Place it in: `E:\\data\fer2013\fer2013.csv`

#### B. UTKFace Dataset (for age detection):
1. Go to: https://susanqq.github.io/UTKFace/
2. Download images (you can use a subset - 1000-5000 images is enough)
3. Place all `.jpg` files in: `E:\\data\utk\`

#### C. Your Custom Drink Dataset:
1. Open `data/drinks_template.csv` as a reference
2. Create your `data/drinks.csv` with your 400 data points
3. Format: `gender,age,mood,drink` (one row per data point)

**Example format:**
```csv
gender,age,mood,drink
male,15,angry,Mint Lemon Cooler
female,25,happy,Classic Lemonade
male,30,neutral,Ginger Lemon Sparkler
```

**Available drinks** (from recipes.json):
- Classic Lemonade
- Mint Lemon Cooler
- Ginger Lemon Sparkler
- Salty Lemon Fizz
- Minty Ginger Punch
- Sweet Lemon Delight
- Tangy Masala Lemon
- Mint Salt Cooler
- Ginger Mint Twist
- Cool Fusion Lemon Mix

**Available moods:**
- angry
- disgust
- fear
- happy
- sad
- surprise
- neutral

---

### ✅ Step 4: Train Model 1 (Age + Mood Detection)

```bash
python src/train_model1.py
```

**What happens:**
- Loads FER2013 and UTKFace datasets
- Trains CNN for 50 epochs (this takes 30-60 minutes on GPU, 2-4 hours on CPU)
- Saves model to `models/age_mood_model.h5`
- Creates graphs in `models/training_history/`

**You'll see:**
- Training progress with loss and accuracy
- Final metrics when done
- Graphs saved automatically

---

### ✅ Step 5: Train Model 2 (Drink Suggestion)

```bash
python src/train_model2.py
```

**What happens:**
- Loads your `data/drinks.csv`
- Trains MLP for 100 epochs (takes 5-15 minutes)
- Saves model to `models/drink_model.h5`
- Creates graphs

---

### ✅ Step 6: Run the Application!

```bash
python run_project.py
```

**Or directly:**
```bash
python src/main.py
```

**Controls:**
- `c` - Capture face and get drink suggestion
- `g` - Toggle gender (male ↔ female)
- `r` - Rate the drink (1-10)
- `q` - Quit

**Workflow:**
1. Camera opens
2. Position your face in front of camera
3. Press `c` → System detects age and mood
4. System shows drink ratios (7 bottles)
5. Press `r` → Enter rating (1-10)
6. Feedback saved automatically!

---

## 📊 For PPT Presentation

After training, check these files in `models/training_history/`:

1. **`model1_training_history.png`** - Training curves for age and mood
2. **`model1_info.json`** - Final metrics, epochs, parameters
3. **`model1_summary.txt`** - Model architecture details
4. **`model2_training_history.png`** - Training curves for drink model
5. **`model2_info.json`** - Final metrics

Use these for your presentation slides!

---

## 🐛 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:** Run `pip install -r requirements.txt` again

### Issue: "Failed to open camera"
**Solution:** 
- Close other apps using camera (Zoom, Teams, etc.)
- Try changing camera index in `src/main.py`: `cv2.VideoCapture(1)` instead of `0`

### Issue: "Dataset not found"
**Solution:** 
- Check file paths are correct
- FER2013 should be at `data/fer2013/fer2013.csv`
- UTK images should be in `data/utk/`
- Drinks CSV should be at `data/drinks.csv`

### Issue: Training is too slow
**Solution:**
- Use fewer samples (limit UTK to 1000-5000 images)
- Reduce epochs in training scripts
- Use GPU if available (install tensorflow-gpu)

### Issue: "Models not found" when running main.py
**Solution:** 
- Make sure you trained both models first
- Check `models/age_mood_model.h5` and `models/drink_model.h5` exist

---

## 📝 Notes

- **First time training**: Takes 1-4 hours total (depending on GPU/CPU)
- **After training**: Models are saved, you only need to run main.py
- **Feedback learning**: System collects ratings in `data/feedback.csv`
- **Arduino**: Communication code is ready, just uncomment when hardware is ready

---

## 🎯 Next Steps After Setup

1. ✅ Train both models
2. ✅ Test with camera
3. ✅ Collect feedback (rate drinks 100-200 times)
4. ✅ Fine-tune model with feedback (optional)
5. ✅ Connect Arduino (when hardware ready)
6. ✅ Prepare PPT with graphs and metrics

---

## 💡 Tips

- Start with small datasets for testing (100-500 images)
- Use GPU for faster training (RTX 3060 will be much faster!)
- Test camera before presentation
- Keep feedback.csv for showing learning progress

---

**Good luck! 🎉**

