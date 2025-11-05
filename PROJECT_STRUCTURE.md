# Project Structure

## ✅ Core Files (Required for Running)

### Essential Code
- `src/` - All source code
  - `model1_age_mood.py` - CNN architecture
  - `model2_drink.py` - MLP architecture
  - `train_model1.py` - Training script for Model 1
  - `train_model2.py` - Training script for Model 2
  - `preprocess_data.py` - Data loading
  - `main.py` - Main application
  - `utils.py` - Helper functions
  - `feedback_learning.py` - Feedback learning
  - `arduino_comm.py` - Arduino communication

### Configuration
- `requirements.txt` - Python dependencies
- `run_project.py` - Main entry point
- `data/recipes.json` - Drink recipes
- `data/drinks.csv` - Your dataset (create this)

### Documentation
- `README.md` - Main documentation
- `QUICK_START.md` - Quick start guide
- `TRAINING_STEPS.md` - Training instructions

## 📝 Optional Files (Development/Presentation)

### Optional Scripts (in `optional/` folder)
- Graph generation scripts
- Utility scripts
- Data conversion scripts

### Documentation
- `OPTIONAL_SCRIPTS.md` - Optional scripts documentation
- `HOW_TO_CHECK_GPU.md` - GPU checking guide
- `GITHUB_UPLOAD.md` - GitHub upload guide

## ❌ Excluded from GitHub (via .gitignore)

- `models/*.h5` - Trained models (too large)
- `models/training_history/*.png` - Generated graphs
- `data/feedback.csv` - User feedback data
- `data/fer2013/` - Dataset (download separately)
- `data/utk/` - Dataset (download separately)
- `__pycache__/` - Python cache
- `*.pyc` - Compiled files

## 🚀 What Others Need to Run

1. Clone repository
2. Download datasets (FER2013, UTKFace)
3. Create `data/drinks.csv`
4. Install: `pip install -r requirements.txt`
5. Train: `python src/train_model1.py` and `python src/train_model2.py`
6. Run: `python run_project.py`

## 💻 What You Keep Locally

Keep everything on your PC:
- Trained models
- Generated graphs
- Feedback data
- All datasets

The `.gitignore` ensures these won't be uploaded to GitHub.

