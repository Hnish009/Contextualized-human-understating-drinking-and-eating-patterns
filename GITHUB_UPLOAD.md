# GitHub Upload Guide

## Files to Upload

### ✅ Core Files (Required)
- `src/` - All source code
- `requirements.txt` - Dependencies
- `README.md` - Main documentation
- `run_project.py` - Main entry point
- `data/recipes.json` - Drink recipes
- `data/drinks_template.csv` - Template for dataset
- `QUICK_START.md` - Quick start guide
- `TRAINING_STEPS.md` - Training instructions
- `.gitignore` - Git ignore rules

### ❌ Don't Upload (Excluded by .gitignore)
- `models/*.h5` - Trained models (too large)
- `models/training_history/*.png` - Generated graphs
- `data/feedback.csv` - User feedback data
- `data/fer2013/` - Dataset (download separately)
- `data/utk/` - Dataset (download separately)
- `__pycache__/` - Python cache
- `*.pyc` - Compiled Python files

### 📝 Optional Files (You can include)
- `OPTIONAL_SCRIPTS.md` - Documentation for optional scripts
- `HOW_TO_CHECK_GPU.md` - GPU checking guide
- Optional graph generation scripts (for development)

## What Others Need to Run

1. Clone repository
2. Download datasets (FER2013, UTKFace)
3. Create `data/drinks.csv` with their dataset
4. Install dependencies: `pip install -r requirements.txt`
5. Train models: `python src/train_model1.py` and `python src/train_model2.py`
6. Run: `python run_project.py`

## Your Local Files (Keep Everything)

On your PC, keep all files including:
- Trained models (`models/*.h5`)
- Generated graphs (`models/training_history/*.png`)
- Feedback data (`data/feedback.csv`)
- All datasets (`data/fer2013/`, `data/utk/`)

The `.gitignore` ensures these won't be uploaded to GitHub.

## Quick Upload Commands

```bash
# Initialize git (if not done)
git init

# Add all files (respects .gitignore)
git add .

# Commit
git commit -m "Initial commit:  project"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/yourusername/cd.git

# Push
git push -u origin main
```

