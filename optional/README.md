# Optional Scripts

These scripts are useful for development and presentation but **not required** for core functionality.

## Graph Generation Scripts

- `generate_presentation_graphs.py` - Creates all presentation graphs
- `generate_detailed_curves.py` - Creates detailed training curves from CSV logs  
- `create_curves_from_history.py` - Creates curves from training history

**Usage**: Run these after training to generate graphs for presentations.

## Utility Scripts

- `check_gpu.py` - Check GPU/CPU availability
- `check_progress.py` - Check training progress
- `monitor_training.py` - Monitor training status
- `fix_drinks_csv.py` - Fix CSV format issues
- `convert_drinks.py` / `convert_drinks_simple.py` / `ingest_drinks.py` - Data conversion utilities

## Core Functionality

The core project works **without these scripts**. They're only for:
- Development and debugging
- Presentation material generation
- Data conversion utilities

## To Run Core Project

Only these files are needed:
- `src/` folder (all source code)
- `requirements.txt`
- `run_project.py`
- `data/recipes.json`
- `data/drinks.csv` (your dataset)

Everything else is optional for development/presentation.

