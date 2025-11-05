"""
Simple entry point to run the Chudai project
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == '__main__':
    print("Chudai Project - Starting...")
    print("=" * 60)
    
    # Check if models exist
    from pathlib import Path
    
    model1_path = Path('models/age_mood_model.h5')
    model2_path = Path('models/drink_model.h5')
    
    if not model1_path.exists() or not model2_path.exists():
        print("⚠ Models not found!")
        print("\nPlease train the models first:")
        print("  1. python src/train_model1.py")
        print("  2. python src/train_model2.py")
        print("\nThen run this script again.")
        sys.exit(1)
    
    # Run main application
    from src.main import main
    main()

