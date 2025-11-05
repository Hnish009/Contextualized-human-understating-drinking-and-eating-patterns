"""
Fix the drinks.csv file format - convert from {"male,15,angry, Mint Lemon Cooler"} 
to proper CSV: gender,age,mood,drink
"""

import re
from pathlib import Path

MOOD_MAP = {
    "happiness": "happy", "happy": "happy", "Happy": "happy",
    "sadness": "sad", "sad": "sad", "Sad": "sad",
    "anger": "angry", "angry": "angry", "Angry": "angry",
    "surprise": "surprise", "Surprise": "surprise",
    "neutral": "neutral", "Neutral": "neutral",
    "tired": "neutral", "Tired": "neutral"
}

VALID_DRINKS = {
    "Classic Lemonade", "Mint Lemon Cooler", "Ginger Lemon Sparkler",
    "Salty Lemon Fizz", "Minty Ginger Punch", "Sweet Lemon Delight",
    "Tangy Masala Lemon", "Mint Salt Cooler", "Ginger Mint Twist",
    "Cool Fusion Lemon Mix"
}

DRINK_FIXES = {
    "Lemon Cooler": "Mint Lemon Cooler",
    "lemon cooler": "Mint Lemon Cooler"
}

def normalize_drink(drink):
    """Normalize drink name - remove spaces, fix case, handle invalid names"""
    drink = drink.strip()
    
    # Try exact match first
    if drink in VALID_DRINKS:
        return drink
    
    # Try case-insensitive match
    for vd in VALID_DRINKS:
        if vd.lower() == drink.lower():
            return vd
    
    # Try fixes
    if drink in DRINK_FIXES:
        return DRINK_FIXES[drink]
    
    # Try case-insensitive fixes
    for key, value in DRINK_FIXES.items():
        if key.lower() == drink.lower():
            return value
    
    # Default to Mint Lemon Cooler if not found
    print(f"Warning: Unknown drink '{drink}', using 'Mint Lemon Cooler'")
    return "Mint Lemon Cooler"

def normalize_mood(mood):
    """Normalize mood to lowercase standard format"""
    mood = mood.strip()
    return MOOD_MAP.get(mood, "neutral")

def normalize_gender(gender):
    """Normalize gender to lowercase"""
    return gender.strip().lower()

def fix_csv():
    """Read the malformed CSV and create proper CSV"""
    csv_path = Path('data/drinks.csv')
    backup_path = Path('data/drinks.csv.backup')
    
    if not csv_path.exists():
        print("ERROR: drinks.csv not found!")
        return
    
    # Read the file
    text = csv_path.read_text(encoding='utf-8')
    lines = text.split('\n')
    
    entries = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Extract from {"male,15,angry, Mint Lemon Cooler"} format
        match = re.match(r'\{"?([^,]+),(\d+),([^,]+),\s*([^}"]+)"?\}', line)
        if match:
            gender_raw = match.group(1).strip()
            age_str = match.group(2).strip()
            mood_raw = match.group(3).strip()
            drink_raw = match.group(4).strip()
            
            gender = normalize_gender(gender_raw)
            try:
                age = int(age_str)
            except:
                continue
            
            mood = normalize_mood(mood_raw)
            drink = normalize_drink(drink_raw)
            
            if gender in ["male", "female"] and age > 0:
                entries.append((gender, age, mood, drink))
    
    if not entries:
        print("ERROR: No valid entries found!")
        return
    
    # Create backup
    if backup_path.exists():
        backup_path.unlink()
    csv_path.rename(backup_path)
    print(f"Created backup: {backup_path}")
    
    # Write proper CSV
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("gender,age,mood,drink\n")
        for g, a, m, d in entries:
            f.write(f"{g},{a},{m},{d}\n")
    
    print(f"OK: Fixed CSV with {len(entries)} entries")
    print(f"Sample entries:")
    for i in range(min(5, len(entries))):
        print(f"  {entries[i]}")

if __name__ == '__main__':
    fix_csv()

