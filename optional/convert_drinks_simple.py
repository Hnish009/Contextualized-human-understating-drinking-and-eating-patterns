"""
Simple direct converter - handles the exact format in drinks_raw.txt
"""

import re
from pathlib import Path

MOOD_MAP = {
    "happiness": "happy", "happy": "happy",
    "sadness": "sad", "sad": "sad",
    "anger": "angry", "angry": "angry",
    "surprise": "surprise",
    "neutral": "neutral",
    "tired": "neutral"
}

VALID_DRINKS = {
    "Classic Lemonade", "Mint Lemon Cooler", "Ginger Lemon Sparkler",
    "Salty Lemon Fizz", "Minty Ginger Punch", "Sweet Lemon Delight",
    "Tangy Masala Lemon", "Mint Salt Cooler", "Ginger Mint Twist",
    "Cool Fusion Lemon Mix"
}

def convert():
    raw = Path('data/drinks_raw.txt')
    out = Path('data/drinks.csv')
    
    text = raw.read_text(encoding='utf-8')
    entries = []
    
    # Pattern: {"gender: X, age: Y, mood: Z, drink: W"}
    pattern = r'\{[^}]*\}'
    matches = re.findall(pattern, text)
    
    print(f"Found {len(matches)} matches...")
    
    for match in matches:
        # Extract gender
        gender_match = re.search(r'gender:\s*(\w+)', match, re.I)
        # Extract age
        age_match = re.search(r'age:\s*(\d+)', match, re.I)
        # Extract mood
        mood_match = re.search(r'mood:\s*(\w+)', match, re.I)
        # Extract drink (everything after "drink:")
        drink_match = re.search(r'drink:\s*([^}"]+)', match, re.I)
        
        if all([gender_match, age_match, mood_match, drink_match]):
            gender = gender_match.group(1).lower()
            age = int(age_match.group(1))
            mood_raw = mood_match.group(1).lower()
            drink = drink_match.group(1).strip().strip('"')
            
            mood = MOOD_MAP.get(mood_raw, "neutral")
            
            # Validate drink (case-insensitive)
            drink_valid = None
            for vd in VALID_DRINKS:
                if vd.lower() == drink.lower():
                    drink_valid = vd
                    break
            
            # Fix known issues
            if drink.lower() == "juice lemon cooler":
                drink_valid = "Mint Lemon Cooler"
            
            if gender in ["male", "female"] and drink_valid:
                entries.append((gender, age, mood, drink_valid))
    
    if not entries:
        print("ERROR: No valid entries!")
        return
    
    # Write CSV
    with open(out, 'w', encoding='utf-8') as f:
        f.write("gender,age,mood,drink\n")
        for g, a, m, d in entries:
            f.write(f"{g},{a},{m},{d}\n")
    
    print(f"OK: Converted {len(entries)} entries to {out}")
    print(f"Sample: {entries[0]}")

if __name__ == '__main__':
    convert()

