"""
Simple converter to convert drinks_raw.txt to drinks.csv
No pandas required - uses basic Python
"""

import json
import re
from pathlib import Path

VALID_MOODS = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}
MOOD_MAP = {
    "happiness": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "anger": "angry",
    "angry": "angry",
    "surprise": "surprise",
    "neutral": "neutral",
    "tired": "neutral"
}

DRINK_FIX_MAP = {
    "juice lemon cooler": "Mint Lemon Cooler"
}


def load_recipes_names():
    with open('data/recipes.json', 'r') as f:
        recipes = json.load(f)
    return set(recipes['drinks'].keys())


def normalize_mood(mood_raw):
    m = MOOD_MAP.get(mood_raw.strip().lower(), None)
    if m is None:
        return "neutral"
    return m


def normalize_drink(name_raw, valid_drinks):
    name = name_raw.strip()
    if name in valid_drinks:
        return name
    fix = DRINK_FIX_MAP.get(name.lower())
    if fix and fix in valid_drinks:
        return fix
    # Try case-insensitive match
    for vd in valid_drinks:
        if vd.lower() == name.lower():
            return vd
    return None


def parse_entry(obj, valid_drinks):
    if isinstance(obj, dict):
        gender = str(obj.get('gender', '')).strip().lower()
        age = int(obj.get('age', 0))
        mood = normalize_mood(str(obj.get('mood', '')))
        drink = normalize_drink(str(obj.get('drink', '')).strip(), valid_drinks)
        return gender, age, mood, drink
    
    # String format: {"gender: male, age: 30, mood: Happiness, drink: ..."}
    s = obj.strip().strip('{}').strip()
    parts = [p for p in re.split(r',\s*', s) if p]
    kv = {}
    for p in parts:
        if ':' in p:
            k, v = p.split(':', 1)
            kv[k.strip().lower()] = v.strip()
    
    gender = kv.get('gender', '').lower()
    age_str = kv.get('age', '0')
    age = int(re.sub(r'[^0-9]', '', age_str) or '0')
    mood = normalize_mood(kv.get('mood', ''))
    drink = normalize_drink(kv.get('drink', ''), valid_drinks)
    return gender, age, mood, drink


def convert_data():
    raw_path = Path('data/drinks_raw.txt')
    out_csv = Path('data/drinks.csv')
    
    if not raw_path.exists():
        print(f"ERROR: File not found: {raw_path}")
        return
    
    valid_drinks = load_recipes_names()
    text = raw_path.read_text(encoding='utf-8').strip()
    entries = []
    
    # Try JSON parsing first
    try:
        data = json.loads(text)
        print(f"Parsed as JSON array with {len(data)} items")
        for item in data:
            g, a, m, d = parse_entry(item, valid_drinks)
            if g in {"male", "female"} and a > 0 and m in VALID_MOODS and d:
                entries.append((g, a, m, d))
    except json.JSONDecodeError as e:
        # Fallback: extract {...} blocks from all lines
        print(f"JSON parsing failed, extracting entries from text...")
        # Find all entries that look like {"gender: ..., age: ..., mood: ..., drink: ..."}
        pattern = r'\{[^}]*gender[^}]*age[^}]*mood[^}]*drink[^}]*\}'
        matches = re.findall(pattern, text, re.IGNORECASE)
        print(f"Found {len(matches)} potential entries...")
        
        for match in matches:
            try:
                g, a, m, d = parse_entry(match, valid_drinks)
                if g in {"male", "female"} and a > 0 and m in VALID_MOODS and d:
                    entries.append((g, a, m, d))
            except Exception:
                continue
        print(f"Successfully parsed {len(entries)} entries")
    
    if not entries:
        print("ERROR: No valid entries found!")
        return
    
    # Write CSV
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write("gender,age,mood,drink\n")
        for g, a, m, d in entries:
            f.write(f"{g},{a},{m},{d}\n")
    
    print(f"OK: Converted {len(entries)} entries to {out_csv}")
    print(f"  Sample: {entries[0]}")


if __name__ == '__main__':
    convert_data()

