"""
Ingest raw drinks dataset text into a clean CSV.
Usage:
  python src/ingest_drinks.py data/drinks_raw.txt
Accepts a JSON-like list of objects or simple lines with key:value pairs.
Normalizes:
- Moods: Happiness->happy, Sadness->sad, Anger->angry, Surprise->surprise, Neutral->neutral, Tired->neutral
- Drink name fix: 'Juice Lemon Cooler' -> 'Mint Lemon Cooler'
Outputs:
- data/drinks.csv with columns: gender,age,mood,drink
"""

import sys
import json
import re
from pathlib import Path
import pandas as pd

VALID_MOODS = {"angry","disgust","fear","happy","sad","surprise","neutral"}
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
	with open('data/recipes.json','r') as f:
		recipes = json.load(f)
	return set(recipes['drinks'].keys())


def normalize_mood(mood_raw: str) -> str:
	m = MOOD_MAP.get(mood_raw.strip().lower(), None)
	if m is None:
		return "neutral"
	return m


def normalize_drink(name_raw: str, valid_drinks: set) -> str:
	name = name_raw.strip()
	if name in valid_drinks:
		return name
	fix = DRINK_FIX_MAP.get(name.lower())
	if fix and fix in valid_drinks:
		return fix
	# Last resort: try case-insensitive match
	for vd in valid_drinks:
		if vd.lower() == name.lower():
			return vd
	return None


def parse_entry(obj, valid_drinks):
	# obj can be dict or string
	if isinstance(obj, dict):
		gender = str(obj.get('gender', '')).strip().lower()
		age = int(obj.get('age'))
		mood = normalize_mood(str(obj.get('mood','')))
		drink = normalize_drink(str(obj.get('drink','')).strip(), valid_drinks)
		return gender, age, mood, drink
	
	# string like: {"gender: male, age: 30, mood: Happiness, drink: ..."}
	s = obj.strip().strip('{}').strip()
	parts = [p for p in re.split(r',\s*', s) if p]
	kv = {}
	for p in parts:
		if ':' in p:
			k, v = p.split(':', 1)
			kv[k.strip().lower()] = v.strip()
	gender = kv.get('gender', '').lower()
	age = int(re.sub(r'[^0-9]','', kv.get('age','0')) or 0)
	mood = normalize_mood(kv.get('mood',''))
	drink = normalize_drink(kv.get('drink',''), valid_drinks)
	return gender, age, mood, drink


def ingest(raw_path: Path, out_csv: Path):
	valid_drinks = load_recipes_names()
	text = raw_path.read_text(encoding='utf-8').strip()
	entries = []
	
	# Try JSON first
	try:
		data = json.loads(text)
		for item in data:
			g, a, m, d = parse_entry(item, valid_drinks)
			if g in {"male","female"} and a>0 and m in VALID_MOODS and d:
				entries.append((g,a,m,d))
	except Exception:
		# Fallback: extract {...} blocks
		blocks = re.findall(r'\{[^}]*\}', text, flags=re.DOTALL)
		for b in blocks:
			g, a, m, d = parse_entry(b, valid_drinks)
			if g in {"male","female"} and a>0 and m in VALID_MOODS and d:
				entries.append((g,a,m,d))
	
	if not entries:
		raise SystemExit("No valid entries parsed. Please check the raw file format.")
	
	df = pd.DataFrame(entries, columns=["gender","age","mood","drink"])
	df.to_csv(out_csv, index=False)
	print(f"✓ Wrote {len(df)} rows to {out_csv}")


def main():
	if len(sys.argv) < 2:
		print("Usage: python src/ingest_drinks.py data/drinks_raw.txt")
		sys.exit(1)
	raw_path = Path(sys.argv[1])
	out_csv = Path('data/drinks.csv')
	if not raw_path.exists():
		raise SystemExit(f"Input not found: {raw_path}")
	ingest(raw_path, out_csv)

if __name__ == '__main__':
	main()
