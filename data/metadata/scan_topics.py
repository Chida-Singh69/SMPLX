import json

with open(r'data\metadata\how2sign_mapping.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pkls = [
    "0CAt2QlIcco_0-8-rgb_front.pkl",
    "0CAt2QlIcco_1-8-rgb_front.pkl",
    "0CAt2QlIcco_3-8-rgb_front.pkl",
    "0CAt2QlIcco_4-8-rgb_front.pkl",
    "0CAt2QlIcco_5-8-rgb_front.pkl",
    "0CAt2QlIcco_6-8-rgb_front.pkl",
    "0CAt2QlIcco_8-8-rgb_front.pkl",
    "0CAt2QlIcco_9-8-rgb_front.pkl",
    "1EYhtLj97b8_1-5-rgb_front.pkl",
    "1EYhtLj97b8_2-5-rgb_front.pkl",
    "1EYhtLj97b8_3-5-rgb_front.pkl",
    "1EYhtLj97b8_4-5-rgb_front.pkl",
    "1EYhtLj97b8_5-5-rgb_front.pkl",
    "dcN7W0lxTpE_0-8-rgb_front.pkl",
    "dcN7W0lxTpE_3-8-rgb_front.pkl",
    "dcN7W0lxTpE_4-8-rgb_front.pkl",
    "dcN7W0lxTpE_5-8-rgb_front.pkl",
    "dcN7W0lxTpE_6-8-rgb_front.pkl",
    "dcN7W0lxTpE_7-8-rgb_front.pkl",
    "dcN7W0lxTpE_8-8-rgb_front.pkl",
    "dcN7W0lxTpE_9-8-rgb_front.pkl",
    "dcN7W0lxTpE_15-8-rgb_front.pkl",
    "dcN7W0lxTpE_37-8-rgb_front.pkl",
    "dcN7W0lxTpE_39-8-rgb_front.pkl",
    "dcN7W0lxTpE_40-8-rgb_front.pkl",
    "dcN7W0lxTpE_48-8-rgb_front.pkl",
    "dcN7W0lxTpE_13-8-rgb_front.pkl",
]

print(f"Checking {len(pkls)} PKL filenames against dataset...")
all_ok = True
for i, pkl in enumerate(pkls, 1):
    if pkl in data:
        print(f"  [{i:2d}] OK: {pkl}")
    else:
        print(f"  [{i:2d}] MISSING: {pkl}")
        all_ok = False

print(f"\n{'All 27 sentences verified!' if all_ok else 'SOME SENTENCES MISSING!'}")
