#!/usr/bin/env python3
"""
Combine all per-run DINO prototype JSON files into one consolidated prototype set.

- Looks for runs/*/prototypes/dino_prototypes.json
- Keeps ALL 768D prototypes for each class (does NOT average them).
- Writes runs/combined_dino_prototypes.json
"""

import json
import glob
import os

# Find all DINO prototype JSON files
files = glob.glob("runs/*/prototypes/dino_prototypes.json")

if not files:
    raise SystemExit("❌ No dino_prototypes.json files found under runs/*/prototypes/")

combined = {"__meta__": None}

for f in files:
    with open(f, "r") as fh:
        data = json.load(fh)

    # Copy metadata from the first file
    if combined["__meta__"] is None:
        combined["__meta__"] = data.get("__meta__", {"source": "dino"})

    for cls, vals in data.items():
        if cls == "__meta__":
            continue
        if cls not in combined:
            combined[cls] = {"prototypes": [], "names": []}
        # Append all prototypes + names
        if isinstance(vals, dict):
            combined[cls]["prototypes"].extend(vals.get("prototypes", []))
            combined[cls]["names"].extend(vals.get("names", []))
        else:
            # Handle legacy format (just a list of vectors)
            combined[cls]["prototypes"].extend(vals)
            combined[cls]["names"].extend([os.path.basename(f)] * len(vals))

# Output path
out_path = "runs/combined_dino_prototypes.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w") as out:
    json.dump(combined, out, indent=2)

print(f"✅ Combined DINO prototypes written to {out_path}")
print("   Classes:", [k for k in combined.keys() if k != "__meta__"])
for cls, vals in combined.items():
    if cls == "__meta__":
        continue
    print(f"   - {cls}: {len(vals['prototypes'])} prototypes")
