#!/usr/bin/env python3
import json, glob, os, numpy as np

SEARCH_PATTERN = "runs/*/prototypes/"
OUT_PATH = "runs/combined_dino_prototypes.json"

files = []
for folder in glob.glob(os.path.join(SEARCH_PATTERN, "")):
    full_pca = os.path.join(folder, "dino_prototypes.json")
    compact = os.path.join(folder, "dino_prototypes_pca_compact.json")
    if os.path.exists(full_pca):
        files.append(full_pca)
    elif os.path.exists(compact):
        files.append(compact)

if not files:
    print("❌ No prototype JSONs found under runs/*/prototypes/")
    exit(1)

print(f"🔍 Found {len(files)} prototype files.")
combined = {
    "__meta__": {"note": "Combined PCA-whitened prototypes"},
    "pca_dim": None,
    "scaler_mean": None,
    "scaler_scale": None,
    "components": None,
    "explained_var": None,
    "classes": {}
}

for f in files:
    with open(f, "r") as fh:
        data = json.load(fh)

    # --- Full PCA JSON ---
    if "classes" in data and "pca_dim" in data:
        if combined["pca_dim"] is None:
            for key in ["pca_dim", "mean", "scale", "components", "explained_var"]:
                combined[key] = data.get(key)
            print(f"📦 Loaded PCA metadata from first valid file: {os.path.basename(f)}")

        for cls, vals in data["classes"].items():
            protos = vals.get("prototypes")
            if not protos:
                continue
            combined["classes"].setdefault(cls, {"prototypes": [], "count": 0})
            combined["classes"][cls]["prototypes"].extend(protos)
            combined["classes"][cls]["count"] += vals.get("count", len(protos))

    # --- Compact JSON ---
    elif "__meta__" in data and any("prototypes" in v for v in data.values() if isinstance(v, dict)):
        for cls, vals in data.items():
            if cls == "__meta__":
                continue
            combined["classes"].setdefault(cls, {"prototypes": [], "names": [], "count": 0})
            combined["classes"][cls]["prototypes"].extend(vals.get("prototypes", []))
            combined["classes"][cls]["count"] += vals.get("count", 1)

if not combined["classes"]:
    print("⚠️ No valid prototypes found in any file.")
else:
    print(f"✅ Merged {len(combined['classes'])} classes from {len(files)} files.")
    print(f"🔁 Traversed {len(files)} prototype files total.")
    print(f"📂 PCA metadata preserved from first valid file.")
    print("📊 Prototype counts per class:")
    for cls, vals in combined["classes"].items():
        num_vecs = len(vals.get("prototypes", []))
        print(f"  - {cls:<15} : {num_vecs:>3} vectors")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as out:
    json.dump(combined, out, indent=2)

# --- Dimension summary ---
dims = []
for cls, val in combined["classes"].items():
    for proto in val["prototypes"]:
        try:
            dims.append(len(proto))
        except Exception:
            continue

print(f"✅ Combined {len(files)} files with {len(combined['classes'])} classes.")
if dims:
    print(f"ℹ Average prototype vector dim: {np.mean(dims):.1f}")
else:
    print("⚠️ No valid prototype vectors found.")
print(f"✅ Full PCA metadata preserved → {OUT_PATH}")
