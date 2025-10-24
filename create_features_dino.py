#!/usr/bin/env python3
# Build prototypes from polygons using DINO embeddings
import os, json, argparse
import random

import numpy as np
import cv2, rasterio
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# ===== Defaults =====
SAMPLES_DIR_DEFAULT = "dataset/sample_set/allfiles"
OUT_DIR_DEFAULT = "outputs_prototypes_dino"
MIN_POLY_PIXELS = 500
DEBUG_PATCH_LIMIT = 3


# ===== Helpers =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
# model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant", use_fast=True)
model = AutoModel.from_pretrained("facebook/dinov2-giant").to(device)
# processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
# model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
# processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
# model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
model.eval()

def extract_dino_embedding(patch_rgb: np.ndarray):
    img = Image.fromarray(patch_rgb)
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        feat = outputs.last_hidden_state[:, 0, :]
    vec = feat.squeeze().cpu().numpy()
    return vec / (np.linalg.norm(vec) + 1e-12)

def augment_patch(patch):
    """Return a list of augmented versions of a patch (HxWx3 uint8)."""
    aug_patches = []

    # Always include original
    aug_patches.append(patch)

    # Horizontal flip
    aug_patches.append(cv2.flip(patch, 1))

    # Vertical flip
    aug_patches.append(cv2.flip(patch, 0))

    # Rotations (90, 180, 270)
    for k in [1, 2, 3]:
        aug_patches.append(np.ascontiguousarray(np.rot90(patch, k)))

    # Small random crops / scales
    H, W = patch.shape[:2]
    if H > 32 and W > 32:  # only if reasonably large
        for _ in range(2):
            scale = random.uniform(0.8, 1.0)
            h_new, w_new = int(H*scale), int(W*scale)
            y0 = random.randint(0, H-h_new)
            x0 = random.randint(0, W-w_new)
            crop = patch[y0:y0+h_new, x0:x0+w_new]
            crop_resized = cv2.resize(crop, (W, H))
            aug_patches.append(crop_resized)

    return aug_patches

def normalize_to_uint8(b):
    b = b.astype(np.float32)
    b = (b - b.min()) / (b.max() - b.min() + 1e-6)
    return (b * 255).astype(np.uint8)


def polygon_to_mask(polygon, W, H):
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def read_rgb_from_tif(path, rgb_band_indexes, nir_index=None):
    with rasterio.open(path) as src:
        count = src.count
        for b in rgb_band_indexes:
            if not (1 <= b <= count):
                raise ValueError(f"[{path}] band {b} outside 1..{count}")
        bands = [normalize_to_uint8(src.read(b)) for b in rgb_band_indexes]
        img = np.stack(bands, axis=-1)  # HxWx3 RGB
        if nir_index is not None and 1 <= nir_index <= count:
            nir = normalize_to_uint8(src.read(nir_index))
            print(f"ℹ️ NIR (band {nir_index}) loaded {nir.shape} — not used by DINO.")
        return img


# ===== Main =====
def main():
    ap = argparse.ArgumentParser("Build DINO prototypes from polygons")
    ap.add_argument("--samples", type=str, default=SAMPLES_DIR_DEFAULT,
                    help="Folder with paired .tif and .json files")
    ap.add_argument("--out", type=str, default=OUT_DIR_DEFAULT,
                    help="Output folder for prototypes + debug")
    ap.add_argument("--band-indexes", type=str, default="1,2,3",
                    help="1-based bands mapped to R,G,B (e.g., '3,2,1')")
    ap.add_argument("--nir-index", type=int, default=None,
                    help="Optional NIR band (not used by DINO)")
    ap.add_argument("--min-poly-pixels", type=int, default=MIN_POLY_PIXELS)
    ap.add_argument("--debug-limit", type=int, default=DEBUG_PATCH_LIMIT)
    ap.add_argument("--debug-save-all", action="store_true")
    ap.add_argument("--single-tif", type=str, default=None,
                    help="Process a single .tif file instead of folder")
    ap.add_argument("--single-json", type=str, default=None,
                    help="Use this .json annotation with --single-tif")
    args = ap.parse_args()

    # Parse bands
    try:
        parts = [int(x.strip()) for x in args.band_indexes.split(",")]
        if len(parts) != 3: raise ValueError
        RGB_BANDS = (parts[0], parts[1], parts[2])
    except Exception:
        raise SystemExit("❌ --band-indexes must be three comma-separated ints, e.g. '1,2,3'")

    print(f"ℹ️ Using bands R,G,B ← {RGB_BANDS}" + (f", NIR={args.nir_index}" if args.nir_index else ""))

    # Output dirs
    os.makedirs(args.out, exist_ok=True)
    dbg_feat_dir = os.path.join(args.out, "debug_features_dino")
    dbg_patch_dir = os.path.join(args.out, "debug_patches_dino")
    dbg_overlay = os.path.join(args.out, "debug_overlays_dino")
    for d in [dbg_feat_dir, dbg_patch_dir, dbg_overlay]:
        os.makedirs(d, exist_ok=True)

    # Files to process
    if args.single_tif and args.single_json:
        files = [os.path.basename(args.single_json)]
        samples_dir = os.path.dirname(args.single_json) or "."
    else:
        samples_dir = args.samples
        files = [f for f in os.listdir(samples_dir) if f.endswith(".json")]

    class_feats, class_names = {}, {}
    print(f"Processing {len(files)} annotation files...")

    for jf in tqdm(files, desc="Prototyping"):
        jpath = os.path.join(samples_dir, jf)
        if not os.path.exists(jpath):
            continue
        with open(jpath, "r") as f:
            anno = json.load(f)

        tif = jf.replace(".json", ".tif")
        ipath = os.path.join(samples_dir, tif)
        if not os.path.exists(ipath):
            continue

        img = read_rgb_from_tif(ipath, RGB_BANDS, args.nir_index)
        H, W = img.shape[:2]

        for feat_idx, feat in enumerate(anno.get("features", [])):
            cls = feat.get("properties", {}).get("Class Name", "")
            geom = feat.get("geometry", {})
            if geom.get("type") != "MultiPolygon":
                continue

            for k, poly in enumerate(geom.get("coordinates", [])):
                ring = poly[0]
                mask = polygon_to_mask(ring, W, H)
                npx = int(np.count_nonzero(mask))

                if npx < args.min_poly_pixels:
                    x, y, w, h = cv2.boundingRect(np.array(ring, np.int32))
                    mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

                # Crop patch
                x, y, w, h = cv2.boundingRect(np.array(ring, np.int32))
                patch = img[y:y+h, x:x+w]

                augmented = augment_patch(patch)
                for aug in augmented:
                    vec = extract_dino_embedding(aug)
                    class_feats.setdefault(cls, []).append(vec)

                save_debug = args.debug_save_all or (k < args.debug_limit)
                if save_debug:
                    patch_masked = cv2.bitwise_and(img, img, mask=mask)
                    fname = f"{cls}_{os.path.basename(jpath)[:-5]}_f{feat_idx+1}_p{k+1}"
                    cv2.imwrite(os.path.join(dbg_patch_dir, f"{fname}.png"), patch_masked)
                    csv_path = os.path.join(dbg_feat_dir, f"{fname}.csv")
                    np.savetxt(csv_path, vec.reshape(1, -1), delimiter=",", fmt="%.6f")
                    class_names.setdefault(cls, []).append(os.path.basename(csv_path))
                    overlay = img.copy()
                    cv2.polylines(overlay, [np.array(ring, np.int32)], True, (0,255,0), 2)
                    cv2.imwrite(os.path.join(dbg_overlay, f"{fname}.png"), overlay)

    proto_index = {"__meta__": {"norm": "l2", "band_indexes": list(RGB_BANDS)}}
    for cls, vecs in class_feats.items():
        proto_index[cls] = {"prototypes": [v.tolist() for v in vecs],
                            "names": class_names.get(cls, [])}

    out_json = os.path.join(args.out, "dino_prototypes.json")
    with open(out_json, "w") as f:
        json.dump(proto_index, f, indent=2)
    print(f"\n✅ Saved prototypes → {out_json}")
    print("✅ Done.")


if __name__ == "__main__":
    main()
