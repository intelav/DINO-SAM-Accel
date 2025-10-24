#!/usr/bin/env python3
"""
Create augmented multi-view DINO prototypes with PCA whitening.
- Fully CLI-compatible with the old create_features_dino.py
- Generates PCA-whitened prototypes per class
- Saves debug patches and overlays
- Outputs both new and legacy JSON formats for compatibility
"""

import os, json, cv2, torch, argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from train_global_pca import *

# ======= DEFAULTS =======
SAMPLES_DIR_DEFAULT = "dataset/sample_set/allfiles"
OUT_DIR_DEFAULT = "runs/default_run/prototypes"
MIN_POLY_PIXELS = 500
DEBUG_PATCH_LIMIT = 3
PATCH_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/dinov2-base"  # next iteration: dinov2-large / clip-vit-l
N_AUGS = 8
PCA_DIM = 256
SAVE_RAW_FEATURES_ONLY = False


# ======= MODEL LOADER =======
@torch.no_grad()
def load_dino_model():
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return proc, model


# ======= AUGMENTATION PIPELINE =======
AUG = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(PATCH_SIZE, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
])


def apply_clahe(img):
    # Ensure 8-bit RGB
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


@torch.no_grad()
def get_embedding(proc, model, img_rgb):
    pil = Image.fromarray(img_rgb)
    inputs = proc(images=pil, return_tensors="pt").to(DEVICE)
    feats = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()[0]
    feats = feats / (np.linalg.norm(feats) + 1e-12)
    return feats.astype(np.float32)


def normalize_to_uint8(b):
    b = b.astype(np.float32)
    b = (b - b.min()) / (b.max() - b.min() + 1e-6)
    return (b * 255).astype(np.uint8)


def read_rgb_from_tif(path, rgb_band_indexes, nir_index=None):
    import rasterio
    with rasterio.open(path) as src:
        count = src.count
        for b in rgb_band_indexes:
            if not (1 <= b <= count):
                raise ValueError(f"[{path}] band {b} outside 1..{count}")
        bands = [normalize_to_uint8(src.read(b)) for b in rgb_band_indexes]
        img = np.stack(bands, axis=-1)
        if nir_index is not None and 1 <= nir_index <= count:
            nir = normalize_to_uint8(src.read(nir_index))
            print(f"ℹ️ NIR (band {nir_index}) loaded {nir.shape} — not used by DINO.")
        return img


# ======= MAIN =======
def main():
    ap = argparse.ArgumentParser("Build DINO prototypes from polygons (PCA + augment)")
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
    os.makedirs(args.out, exist_ok=True)

    dbg_patch_dir = os.path.join(args.out, "debug_patches_dino")
    dbg_overlay_dir = os.path.join(args.out, "debug_overlays_dino")
    dbg_feat_dir = os.path.join(args.out, "debug_features_dino")
    for d in [dbg_patch_dir, dbg_overlay_dir, dbg_feat_dir]:
        os.makedirs(d, exist_ok=True)

    proc, model = load_dino_model()
    all_feats, class_to_feats = [], defaultdict(list)

    # Files
    if args.single_tif and args.single_json:
        files = [os.path.basename(args.single_json)]
        samples_dir = os.path.dirname(args.single_json) or "."
    else:
        samples_dir = args.samples
        files = [f for f in os.listdir(samples_dir) if f.endswith(".json")]

    print(f"Processing {len(files)} annotation files...")

    for jf in tqdm(files, desc="Prototyping"):
        jpath = os.path.join(samples_dir, jf)
        tif = jf.replace(".json", ".tif")
        ipath = os.path.join(samples_dir, tif)
        if not os.path.exists(ipath):
            continue

        with open(jpath, "r") as f:
            ann = json.load(f)

        img = read_rgb_from_tif(ipath, RGB_BANDS, args.nir_index)
        H, W = img.shape[:2]

        for feat_idx, feat in enumerate(ann.get("features", [])):
            cls = feat.get("properties", {}).get("Class Name", "")
            geom = feat.get("geometry", {})
            if geom.get("type") != "MultiPolygon":
                continue

            for k, poly in enumerate(geom.get("coordinates", [])):
                ring = poly[0]
                xs = [p[0] for p in ring];
                ys = [p[1] for p in ring]
                x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                patch = img[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                patch = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                emb_list = []
                for _ in range(N_AUGS):
                    aug_img = AUG(Image.fromarray(apply_clahe(patch)))
                    emb = get_embedding(proc, model, np.array(aug_img))
                    emb_list.append(emb)
                emb_mean = np.mean(emb_list, axis=0)
                class_to_feats[cls].append(emb_mean)
                all_feats.append(emb_mean)

                # ---- Debug saving ----
                save_debug = args.debug_save_all or (len(class_to_feats[cls]) <= args.debug_limit)
                if save_debug:
                    fname = f"{cls}_{os.path.basename(jpath)[:-5]}_f{feat_idx + 1}_p{k + 1}"
                    cv2.imwrite(os.path.join(dbg_patch_dir, f"{fname}.png"), patch)
                    overlay = img.copy()
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(dbg_overlay_dir, f"{fname}.png"), overlay)
                    np.savetxt(os.path.join(dbg_feat_dir, f"{fname}.csv"),
                               emb_mean.reshape(1, -1), delimiter=",", fmt="%.6f")

    if not all_feats:
        print("❌ No features extracted — check sample paths or band indexes.")
        return

    if SAVE_RAW_FEATURES_ONLY:
        os.makedirs("runs/pca_stage/features_raw", exist_ok=True)
        base_name = Path(args.single_tif).stem or "unknown"
        print(f"[FeatureDump] Processing {args.single_tif or args.samples} → {base_name}")
        out_npy = f"runs/pca_stage/features_raw/{base_name}.npy"
        np.save(out_npy, np.vstack(all_feats))
        print(f"✅ Saved raw DINO features → {out_npy} ({np.vstack(all_feats).shape})")
        #print(f"✅ Saved raw features → runs/pca_stage/features_raw/{os.path.basename(args.out)}.npy")
        return  # skip PCA for now
    else:
        train_global_pca()
    # --- GLOBAL PCA MODE ---
    from train_global_pca import load_global_pca

    print("🧮 Applying GLOBAL PCA whitening ...")
    pca_info = load_global_pca()
    print(f"📂 Global PCA loaded: dim={len(pca_info['components'])}, "
          f"explained_var={sum(pca_info['explained_var']) * 100:.2f}%")
    # Convert PCA info to torch tensors
    mean_t = torch.tensor(pca_info["scaler_mean"], dtype=torch.float32, device="cuda")
    scale_t = torch.tensor(pca_info["scaler_scale"], dtype=torch.float32, device="cuda")
    components_t = torch.tensor(pca_info["components"], dtype=torch.float32, device="cuda")

    def project_with_global_pca(feat_t):
        """Project feature using fixed global PCA (101D)."""
        v_std = (feat_t - mean_t) / scale_t
        return v_std @ components_t.T

    all_feats_pca = [(project_with_global_pca(torch.tensor(f, device="cuda"))).cpu().numpy()
                     for f in all_feats]
    all_feats_pca = np.stack(all_feats_pca)

    # Reassign transformed features classwise
    offset = 0
    class_prototypes = {}
    legacy_index = {"__meta__": {"norm": "pca", "band_indexes": list(RGB_BANDS)}}

    # --- Preserve all PCA-projected embeddings (no averaging) ---
    for cls, feats in class_to_feats.items():
        n = len(feats)
        f_cls = all_feats_pca[offset:offset + n]
        offset += n

        # Keep all polygon-level vectors
        class_prototypes[cls] = {
            "prototypes": f_cls.tolist(),
            "count": n
        }

        # legacy / compact version (still needed by older scripts)
        legacy_index[cls] = {
            "prototypes": f_cls.tolist(),
            "names": [],
            "count": n
        }

    # ===== Save outputs =====
    out_json_new = os.path.join(args.out, "dino_prototypes.json")
    out_json_compact = os.path.join(args.out, "dino_prototypes_pca_compact.json")
    os.makedirs(os.path.dirname(out_json_new), exist_ok=True)

    # --- Save outputs using the already-loaded GLOBAL PCA ---
    out_json_new = os.path.join(args.out, "dino_prototypes.json")
    out_json_compact = os.path.join(args.out, "dino_prototypes_pca_compact.json")
    os.makedirs(os.path.dirname(out_json_new), exist_ok=True)

    out_new = {
        "pca_dim": len(pca_info["components"]),
        "scaler_mean": pca_info["scaler_mean"],
        "scaler_scale": pca_info["scaler_scale"],
        "components": pca_info["components"],
        "explained_var": pca_info["explained_var"],
        "classes": class_prototypes
    }

    with open(out_json_new, "w") as f:
        json.dump(out_new, f, indent=2)
    with open(out_json_compact, "w") as f:
        json.dump(legacy_index, f, indent=2)

    print(f"✅ Saved full PCA prototypes → {out_json_new}")
    print(f"✅ Saved PCA-compact prototypes → {out_json_compact}")


if __name__ == "__main__":
    main()
