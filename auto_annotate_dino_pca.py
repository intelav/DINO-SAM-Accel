#!/usr/bin/env python3
# Auto-annotate with DINO embeddings + optional SAM2 masked refinement
import ast
import os, json, argparse, glob

import joblib
import numpy as np
import cv2, rasterio
import pandas as pd
import torch
from torch import cosine_similarity
from torch.cuda import nvtx
from torchvision.ops import nms
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from batch_stats import *
from ui_review import ReviewDeps, SamConfig, review_and_save
from train_global_pca import load_global_pca

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
# processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
# model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
model.eval()

MAX_TOTAL = 400
MAX_PER_CLASS = 50

#PATCH_SIZES = [32, 40, 48, 64]
CLASS_PATCH_SIZES = {
    "STP": [32, 40],
    "Sheds": [32, 40],
    "Metro Shed": [32, 40, 48],
    "Solar Panel": [48, 64],
    "Play Ground": [48, 64],
    "Pond-1": [32, 40],
    "Pond-2": [32, 40],
    "Brick Kiln": [48, 64],
}

CLASS_PATCH_SIZES_TUNED = {
    "STP": [16, 24],  # very small (10–23 px) → fine patches
    "Sheds": [32, 40],  # medium (similar to Pond-1/2 range)
    "Metro Shed": [48, 64],  # larger (median ~58 px, up to 66–107 px height)
    "Solar Panel": [48, 64],  # large, consistent with playground
    "Play Ground": [48, 64],  # large (~53 px, range 41–68 px)
    "Pond-1": [24, 32, 40],  # small-medium (~30 px, some as low as 23 px)
    "Pond-2": [24, 32, 40],  # variable (~30 px median, can go 16–60 px)
    "Brick Kiln": [32, 40, 48],  # full observed range 26–48 px
}

STRIDE_FRAC = 1.0
SIM_THRESH = 0.90
# CLASS_SIM_THRESH = {
#     "Brick Kiln": 0.90,
#     "Pond-1": 0.92,
#     "Pond-2": 0.92,
#     "Sheds": 0.93,
#     "Metro Shed": 0.93,
#     "Solar Panel": 0.95,
#     "Play Ground": 0.94,
#     "STP": 0.93,
# }


CLASS_SIM_THRESH = {cls: 0.50 for cls in
                    ["Brick Kiln", "Metro Shed", "Play Ground", "Pond-1", "Pond-2", "STP", "Sheds", "Solar Panel"]}


# ===== DINO Embedding =====
def extract_dino_embeddings_batch(patches, batch_size=64):
    """
    patches: list of HxWx3 uint8 numpy arrays
    Returns: list of normalized embeddings (numpy arrays)
    """
    all_feats = []
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i + batch_size]
        pil_images = [Image.fromarray(p) for p in batch]
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        with torch.no_grad():
            nvtx.range_push("dino_forward")
            outputs = model(**inputs)
            nvtx.range_pop()
            feats = outputs.last_hidden_state[:, 0, :]  # (B, D) on GPU
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)  # L2 normalize on GPU
        #feats = feats.clone()
        all_feats.append(feats)

        # # 👇 Add this to monitor GPU memory usage after each batch
        # allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        # reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        # print(f"[MEM][Batch {i // batch_size}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    return torch.cat(all_feats, dim=0)


def save_yolo_func(save_path, boxes, W, H, classes):
    """
    Write accepted boxes to YOLO-format .txt
    boxes: list of (cls, x, y, w, h) in pixel coords
    """
    out_txt = save_path if save_path.endswith(".txt") else save_path + ".txt"
    with open(out_txt, "w") as f:
        for cls, x, y, w, h in boxes:
            if cls not in classes:
                continue
            cid = classes.index(cls)
            xc = (x + w / 2.0) / W
            yc = (y + h / 2.0) / H
            bw = w / W
            bh = h / H
            f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"💾 YOLO labels saved → {out_txt}")


def extract_features_masked_dino(roi_rgb, mask_u8):
    roi_masked = roi_rgb.copy()
    roi_masked[mask_u8 == 0] = 0
    return extract_dino_embeddings_batch([roi_masked], batch_size=1)[0]


def _iou(box1, box2):
    # box: [x1,y1,x2,y2]
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def relax_thresholds(thr_dict, factor=0.02):
    return {k: max(0.5, v - factor) for k, v in thr_dict.items()}


def enforce_balance(candidates, thresholds, max_per_class=20):
    """
    Balance candidates so that no class exceeds max_per_class.
    Keeps top-N by similarity per class.
    """
    from collections import defaultdict
    balanced = []
    grouped = defaultdict(list)

    # Group candidates by class
    for c in candidates:
        grouped[c["class"]].append(c)

    # Process each class
    for cls, cls_candidates in grouped.items():
        # Sort by similarity (highest first)
        cls_candidates = sorted(cls_candidates, key=lambda c: c["sim"], reverse=True)

        total_found = len(cls_candidates)
        kept = cls_candidates[:max_per_class]
        balanced.extend(kept)

        if kept:
            sims = [c["sim"] for c in kept]
            print(f"[DEBUG][Balance] {cls}: found {total_found}, kept {len(kept)}, "
                  f"sim range [{min(sims):.3f}, {max(sims):.3f}]")
        else:
            print(f"[DEBUG][Balance] {cls}: found {total_found}, kept 0")

    print(f"[DEBUG][Balance] Final balanced candidates: {len(balanced)}")
    return balanced


def extract_dino_features(img, band_indexes, class_patch_sizes, stride_frac=1.0, batch_size=64):
    """
    Extract DINO embeddings once for all patch sizes used by all classes.
    Returns:
        windows: list of (x, y, w, h)
        feats: list of torch tensors on GPU
    """
    windows = []
    patches = []
    H, W, _ = img.shape

    unique_sizes = sorted({ps for sizes in class_patch_sizes.values() for ps in sizes})
    size_counts = {}
    for ps in unique_sizes:
        stride = int(ps * stride_frac)
        count_for_ps = 0
        for y in range(0, H - ps + 1, stride):
            for x in range(0, W - ps + 1, stride):
                crop = img[y:y + ps, x:x + ps, :]
                patches.append(crop)
                windows.append((x, y, ps, ps))
                count_for_ps += 1
        size_counts[ps] = count_for_ps

    feats_batch = extract_dino_embeddings_batch(patches, batch_size=batch_size)  # (N, D) tensor on GPU
    feats = [feats_batch[i] for i in range(feats_batch.shape[0])]  # keep as torch tensors on GPU

    print("[DEBUG] Window counts per patch size:")
    for ps, cnt in size_counts.items():
        print(f"   {ps}x{ps}: {cnt}")
    print(f"[DEBUG] Total windows across all sizes: {len(windows)}")

    return windows, feats


def mask_to_box(mask, offset=None):
    """
    Convert a binary mask (H,W) or (1,H,W) to a bounding box [x, y, w, h].

    If 'offset'=(x0,y0) is given, it will shift the box into global coordinates.
    """
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]  # squeeze channel
    elif mask.ndim > 2:
        mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"mask_to_box expected 2D mask, got shape {mask.shape}")

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # Local box (relative to cropped mask)
    box = (int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1))

    # Shift into global coordinates if offset is provided
    if offset is not None:
        ox, oy = offset
        box = (box[0] + ox, box[1] + oy, box[2], box[3])

    return box


# Stage A: Global embeddings

def extract_dino_features_all(img, stride_frac=0.5, batch_size=64):
    import time
    t0 = time.time()

    H, W, _ = img.shape
    windows, patches = [], []
    size_counts = {}
    size_to_indices = {}

    unique_sizes = sorted({ps for sizes in CLASS_PATCH_SIZES_TUNED.values() for ps in sizes})

    # ---------------- Collect patches ----------------
    for ps in unique_sizes:
        stride = int(ps * stride_frac)
        indices = []
        count_for_ps = 0
        for y in range(0, H - ps + 1, stride):
            for x in range(0, W - ps + 1, stride):
                crop = img[y:y + ps, x:x + ps, :]
                patches.append(crop)
                windows.append((x, y, ps, ps))
                indices.append(len(windows) - 1)
                count_for_ps += 1
        size_counts[ps] = count_for_ps
        size_to_indices[ps] = indices

    t1 = time.time()
    print(f"[Stage A] Patch extraction finished in {t1 - t0:.2f} sec "
          f"(collected {len(patches)} crops across {len(unique_sizes)} sizes)")

    # ---------------- Run DINO embeddings ----------------
    feats_batch = extract_dino_embeddings_batch(patches, batch_size=batch_size)
    feats = [feats_batch[i] for i in range(feats_batch.shape[0])]

    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"[MEM][Post Stage A concat] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

    t2 = time.time()
    print(f"[Stage A] DINO embedding finished in {t2 - t1:.2f} sec "
          f"(batch_size={batch_size})")
    print(f"[Stage A] Total Stage A time = {t2 - t0:.2f} sec")

    print("[STAGE A] Window counts per patch size:")
    for ps, cnt in size_counts.items():
        print(f"   {ps}x{ps}: {cnt}")
    print(f"[STAGE A] Total windows across all sizes: {len(windows)}")

    return windows, feats, size_to_indices


def get_class_windows_feats(cls_name, windows, feats, size_to_indices):
    patch_sizes = CLASS_PATCH_SIZES_TUNED.get(cls_name, [32, 40, 48])
    indices = []
    for ps in patch_sizes:
        indices.extend(size_to_indices[ps])
    return [windows[i] for i in indices], [feats[i] for i in indices]


import time
from collections import Counter


def detect_candidates_precomputed(
        all_windows,
        all_feats,
        size_to_indices,
        classes,
        prototypes,
        thr_dict,
        enforce_balance=False,
        sam_enabled=False,
        masks_func=None,
        predictor_func=None,
        img=None,
        pcaproj=None

):
    """
        Detect candidates given precomputed DINO embeddings.
        Stages: B (per-class filtering), C (SAM refinement), D (ROI re-embedding).
        """

    # ---------------- Stage B: Per-Class Filtering ----------------
    import time
    from collections import Counter

    t0 = time.time()

    # ---------------- Build similarity cache ----------------
    sim_cache = []  # [{cls: sim, ...}, ...]
    i = 0
    for feat_t in all_feats:
        win_sims = {}
        for cls in classes:
            # Prototype mode
            proto_list = prototypes.get(cls, [])
            if proto_list:
                P = torch.stack(proto_list)

                feat_proj = pcaproj(feat_t)

                if i < 10:
                    print("[DEBUG] feat_proj mean/std:", feat_proj.mean().item(), feat_proj.std().item())
                    print("[DEBUG] proto mean/std:", P.mean().item(), P.std().item())
                i = i + 1

                sims = torch.matmul(P, feat_proj) / (
                        torch.norm(P, dim=1) * torch.norm(feat_proj) + 1e-12
                )
                win_sims[cls] = float(sims.max())

        sim_cache.append(win_sims)
    # === Dynamic per-class threshold tuning ===
    # Collect per-class max similarity across all windows (like you just did)
    per_class_max = {cls: 0.0 for cls in classes}
    for win_sims in sim_cache:
        for cls, sim in win_sims.items():
            if sim > per_class_max[cls]:
                per_class_max[cls] = sim

    # Compute adaptive thresholds
    CLASS_SIM_THRESH_LOCAL = {}
    for cls, max_sim in per_class_max.items():
        CLASS_SIM_THRESH_LOCAL[cls] = max(0.5, max_sim * 0.85)  # tweak 0.85 if needed
        print(f"[ADAPTIVE] {cls}: max_sim={max_sim:.3f} → threshold={CLASS_SIM_THRESH_LOCAL[cls]:.3f}")
    # all_max_sims = {cls: [] for cls in classes}
    # for win in sim_cache:
    #     for cls, sim in win.items():
    #         all_max_sims[cls].append(sim)
    # for cls in classes:
    #     if all_max_sims[cls]:
    #         print(f"{cls}: max={np.max(all_max_sims[cls]):.3f}, mean={np.mean(all_max_sims[cls]):.3f}")

    print(f"[DEBUG] Total windows processed: {len(all_feats)}")
    print(f"[DEBUG] Total win_sims cached: {len(sim_cache)} (each up to {len(classes)} entries)")
    t1 = time.time()
    print(f"[Stage B] Similarity cache built in {t1 - t0:.2f} sec")

    # ---------------- Adaptive threshold loop ----------------
    candidates = []
    current_thr = thr_dict.copy()
    attempt = 0
    MAX_ATTEMPTS = 5
    current_thr = CLASS_SIM_THRESH_LOCAL.copy()
    while min(current_thr.values()) >= 0.20 and len(candidates) < MAX_TOTAL and attempt < MAX_ATTEMPTS:
        new_candidates = []

        for cls in classes:
            patch_sizes = CLASS_PATCH_SIZES_TUNED.get(cls, [32, 40, 48])
            allowed_indices = []
            for ps in patch_sizes:
                allowed_indices.extend(size_to_indices.get(ps, []))

            for i in allowed_indices:
                (x, y, w, h) = all_windows[i]
                feat_t = all_feats[i]
                win_sims = sim_cache[i]
                sim = win_sims.get(cls, -1.0)
                if sim >= current_thr.get(cls, 0.9):
                    new_candidates.append({
                        "x": x, "y": y, "w": w, "h": h,
                        "class": cls,
                        "sim": sim,
                        "feat": feat_t
                    })  # assign to first qualifying class

        print(f"[Stage B] Attempt {attempt + 1}: accepted {len(new_candidates)} candidates")

        # Per-class balancing
        if enforce_balance and new_candidates:
            balanced = []
            for cls in classes:
                cls_cands = [c for c in new_candidates if c["class"] == cls]
                cls_cands = sorted(cls_cands, key=lambda c: c["sim"], reverse=True)[:MAX_PER_CLASS]
                balanced.extend(cls_cands)
            new_candidates = balanced
            print(f"[Stage B] After balancing → {len(new_candidates)} candidates")

            # Enforce global cap
            if len(new_candidates) > MAX_TOTAL:
                new_candidates = sorted(new_candidates, key=lambda c: c["sim"], reverse=True)[:MAX_TOTAL]
                print(f"[Stage B] Truncated to MAX_TOTAL={MAX_TOTAL} → {len(new_candidates)}")

        candidates = new_candidates
        cls_counts = Counter([c["class"] for c in candidates])
        for cls, count in cls_counts.items():
            if count > MAX_PER_CLASS:
                current_thr[cls] = min(0.99, current_thr[cls] + 0.02)
                print(f"[Stage B] Raised threshold for {cls} → {current_thr[cls]:.2f} (count={count})")

        if len(candidates) >= MAX_TOTAL:
            break
        current_thr = {k: max(0.75, v - 0.04) for k, v in current_thr.items()}
        attempt += 1

    t2 = time.time()
    print(f"[Stage B] Candidate selection finished in {t2 - t1:.2f} sec, total={len(candidates)}")

    # =========================
    # Stage B: SAM refinement + batched masked features
    # =========================
    if sam_enabled and img is not None and len(candidates) > 0:
        t2 = time.time()
        print(f"[Stage B] SAM refinement enabled → processing {len(candidates)} candidates")

        # ---- Phase 1: Collect all ROIs and bookkeeping
        roi_list = []
        roi_meta = []  # (cand_idx, mask_idx, cand_cls, mask_box)

        # Collect candidate boxes
        cand_boxes = [(cand["x"], cand["y"], cand["w"], cand["h"]) for cand in candidates]

        # Run SAM ONCE
        all_masks = masks_func(img, cand_boxes, predictor_func, sam_max_masks=3)

        # Now iterate candidates + masks together
        for cand_idx, (cand, masks) in enumerate(zip(candidates, all_masks)):
            cls = cand["class"]
            x0, y0, w, h = cand["x"], cand["y"], cand["w"], cand["h"]

            for mask_idx, m in enumerate(masks):
                # ✅ use offset=(x0,y0) so boxes are in global coords
                mask_box = mask_to_box(m, offset=(x0, y0))
                new_x, new_y, new_w, new_h = mask_box
                roi_rgb = img[new_y:new_y + new_h, new_x:new_x + new_w].copy()
                if roi_rgb.size == 0:
                    continue

                # force RGB
                if roi_rgb.ndim == 2:
                    roi_rgb = np.stack([roi_rgb] * 3, axis=-1)
                elif roi_rgb.shape[-1] != 3:
                    try:
                        roi_rgb = cv2.cvtColor(roi_rgb, cv2.COLOR_GRAY2RGB)
                    except Exception:
                        continue

                # apply mask
                mask_crop = m[new_y:new_y + new_h, new_x:new_x + new_w]
                if mask_crop.shape[:2] != roi_rgb.shape[:2]:
                    continue
                roi_rgb[mask_crop == 0] = 0

                # skip tiny crops
                if roi_rgb.shape[0] < 4 or roi_rgb.shape[1] < 4:
                    continue

                roi_list.append(roi_rgb)
                roi_meta.append((cand_idx, mask_idx, cls, mask_box))
        t_d0 = time.time()
        # ---- Phase 2: Batch embed all ROIs
        all_feats = batch_embed_rois(roi_list, batch_size=32)  # (N, D)
        t_d1 = time.time()
        print(f"[Stage D] Masked DINO embedding finished in {t_d1 - t_d0:.2f} sec, N={len(roi_list)}")
        # ---- Phase 3: Assign back to candidates
        refined = candidates.copy()
        best_by_cand = {}

        for feat, (cand_idx, mask_idx, cls, mask_box) in zip(all_feats, roi_meta):
            proto_list = prototypes.get(cls, [])
            if proto_list:
                P = torch.stack(proto_list)
                sims = torch.matmul(P, feat) / (
                        torch.norm(P, dim=1) * torch.norm(feat) + 1e-12
                )
                score = float(sims.max())
            else:
                score = -1.0

            if cand_idx not in best_by_cand or score > best_by_cand[cand_idx][0]:
                best_by_cand[cand_idx] = (score, cls, mask_box, feat)

        # ---- Phase 4: Update candidates
        for cand_idx, (score, cls, mask_box, feat) in best_by_cand.items():
            if score >= thr_dict.get(cls, 0.9):
                new_x, new_y, new_w, new_h = mask_box
                refined[cand_idx]["x"], refined[cand_idx]["y"] = new_x, new_y
                refined[cand_idx]["w"], refined[cand_idx]["h"] = new_w, new_h
                refined[cand_idx]["class"] = cls
                refined[cand_idx]["sim"] = score
                refined[cand_idx]["feat"] = feat

        candidates = refined
        t3 = time.time()
        print(f"[Stage B] SAM refinement finished in {t3 - t2:.2f} sec")
        if candidates:
            cls_counts = Counter([c["class"] for c in candidates])
            print("[Stage B] Per-class counts after SAM:", dict(cls_counts))

    print(f"[Summary] Total candidates after SAM: {len(candidates)} "
          f"(features kept as GPU tensors, export will convert to NumPy)")
    return candidates


def extract_stp_extra_windows(img, sizes=[16, 24], stride_frac=1.0, batch_size=64):
    """
    Extract additional small windows ONLY for STP class.
    Returns:
        extra_windows: list of (x, y, w, h, class)
        extra_feats:   list of torch tensors (embeddings)
    """
    H, W, _ = img.shape
    extra_windows, extra_patches = [], []

    for ps in sizes:
        stride = int(ps * stride_frac)
        count_for_ps = 0
        for y in range(0, H - ps + 1, stride):
            for x in range(0, W - ps + 1, stride):
                crop = img[y:y + ps, x:x + ps, :]
                extra_patches.append(crop)
                extra_windows.append((x, y, ps, ps, "STP"))
                count_for_ps += 1
        print(f"[DEBUG][STP] {ps}x{ps} → {count_for_ps} windows")

    if not extra_patches:
        return [], []

    feats_batch = extract_dino_embeddings_batch(extra_patches, batch_size=batch_size)
    extra_feats = [feats_batch[i] for i in range(feats_batch.shape[0])]

    print(f"[DEBUG][STP] Total injected STP windows: {len(extra_windows)}")
    return extra_windows, extra_feats


def apply_pca_whitening(feat, scaler_mean, scaler_scale, components):
    feat_std = (feat - scaler_mean) / scaler_scale
    return np.dot(feat_std, components.T)


def batch_embed_rois(rois, batch_size=64):
    """
    Efficiently embed a list of ROI crops with DINO.
    rois: list of HxWx3 numpy arrays (uint8)
    Returns: torch.Tensor of shape (N, D) on GPU
    """
    if not rois:
        return torch.empty(0, 768, device=device)  # adjust D if using another model
    return extract_dino_embeddings_batch(rois, batch_size=batch_size)


# ============================================================
# === PCA/Whitening support utilities =======================
# ============================================================


def apply_pca_transform(feat_np, pca_info):
    """
    Apply StandardScaler + PCA projection to a numpy feature vector.
    Used at inference time to ensure features are in same space as prototypes.
    """
    v_std = (feat_np - pca_info["mean"]) / pca_info["scale"]
    return np.dot(v_std, pca_info["components"].T)


def project_feature_torch_pure(feat_t, pca_info):
    mean_t = torch.tensor(pca_info["mean"], device=feat_t.device, dtype=feat_t.dtype)
    scale_t = torch.tensor(pca_info["scale"], device=feat_t.device, dtype=feat_t.dtype)
    components_t = torch.tensor(pca_info["components"], device=feat_t.device, dtype=feat_t.dtype)
    v_std_t = (feat_t - mean_t) / scale_t
    return v_std_t @ components_t.T


# ===== Main =====
def main():
    global classes
    ap = argparse.ArgumentParser("Auto-annotate with DINO embeddings + SAM2")
    ap.add_argument("--proto-file", type=str, default=None)
    ap.add_argument("-i", "--image", type=str, required=True)
    ap.add_argument("--band-indexes", type=str, default="3,2,1")
    ap.add_argument("--nir-index", type=int, default=None)
    ap.add_argument("--anno", type=str, default=None)
    ap.add_argument("--export-candidates", type=str, default=None)
    ap.add_argument("--use-masked-feats", action="store_true")
    ap.add_argument("--sam-max-masks", type=int, default=10,
                    help="Max masks to evaluate per ROI (default: 10)")
    ap.add_argument("--sam-min-area", type=int, default=200,
                    help="Ignore tiny masks (< pixels) in ROI coords")
    ap.add_argument("--sam2", action="store_true")
    ap.add_argument("--sam-checkpoint", type=str, default=None)
    ap.add_argument("--sam-topk", type=int, default=200)
    ap.add_argument("--review-only", action="store_true",
                    help="Skip detection, load existing candidates and open review UI")
    ap.add_argument("--classifier-file", type=str, default=None,
                    help="Path to trained sklearn classifier (joblib pkl). If given, replaces prototypes.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- Load prototypes ---
    with open(args.proto_file, "r") as f:
        data = json.load(f)

    if "classes" in data:
        # ✅ New PCA/global prototype format
        prototypes = {}
        for cls, entry in data["classes"].items():
            # Allow both old (dict with "prototypes") and new (list) formats
            if isinstance(entry, dict) and "prototypes" in entry:
                proto_list = entry["prototypes"]
            else:
                proto_list = entry

            normed_protos = []
            for p in proto_list:
                if isinstance(p, str):
                    try:
                        p = ast.literal_eval(p)
                    except Exception:
                        p = [float(x) for x in p.split(",") if x.strip()]

                pt = torch.tensor(p, dtype=torch.float32, device=device)
                # ✅ Normalize each prototype in PCA space
                pt = torch.nn.functional.normalize(pt, p=2, dim=0)
                normed_protos.append(pt)

            prototypes[cls] = normed_protos

        classes = list(prototypes.keys())
        print(f"[DEBUG] Loaded prototypes for {len(prototypes)} classes:")
        for cls, plist in prototypes.items():
            print(f"  {cls:12s}: {len(plist)} vectors, dim={plist[0].shape[0]}")
        use_pca_mode = True

    else:
        # ✅ Legacy JSON (no PCA)
        print("[Legacy Mode] Loading legacy prototype format...")
        prototypes = {}
        for cls, d in data.items():
            if cls == "__meta__":
                continue
            proto_list = []
            for v in d["prototypes"]:
                pt = torch.tensor(p, dtype=torch.float32, device=device)
                proto_list.append(pt)
            prototypes[cls] = proto_list

        classes = list(prototypes.keys())
        use_pca_mode = False

    # --- Load Global PCA Model ---
    print("[PCA Loader] Loading global PCA model ...")
    pca_info = load_global_pca()

    # Move PCA parameters to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_t = torch.tensor(pca_info["scaler_mean"], dtype=torch.float32, device=device)
    scale_t = torch.tensor(pca_info["scaler_scale"], dtype=torch.float32, device=device)
    components_t = torch.tensor(pca_info["components"], dtype=torch.float32, device=device)
    print(f"[PCA Loader] ✅ Using dim={len(pca_info['components'])}, "
          f"explains {sum(pca_info['explained_var']) * 100:.2f}% variance")

    def project_with_global_pca(feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)

        feats_std = (feats - mean_t) / scale_t
        feats_proj = torch.matmul(feats_std, components_t.T)
        feats_proj = torch.nn.functional.normalize(feats_proj, p=2, dim=1)

        # ✅ squeeze back if this was a single vector
        if feats_proj.shape[0] == 1:
            feats_proj = feats_proj.squeeze(0)

        return feats_proj

    # Load image bands
    parts = [int(x) for x in args.band_indexes.split(",")]
    with rasterio.open(args.image) as src:
        img = np.stack([(src.read(b).astype(np.float32) - src.read(b).min()) /
                        (src.read(b).ptp() + 1e-6) * 255 for b in parts], -1).astype(np.uint8)

    if args.use_masked_feats and args.sam2 and args.sam_checkpoint:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2 = build_sam2("sam2_hiera_l.yaml", args.sam_checkpoint)
        sam_predictor = SAM2ImagePredictor(sam2)

        def masks_from_boxes_local(image, boxes, sam_predictor, sam_max_masks=3, chunk_size=16):
            """
            Run SAM2 once per image, predict masks for candidate boxes,
            then crop them back to ROI coordinates instead of keeping full HxW masks.
            Returns list-of-lists of cropped masks aligned with `boxes`.
            """
            if len(boxes) == 0:
                return []

            sam_boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
            sam_predictor.set_image(image)
            H, W = image.shape[:2]

            all_masks_out = []
            for i in range(0, len(sam_boxes), chunk_size):
                batch_boxes = sam_boxes[i:i + chunk_size]

                masks, _, _ = sam_predictor.predict(
                    box=batch_boxes,
                    multimask_output=True
                )

                if masks.ndim == 3:  # (C,H,W) case
                    masks = masks[None, ...]

                N, C, Mh, Mw = masks.shape
                assert (Mh, Mw) == (H, W), f"Expected {H}x{W}, got {Mh}x{Mw}"

                if C > sam_max_masks:
                    masks = masks[:, :sam_max_masks]

                for j, (x0, y0, x1, y1) in enumerate(batch_boxes):
                    cropped = []
                    for k in range(masks.shape[1]):
                        m = masks[j, k].astype(np.uint8)  # (H,W)
                        m_crop = m[y0:y1, x0:x1]  # ✅ Local crop
                        cropped.append(m_crop)
                    all_masks_out.append(cropped)

            return all_masks_out

    meta_path = args.export_candidates
    # Ensure proper suffix
    if not meta_path.endswith("_meta.csv"):
        meta_path = meta_path + "_meta.csv"
    feats_path = meta_path.replace("_meta.csv", "_feats.npy")

    if args.review_only:
        # Review-only mode
        if not (os.path.exists(meta_path) and os.path.exists(feats_path)):
            raise FileNotFoundError(f"Review-only mode requires existing {meta_path} and {feats_path}")
        import pandas as pd
        print(f"⚡ Review-only mode: loading candidates from {meta_path}")
        df = pd.read_csv(meta_path)
        feats = np.load(feats_path, allow_pickle=True)
        candidates = df.to_dict("records")
        for i, c in enumerate(candidates):
            c["feat"] = feats[i]
    else:
        # Normal detection mode
        if os.path.exists(meta_path) and os.path.exists(feats_path):
            print(f"⚡ Found existing candidates → loading {meta_path}")
            import pandas as pd
            df = pd.read_csv(meta_path)
            feats = np.load(feats_path, allow_pickle=True)
            candidates = df.to_dict("records")
            for i, c in enumerate(candidates):
                c["feat"] = feats[i]
        else:
            print("🔍 Running detection...")
            # allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            # reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            # print(f"[MEM][Before Stage A] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

            # Stage A: compute all embeddings once
            all_windows, all_feats, size_to_indices = extract_dino_features_all(img, stride_frac=1.0, batch_size=512)

            candidates = detect_candidates_precomputed(
                all_windows,
                all_feats,
                size_to_indices,
                classes,
                prototypes,
                thr_dict=CLASS_SIM_THRESH.copy(),
                enforce_balance=True,
                sam_enabled=args.sam2,
                masks_func=masks_from_boxes_local,
                predictor_func=sam_predictor,
                img=img,
                pcaproj=project_with_global_pca
            )

            print(f"[DEBUG] Final candidate count: {len(candidates)}")
            #
            # allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            # reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            # print(f"[MEM][After detection] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

            import pandas as pd
            pd.DataFrame([{
                "class": c["class"],
                "x": c["x"],
                "y": c["y"],
                "w": c["w"],
                "h": c["h"],
                "sim": c["sim"]
            } for c in candidates]).to_csv(meta_path, index=False)

            #np.save(feats_path, np.vstack([c["feat"].detach().cpu().numpy() for c in candidates]))
            print(f"✅ Candidates exported → {meta_path}, {feats_path}")
            # print(f"[DEBUG] Exported {len(df)} candidates. First 5 rows:")
            # print(df.head())

    if args.anno:  # only if GT available
        include_ndwi = False
        ndwi_bins = None
        G_raw = None
        NIR_raw = None
        deps = ReviewDeps(save_yolo_func=save_yolo_func)
        sam_cfg = SamConfig(enabled=args.sam2, checkpoint=args.sam_checkpoint)

        # print(f"[DEBUG] Passing {len(candidates)} candidates to review")
        #  if len(candidates) > 0:
        #      print("[DEBUG] First candidate example:", candidates[0])
        review_and_save(
            img,
            candidates,  # full dicts with x,y,w,h, class, etc.
            args.export_candidates,
            classes,
            slices=None,
            prototypes=prototypes,
            lbp_scales=None,
            include_ndwi=False,
            ndwi_bins=None,
            G_raw=None,
            NIR_raw=None,
            deps=deps,
            sam=sam_cfg,
            state_path=os.path.join(os.path.dirname(args.export_candidates), "review_state.json"),
            hide_discarded=False
        )


if __name__ == "__main__":
    main()
