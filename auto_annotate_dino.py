#!/usr/bin/env python3
# Auto-annotate with DINO embeddings + optional SAM2 masked refinement
import os, json, argparse, glob

import joblib
import numpy as np
import cv2, rasterio
import pandas as pd
import torch
from torch import cosine_similarity
from torchvision.ops import nms
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from batch_stats import *
from ui_review import ReviewDeps, SamConfig, review_and_save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use:", device, " | Current GPU:", torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU")

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

# CLASS_PATCH_SIZES = {
#     "STP": [32],
#     "Sheds": [32, 40, 64],
#     "Metro Shed": [64, 96, 128],
#     "Solar Panel": [64, 96, 128],
#     "Play Ground": [96, 128, 160],
#     "Pond-1": [40, 64, 96],
#     "Pond-2": [40, 64, 96],
#     "Brick Kiln": [32, 48, 64],
# }
STRIDE_FRAC = 1.0
SIM_THRESH = 0.90
CLASS_SIM_THRESH = {
    "Brick Kiln": 0.90,
    "Pond-1": 0.92,
    "Pond-2": 0.92,
    "Sheds": 0.93,
    "Metro Shed": 0.93,
    "Solar Panel": 0.95,
    "Play Ground": 0.94,
    "STP": 0.93,
}


# CLASS_SIM_THRESH = {cls: 0.90 for cls in
#                     ["Brick Kiln", "Metro Shed", "Play Ground", "Pond-1", "Pond-2", "STP", "Sheds", "Solar Panel"]}


# ===== DINO Embedding =====
def extract_dino_embeddings_batch(patches, batch_size=1024):
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
            outputs = model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :]  # (B, D) on GPU
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)  # L2 normalize on GPU
        all_feats.append(feats)
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


# ===== Detection =====
def detect_candidates(img, classes, prototypes, per_class_thr,
                      use_masked_feats=False, sam_enabled=False,
                      sam_predictor=None, sam_topk=200,
                      sam_max_masks=10, sam_min_area=150, stats=None):
    H, W = img.shape[:2]
    all_windows, accepted = [], {}
    next_idx = 0

    patch_list, meta_list = [], []

    # ---- Stage 1: Collect patches per class using CLASS_PATCH_SIZES ----
    for cls in classes:
        #sizes = CLASS_PATCH_SIZES.get(cls, [32, 40, 48, 64])
        sizes = [32, 40, 48, 64]
        for PS in sizes:
            stride = max(1, int(PS * STRIDE_FRAC))
            for y in range(0, H - PS + 1, stride):
                for x in range(0, W - PS + 1, stride):
                    patch = img[y:y + PS, x:x + PS]
                    patch_list.append(patch)
                    meta_list.append((x, y, PS, cls))  # bind patch to this class

    print(f"[Stage 1] Total windows collected: {len(patch_list)}")

    # ---- Stage 1b: Batch embedding ----
    feats = extract_dino_embeddings_batch(patch_list, batch_size=1024)  # torch tensor on GPU

    # ---- Attach features with class-specific similarity ----
    for i, (x, y, PS, target_cls) in enumerate(meta_list):
        feat = feats[i]  # torch tensor (1, D)
        P = torch.stack(prototypes[target_cls]).to(device)  # (M, D)
        sims = torch.matmul(feat, P.T)  # (M,)
        best_sim = float(sims.max())
        # Always store NumPy
        feat_np = feat.detach().cpu().numpy()
        all_windows.append({
            "win_idx": next_idx,
            "x": x, "y": y, "PS": PS,
            "feat": feat_np,
            "class": target_cls,
            "best_sim": best_sim
        })
        next_idx += 1

    print(f"[Stage 1] Windows: {len(all_windows)}")
    if stats: stats.mark_stage1(len(all_windows))

    # ---- Stage 2: Accept above-threshold ----
    for w in all_windows:
        thr = per_class_thr.get(w["class"], SIM_THRESH)
        if w["best_sim"] >= thr:
            accepted[(w["win_idx"], w["class"])] = {
                "bbox": (w["x"], w["y"], w["PS"], w["PS"]),
                "best_sim": w["best_sim"],
                "feat": w["feat"],  # already NumPy
                "class": w["class"]
            }
    print(f"[Stage 2] Accepted: {len(accepted)}")
    if stats: stats.mark_stage2(len(all_windows), len(accepted))

    # ---- Stage 3: Optional SAM masked refinement ----
    if use_masked_feats and sam_enabled and sam_predictor is not None:
        scored = sorted(all_windows, key=lambda ww: ww["best_sim"], reverse=True)[:sam_topk]
        for w in scored:
            x, y, PS = w["x"], w["y"], w["PS"]
            roi_rgb = img[y:y + PS, x:x + PS].copy()
            masks = sam_predictor.predict(roi_rgb)
            if masks is None:
                continue
            thr = per_class_thr.get(w["class"], SIM_THRESH)
            best_mask_sim = 0.0
            for m in masks:
                masked_feat = extract_features_masked_dino(roi_rgb, m)  # should return NumPy
                sims = [float(np.dot(masked_feat, p.cpu().numpy()) /
                              (np.linalg.norm(masked_feat) * np.linalg.norm(p.cpu().numpy()) + 1e-12))
                        for p in prototypes[w["class"]]]
                best_mask_sim = max(best_mask_sim, max(sims))
            if best_mask_sim >= thr:
                accepted[(w["win_idx"], w["class"])] = {
                    "bbox": (x, y, PS, PS),
                    "best_sim": best_mask_sim,
                    "feat": w["feat"],  # still NumPy
                    "class": w["class"]
                }
    print(f"[Stage 3] After SAM: {len(accepted)}")
    # if stats: stats.mark_stage3(len(accepted))
    # print("[DEBUG] Accepted before NMS:", len(accepted))
    # for (win_idx, cls), info in accepted.items():
    #     print(f" idx={win_idx}, class={cls}, bbox={info['bbox']}, sim={info['best_sim']:.3f}")

    # ---- Stage 4: NMS ----
    if not accepted:
        return []
    print(f"[Stage 4] Skipped NMS — keeping all {len(accepted)} candidates")
    # boxes, scores, keys = [], [], []
    # for (win_idx, cls), info in accepted.items():
    #     x, y, w, h = info["bbox"]
    #     boxes.append([x, y, x+w, y+h])
    #     scores.append(info["best_sim"])
    #     keys.append((win_idx, cls))
    # print("[DEBUG] IoU checks between accepted boxes:")
    # for i in range(len(boxes)):
    #     for j in range(i + 1, len(boxes)):
    #         iou_val = _iou(boxes[i], boxes[j])
    #         if iou_val > 0:
    #             print(f"  IoU({i},{j})={iou_val:.3f} between {boxes[i]} and {boxes[j]}")
    # keep = nms(torch.tensor(boxes, dtype=torch.float32),
    #            torch.tensor(scores, dtype=torch.float32),
    #            iou_threshold=0.4)
    # accepted = {keys[i]: accepted[keys[i]] for i in keep.tolist()}
    # print(f"[Stage 4] Final: {len(accepted)}")
    # if stats: stats.mark_stage4(len(accepted))

    return [{
        "idx": int(win_idx),
        "class": cls,
        "x": int(info["bbox"][0]),
        "y": int(info["bbox"][1]),
        "w": int(info["bbox"][2]),
        "h": int(info["bbox"][3]),
        "sim": float(info["best_sim"]),
        "feat": info["feat"]  # NumPy
    } for (win_idx, cls), info in accepted.items()]


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


def extract_dino_features(img, band_indexes, class_patch_sizes, stride_frac=1.0, batch_size=256):
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


def mask_to_box(mask):
    """
    Convert a binary mask (H,W) to a bounding box [x, y, w, h].
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return (int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1))


import time

import time
from collections import Counter

def detect_candidates_precomputed(
        all_windows,
        all_feats,
        classes,
        prototypes,
        clf,
        thr_dict,
        enforce_balance=False,
        sam_enabled=False,
        masks_func=None,
        use_masked_feats=False,
        img=None,
        image_name=None
):
    """
    Detect candidate windows given precomputed features and class prototypes,
    with optional SAM refinement (tight bounding boxes + batched masked feats).
    """

    candidates = []
    current_thr = thr_dict.copy()

    # =========================
    # Stage A: adaptive loop
    # =========================
    # =========================
    # Stage A: Precompute similarities ONCE
    # =========================
    t0 = time.time()
    print(f"[Stage A] Precomputing similarities for {len(all_feats)} windows × {len(classes)} classes")

    sim_cache = []  # list of dicts: [{cls: sim, ...}, ...]
    for feat_t in all_feats:
        win_sims = {}
        for cls in classes:
            if clf is not None:
                # Classifier mode
                feat_np = feat_t.detach().cpu().numpy().reshape(1, -1)
                probs = clf.predict_proba(feat_np)[0]
                for c, p in zip(clf.classes_, probs):
                    win_sims[c] = float(p)
            else:
                # Prototype mode
                proto_list = prototypes.get(cls, [])
                if proto_list:
                    P = torch.stack(proto_list)
                    sims = torch.matmul(P, feat_t) / (
                            torch.norm(P, dim=1) * torch.norm(feat_t) + 1e-12
                    )
                    win_sims[cls] = float(sims.max())

        sim_cache.append(win_sims)
    print(f"[DEBUG] Total windows processed: {len(all_feats)}")
    print(f"[DEBUG] Total win_sims cached: {len(sim_cache)} (each has up to {len(classes)} entries)")

    t1 = time.time()
    print(f"[Stage A] Similarity cache built in {t1 - t0:.2f} sec")

    # =========================
    # Stage A: Adaptive threshold loop (cheap now)
    # =========================
    candidates = []
    current_thr = thr_dict.copy()
    attempt = 0
    MAX_ATTEMPTS = 5

    while min(current_thr.values()) >= 0.75 and len(candidates) < MAX_TOTAL and attempt < MAX_ATTEMPTS:
        new_candidates = []
        for (x, y, w, h), win_sims, feat_t in zip(all_windows, sim_cache, all_feats):
            for cls in classes:
                sim = win_sims.get(cls, -1.0)
                if sim >= current_thr.get(cls, 0.9):
                    new_candidates.append({
                        "x": x, "y": y, "w": w, "h": h,
                        "class": cls,
                        "sim": sim,
                        "feat": feat_t
                    })
                    break  # assign to first qualifying class

        print(f"[Stage A] Attempt {attempt+1}: accepted {len(new_candidates)} candidates")

        # Balance
        if enforce_balance and new_candidates:
            balanced = []
            for cls in classes:
                cls_cands = [c for c in new_candidates if c["class"] == cls]
                cls_cands = sorted(cls_cands, key=lambda c: c["sim"], reverse=True)[:MAX_PER_CLASS]
                balanced.extend(cls_cands)
            new_candidates = balanced
            print(f"[Stage A] After balancing → {len(new_candidates)} candidates")

            # Enforce global cap as well
            if len(new_candidates) > MAX_TOTAL:
                new_candidates = sorted(new_candidates, key=lambda c: c["sim"], reverse=True)[:MAX_TOTAL]
                print(f"[Stage A] Truncated to global MAX_TOTAL={MAX_TOTAL} → {len(new_candidates)} candidates")

        candidates = new_candidates
        from collections import Counter
        cls_counts = Counter([c["class"] for c in candidates])
        for cls, count in cls_counts.items():
            if count > MAX_PER_CLASS:
                current_thr[cls] = min(0.99, current_thr[cls] + 0.02)
                print(f"[Stage A] Raised threshold for {cls} → {current_thr[cls]:.2f} (count={count})")

        if len(candidates) >= MAX_TOTAL:
            break
        current_thr = {k: max(0.75, v - 0.04) for k, v in current_thr.items()}
        attempt += 1

    t2 = time.time()
    print(f"[Stage A] Final candidate count before SAM: {len(candidates)} "
          f"(finished in {t2 - t0:.2f} sec, {attempt} attempts)")
    if candidates:
        from collections import Counter
        print("[Stage A] Per-class counts:", dict(Counter([c["class"] for c in candidates])))


    # =========================
    # Stage B: SAM refinement + batched masked features
    # =========================
    if sam_enabled and img is not None and len(candidates) > 0:
        t2 = time.time()
        print(f"[Stage B] SAM refinement enabled → processing {len(candidates)} candidates")

        refined = []

        for idx, cand in enumerate(candidates):
            if idx % 100 == 0 and idx > 0:
                print(f"[Stage B] Processed {idx}/{len(candidates)} candidates for SAM masks")

            x, y, w, h = cand["x"], cand["y"], cand["w"], cand["h"]
            cls = cand["class"]

            try:
                masks = masks_func(img, (x, y, w, h))
            except Exception as e:
                print(f"[Stage B][WARN] SAM failed on {cls} candidate: {e}")
                refined.append(cand)
                continue

            if masks is not None and len(masks) > 0:
                best_mask, best_mask_score, best_mask_cls, best_mask_feat = None, -1.0, None, None
                for m in masks:
                    mask_box = mask_to_box(m)
                    new_x, new_y, new_w, new_h = mask_box
                    roi_rgb = img[new_y:new_y + new_h, new_x:new_x + new_w].copy()
                    if roi_rgb.size == 0:
                        continue
                    if roi_rgb.ndim == 2:  # grayscale fallback
                        roi_rgb = np.stack([roi_rgb] * 3, axis=-1)
                    roi_masked = roi_rgb.copy()
                    mask_crop = m[new_y:new_y + new_h, new_x:new_x + new_w]
                    if mask_crop.shape[:2] != roi_masked.shape[:2]:
                        continue  # skip invalid mask
                    roi_masked[mask_crop == 0] = 0
                    #roi_masked[m[new_y:new_y + new_h, new_x:new_x + new_w] == 0] = 0

                    # Ensure proper RGB shape
                    if roi_masked.ndim == 2:
                        roi_masked = np.stack([roi_masked] * 3, axis=-1)
                    elif roi_masked.shape[-1] != 3:
                        try:
                            roi_masked = cv2.cvtColor(roi_masked, cv2.COLOR_GRAY2RGB)
                        except Exception:
                            continue
                    # Skip absurdly small crops
                    if roi_masked.shape[0] < 4 or roi_masked.shape[1] < 4:
                        continue
                    # Embed masked ROI
                    masked_feat = extract_dino_embeddings_batch([roi_masked], batch_size=1)[0]

                    if clf is not None:
                        # 🔹 Classifier mode: get predicted class + probability
                        feat_np = masked_feat.detach().cpu().numpy().reshape(1, -1)
                        probs = clf.predict_proba(feat_np)[0]
                        best_idx = probs.argmax()
                        cand_cls = clf.classes_[best_idx]
                        score = float(probs[best_idx])
                    else:
                        # 🔹 Prototype mode: keep candidate's original class
                        cand_cls = cls
                        proto_list = prototypes.get(cand_cls, [])
                        if proto_list:
                            P = torch.stack(proto_list)
                            sims = torch.matmul(P, masked_feat) / (
                                    torch.norm(P, dim=1) * torch.norm(masked_feat) + 1e-12
                            )
                            score = float(sims.max())
                        else:
                            score = -1.0

                    # Keep best mask across all hypotheses
                    if score > best_mask_score:
                        best_mask, best_mask_score = mask_box, score
                        best_mask_cls, best_mask_feat = cand_cls, masked_feat

                # 🔹 Update candidate if a valid mask was found
                if best_mask is not None and best_mask_score >= thr_dict.get(best_mask_cls, 0.9):
                    orig_box = (cand["x"], cand["y"], cand["w"], cand["h"])
                    new_x, new_y, new_w, new_h = best_mask
                    cand["x"], cand["y"], cand["w"], cand["h"] = new_x, new_y, new_w, new_h
                    cand["class"] = best_mask_cls  # 🔹 update class
                    cand["sim"] = best_mask_score  # 🔹 update confidence
                    cand["feat"] = best_mask_feat  # 🔹 update embedding

                    # Debug logging
                    debug_log = os.path.join("runs", "debug_bbox_shift.txt")
                    ox, oy, ow, oh = orig_box
                    dx, dy = new_x - ox, new_y - oy
                    dw, dh = new_w - ow, new_h - oh
                    with open(debug_log, "a") as f:
                        f.write(
                            f"{os.path.basename(image_name)}, class={cand['class']}, "
                            f"orig=({ox},{oy},{ow},{oh}) → new=({new_x},{new_y},{new_w},{new_h}), "
                            f"shift=({dx},{dy},{dw},{dh}), sim={best_mask_score:.3f}\n"
                        )
            else:
                print(f"[Stage B] No mask found for candidate {idx} ({cls})")

            refined.append(cand)

        candidates = refined
        t3 = time.time()
        print(f"[Stage B] SAM refinement finished in {t3 - t2:.2f} sec")
        if candidates:
            cls_counts = Counter([c["class"] for c in candidates])
            print("[Stage B] Per-class counts after SAM:", dict(cls_counts))

    print(f"[Summary] Total candidates after SAM: {len(candidates)} "
          f"(features kept as GPU tensors, export will convert to NumPy)")
    return candidates


def extract_stp_extra_windows(img, sizes=[16, 24], stride_frac=1.0, batch_size=256):
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
                crop = img[y:y+ps, x:x+ps, :]
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



# ===== Main =====
def main():
    ap = argparse.ArgumentParser("Auto-annotate with DINO embeddings + SAM2")
    ap.add_argument("--proto-file", type=str,  default=None)
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

    # Load prototypes or Classfier
    prototypes = None
    clf = None

    if args.classifier_file:
        # Classifier mode
        clf = joblib.load(args.classifier_file)
        classes = clf.classes_.tolist()
        print(f"[INFO] Loaded classifier with classes: {classes}")
    else:
        # Prototype mode
        with open(args.proto_file) as f:
            raw_protos = json.load(f)
        prototypes = {
            cls: [torch.tensor(v, dtype=torch.float32, device=device)
                  for v in d["prototypes"]]
            for cls, d in raw_protos.items()
            if cls != "__meta__"
        }
        classes = list(prototypes.keys())
        print(f"[INFO] Loaded prototypes for classes: {classes}")

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

        def masks_from_roi_func(image, box):
            """Run SAM on a candidate ROI and return masks."""
            x, y, w, h = box
            roi = image[y:y + h, x:x + w]
            if roi.size == 0:
                return []
            sam_predictor.set_image(image)
            masks, _, _ = sam_predictor.predict(box=np.array([x, y, x + w, y + h]))
            return masks


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
            # candidates = detect_candidates(
            #     img, classes, prototypes, CLASS_SIM_THRESH,
            #     use_masked_feats=args.use_masked_feats,
            #     sam_enabled=args.sam2, sam_predictor=sam_predictor,
            #     sam_topk=args.sam_topk, sam_max_masks=args.sam_max_masks,
            #     sam_min_area=args.sam_min_area
            # )
            all_windows, all_feats = extract_dino_features(img, args.band_indexes, CLASS_PATCH_SIZES)

            # --- Inject STP-only tiny windows ---
            # extra_windows, extra_feats = extract_stp_extra_windows(img, sizes=[16, 24], stride_frac=STRIDE_FRAC)
            # if extra_windows:
            #     all_windows.extend(extra_windows)
            #     all_feats.extend(extra_feats)
            #     print(f"[DEBUG] Final window count after adding STP extras: {len(all_windows)}")

            candidates = detect_candidates_precomputed(
                all_windows,
                all_feats,
                classes,
                prototypes,
                clf,
                thr_dict=CLASS_SIM_THRESH.copy(),
                enforce_balance=True,
                sam_enabled=args.sam2,
                masks_func=masks_from_roi_func,
                use_masked_feats=args.use_masked_feats,
                img=img,
                image_name=args.image
            )
            print(f"[DEBUG] Final candidate count: {len(candidates)}")

            import pandas as pd
            pd.DataFrame([{
                "class": c["class"],
                "x": c["x"],
                "y": c["y"],
                "w": c["w"],
                "h": c["h"],
                "sim": c["sim"]
            } for c in candidates]).to_csv(meta_path, index=False)

            np.save(feats_path, np.vstack([c["feat"].detach().cpu().numpy() for c in candidates]))
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