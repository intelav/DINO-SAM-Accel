#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# -----------------------------
# batch_auto_annotate_dino.sh
# -----------------------------
# Runs auto_annotate_dino.py on all .tif images in a dataset directory
# using a combined prototypes.json (8 classes).
# Skips images if *_candidates_meta.csv already exists,
# so you can stop/restart safely.
# -----------------------------

AUTO_ANN_PY="./auto_annotate_dino_nvtx_optimized.py"
COMBINED_PROTO="./runs/combined_dino_prototypes.json"

# Input dataset directory (adjust path)
DATASET_DIR="/media/avaish/aiwork/satellite-work/visual_search_dl/dataset/development_set/"

# Output base directorys
RUNS_DIR="./runs"

# Parameters
RGB_BANDS="3,2,1"
NIR_INDEX=4

mkdir -p "${RUNS_DIR}"

# Counters
processed=0
skipped=0
failed=0

for TIF in "${DATASET_DIR}"/*.tif; do
  BASENAME="$(basename "${TIF}" .tif)"
  RUN_DIR="${RUNS_DIR}/${BASENAME}_run"
  CAND_DIR="${RUN_DIR}/candidates"
  mkdir -p "${CAND_DIR}"

  CAND_PREFIX="${CAND_DIR}/${BASENAME}_candidates"
  META="${CAND_PREFIX}_meta.csv"

  # Skip if already done
  if [ -f "${META}" ]; then
    echo "✅ Skipping ${BASENAME} (already has ${META})"
    skipped=$((skipped+1))
    continue
  fi

  echo "=== Processing ${BASENAME} ==="
  if python "${AUTO_ANN_PY}" \
    --proto-file "${COMBINED_PROTO}" \
    -i "${TIF}" \
    --band-indexes "${RGB_BANDS}" \
    --nir-index "${NIR_INDEX}" \
    --export-candidates "${CAND_PREFIX}" \
    --sam2 \
    --use-masked-feats \
    --sam-checkpoint "/media/avaish/aiwork/satellite-work/satellite_annotator/pre-trained-models/sam2_hiera_large.pt"; then
#  if python "${AUTO_ANN_PY}" \
#    --classifier-file "./runs/dino_classifier.pkl" \
#    -i "${TIF}" \
#    --band-indexes "${RGB_BANDS}" \
#    --nir-index "${NIR_INDEX}" \
#    --export-candidates "${CAND_PREFIX}" \
#    --sam2 \
#    --use-masked-feats \
#    --sam-checkpoint "/media/avaish/aiwork/satellite-work/satellite_annotator/pre-trained-models/sam2_hiera_large.pt"; then
  processed=$((processed+1))
 else
  echo "❌ Failed on ${BASENAME}"
  failed=$((failed+1))
 fi


#  if python "${AUTO_ANN_PY}" \
#      --proto-file "${COMBINED_PROTO}" \
#      -i "${TIF}" \
#      --band-indexes "${RGB_BANDS}" \
#      --nir-index "${NIR_INDEX}" \
#      --export-candidates "${CAND_PREFIX}"; then
#    processed=$((processed+1))
#  else
#    echo "❌ Failed on ${BASENAME}"
#    failed=$((failed+1))
#  fi
done

echo
echo "=== Batch summary ==="
echo "  Processed : $processed"
echo "  Skipped   : $skipped"
echo "  Failed    : $failed"
echo "Results saved under ${RUNS_DIR}/*_run/candidates/"

#/media/avaish/aiwork/satellite-work/visual_search_dl/auto_annotate_dino_profiled.py \
#    --proto-file /media/avaish/aiwork/satellite-work/visual_search_dl/combined_prototypes.json \
#    -i /media/avaish/aiwork/satellite-work/visual_search_dl/dataset/development_set/GC01PS03D0028.tif \
#    --band-indexes 3,2,1 \
#    --nir-index 4 \
#    --export-candidates /media/avaish/aiwork/satellite-work/visual_search_dl/runs/GC01PS03D0028_run_augmented_prototypes/candidates/GC01PS03D0028_candidates \
#    --sam2 \
#    --use-masked-feats \
#    --sam-checkpoint /media/avaish/aiwork/satellite-work/satellite_annotator/pre-trained-models/sam2_hiera_large.pt