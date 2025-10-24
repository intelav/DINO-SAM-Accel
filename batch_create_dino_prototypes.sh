#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# -----------------------------
# batch_create_dino_prototypes.sh
# -----------------------------
# Creates DINO prototypes for all .tif + .json pairs in a directory
# and merges them into a single combined_prototypes.json
# -----------------------------

# --- Config ---
#CREATE_FEAT_PY="./create_features_dino_augmented.py"
COMBINE_PY="./combined_dino_prototypes.py"
CREATE_FEAT_PY="./create_features_dino.py"

# Input dataset directory (adjust path)
DATASET_DIR="/media/avaish/aiwork/satellite-work/visual_search_dl/dataset/sample_set/allfiles"

# Output run directory
RUNS_DIR="./runs"
COMBINED_JSON="${RUNS_DIR}/combined_prototypes.json"

# Parameters
RGB_BANDS="3,2,1"
#RGB_BANDS="1,2,3"
NIR_INDEX=4

# Ensure runs dir exists
mkdir -p "${RUNS_DIR}"

# Loop over all .tif files
for TIF in "${DATASET_DIR}"/*.tif; do
  BASENAME="$(basename "${TIF}" .tif)"
  JSON="${DATASET_DIR}/${BASENAME}.json"

  if [ ! -f "${JSON}" ]; then
    echo "⚠️  Skipping ${BASENAME}, no matching JSON found"
    continue
  fi

  echo "=== Processing ${BASENAME} ==="
  OUT_DIR="${RUNS_DIR}/${BASENAME}_run/prototypes"
  mkdir -p "${OUT_DIR}"

  python "${CREATE_FEAT_PY}" \
    --single-tif "${TIF}" \
    --single-json "${JSON}" \
    --out "${OUT_DIR}" \
    --band-indexes "${RGB_BANDS}" \
    --nir-index "${NIR_INDEX}" \
    --debug-save-all
done

#echo
#echo "=== Combining all prototypes into one JSON ==="
#python "${COMBINE_PY}" \
#  --proto-dirs "${RUNS_DIR}"/*_run/prototypes \
#  --out "${COMBINED_JSON}"
#
#echo
#echo "✅ Final combined prototypes → ${COMBINED_JSON}"
