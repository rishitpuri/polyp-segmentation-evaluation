#!/usr/bin/env bash
# 5-fold cross-validation: 3 architectures x 5 folds = 15 training runs.
# Every one of the 1000 Kvasir-SEG images is tested exactly once, so
# per-stratum estimates use n~333 (size tertiles) and n=196 (sessile)
# instead of the 33/19 available from a single split.
#
# Each fold's model is also evaluated on CVC-ClinicDB, giving 5 independent
# cross-dataset estimates per architecture.
#
# Usage (from the project root on the pod):
#   bash polypseg/run_cv.sh 2>&1 | tee polypseg/outputs/run_cv.log
set -euo pipefail

MODELS=("unet" "unetpp" "segformer" "segformer_b2" "unet_pvt")
FOLDS=(0 1 2 3 4)
SEED=42
SIZE=352
BATCH=16
EPOCHS=60
WORKERS=8
RUNS_DIR="polypseg/runs_cv"
OUT_DIR="polypseg/outputs/cv"

mkdir -p "$RUNS_DIR" "$OUT_DIR"

EXTERNAL=()
for d in data/external/*/; do
  [ -d "${d}images" ] && [ -d "${d}masks" ] || continue
  EXTERNAL+=("$(basename "$d")=$d")
done
echo "External test sets: ${EXTERNAL[*]:-none}"

for model in "${MODELS[@]}"; do
  for k in "${FOLDS[@]}"; do
    run="${model}_seed${SEED}_fold${k}"
    if [ -f "${RUNS_DIR}/${run}/summary.json" ]; then
      echo "== skip ${run} (already complete)"
    else
      echo "== train ${run}"
      python -m polypseg.src.train \
        --model "$model" --seed "$SEED" --tag "fold${k}" \
        --splits "polypseg/outputs/folds/fold${k}.csv" \
        --out-dir "$RUNS_DIR" \
        --size "$SIZE" --batch-size "$BATCH" --epochs "$EPOCHS" --num-workers "$WORKERS"
    fi

    ckpt="${RUNS_DIR}/${run}/best.pth"
    [ -f "$ckpt" ] || { echo "!! missing ${ckpt}"; continue; }

    if [ ! -f "${OUT_DIR}/perimage_${run}.csv" ]; then
      echo "== eval ${run}"
      python -m polypseg.src.evaluate \
        --checkpoint "$ckpt" --model "$model" --size "$SIZE" \
        --splits "polypseg/outputs/folds/fold${k}.csv" --split test \
        --out "${OUT_DIR}/perimage_${run}.csv" \
        ${EXTERNAL[@]+--external "${EXTERNAL[@]}"}
    fi

    if [ ! -f "${OUT_DIR}/perimage_${run}_tta.csv" ]; then
      echo "== eval ${run} (TTA)"
      python -m polypseg.src.evaluate \
        --checkpoint "$ckpt" --model "$model" --size "$SIZE" --tta \
        --splits "polypseg/outputs/folds/fold${k}.csv" --split test \
        --out "${OUT_DIR}/perimage_${run}_tta.csv" \
        ${EXTERNAL[@]+--external "${EXTERNAL[@]}"}
    fi
  done
done

echo "== cv complete"
