#!/usr/bin/env bash
# Unattended full experiment: 3 architectures x 3 seeds, then evaluation
# (plain + TTA) on the held-out Kvasir-SEG test set and any external
# cross-dataset test sets found under data/external/.
#
# Usage (from the project root on the pod):
#   bash polypseg/run_all.sh 2>&1 | tee polypseg/outputs/run_all.log
set -euo pipefail

MODELS=("unet" "unetpp" "segformer")
SEEDS=(42 1337 2024)
SIZE=352
BATCH=16
EPOCHS=60
WORKERS=8

mkdir -p polypseg/outputs polypseg/runs

# Assemble --external args for whichever cross-dataset sets were uploaded.
EXTERNAL=()
for d in data/external/*/; do
  [ -d "${d}images" ] && [ -d "${d}masks" ] || continue
  EXTERNAL+=("$(basename "$d")=$d")
done
if [ ${#EXTERNAL[@]} -gt 0 ]; then
  echo "External test sets: ${EXTERNAL[*]}"
else
  echo "No external test sets found under data/external/ — in-domain only."
fi

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run="${model}_seed${seed}"
    if [ -f "polypseg/runs/${run}/summary.json" ]; then
      echo "== skip ${run} (already complete)"
      continue
    fi
    echo "== train ${run}"
    python -m polypseg.src.train \
      --model "$model" --seed "$seed" \
      --size "$SIZE" --batch-size "$BATCH" --epochs "$EPOCHS" --num-workers "$WORKERS"
  done
done

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run="${model}_seed${seed}"
    ckpt="polypseg/runs/${run}/best.pth"
    [ -f "$ckpt" ] || { echo "!! missing ${ckpt}, skipping eval"; continue; }

    echo "== eval ${run}"
    python -m polypseg.src.evaluate \
      --checkpoint "$ckpt" --model "$model" --size "$SIZE" \
      --out "polypseg/outputs/perimage_${run}.csv" \
      ${EXTERNAL[@]+--external "${EXTERNAL[@]}"}

    echo "== eval ${run} (TTA)"
    python -m polypseg.src.evaluate \
      --checkpoint "$ckpt" --model "$model" --size "$SIZE" --tta \
      --out "polypseg/outputs/perimage_${run}_tta.csv" \
      ${EXTERNAL[@]+--external "${EXTERNAL[@]}"}
  done
done

echo "== all done"
