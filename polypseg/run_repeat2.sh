#!/usr/bin/env bash
# Decompose run-to-run variance by source.
#
# The first repeat study (run_repeat.sh) found that identical configurations
# produce test-Dice spreads up to 0.017, and that CUDA determinism flags did
# not remove them. The suspected cause is the data pipeline: DataLoader
# workers seeded NumPy but not Python's `random`, which Albumentations uses,
# so each run drew a different augmentation stream.
#
# `seed_worker` now seeds both. This script re-measures under two conditions
# to separate the contributions:
#
#   C  fixed worker seeding, default kernels
#      -> residual spread is CUDA kernel non-determinism
#   D  fixed worker seeding + --deterministic
#      -> spread should approach zero
#
# Together with the earlier conditions (unfixed seeding, with and without
# determinism flags) this isolates data-pipeline RNG from kernel choice.
#
# Evaluation is in-domain fold-0 test only: variance is the quantity of
# interest and external sets would multiply evaluation cost for no gain.
#
# Usage: bash polypseg/run_repeat2.sh 2>&1 | tee polypseg/outputs/run_repeat2.log
set -euo pipefail

MODELS=("unet" "unetpp" "segformer" "segformer_b2" "unet_pvt")
REPEATS=(1 2 3)
FOLD=0
SEED=42
SIZE=352
BATCH=16
EPOCHS=60
WORKERS=8
RUNS_DIR="polypseg/runs_repeat2"
OUT_DIR="polypseg/outputs/repeat2"
SPLITS="polypseg/outputs/folds/fold${FOLD}.csv"

mkdir -p "$RUNS_DIR" "$OUT_DIR"

run_one () {  # model, tag, extra_flags
  local model="$1" tag="$2" extra="${3:-}"
  local run="${model}_seed${SEED}_${tag}"
  if [ -f "${RUNS_DIR}/${run}/summary.json" ]; then
    echo "== skip ${run}"
  else
    echo "== train ${run} ${extra}"
    # shellcheck disable=SC2086
    python -m polypseg.src.train \
      --model "$model" --seed "$SEED" --tag "$tag" \
      --splits "$SPLITS" --out-dir "$RUNS_DIR" \
      --size "$SIZE" --batch-size "$BATCH" --epochs "$EPOCHS" \
      --num-workers "$WORKERS" $extra
  fi
  local ckpt="${RUNS_DIR}/${run}/best.pth"
  [ -f "$ckpt" ] || { echo "!! missing ${ckpt}"; return; }
  if [ ! -f "${OUT_DIR}/perimage_${run}.csv" ]; then
    python -m polypseg.src.evaluate \
      --checkpoint "$ckpt" --model "$model" --size "$SIZE" \
      --splits "$SPLITS" --split test \
      --out "${OUT_DIR}/perimage_${run}.csv"
  fi
}

echo "### Condition C: fixed worker seeding, default kernels"
for model in "${MODELS[@]}"; do
  for r in "${REPEATS[@]}"; do
    run_one "$model" "fold${FOLD}fixC${r}"
  done
done

echo "### Condition D: fixed worker seeding + deterministic kernels"
for model in "${MODELS[@]}"; do
  for r in "${REPEATS[@]}"; do
    run_one "$model" "fold${FOLD}fixD${r}" "--deterministic"
  done
done

echo "== variance decomposition complete"
