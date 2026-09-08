#!/usr/bin/env bash
# Measure run-to-run variance under a fixed configuration.
#
# Phase A: every architecture is trained three times on fold 0 with an
#          identical command and identical seed. Any spread in the resulting
#          test Dice is pure implementation non-determinism (cuDNN kernel
#          selection, atomic accumulation order), not a modelling choice.
#          This gives the noise floor against which architecture differences
#          must be judged.
#
# Phase B: two architectures are repeated with --deterministic to confirm the
#          spread collapses once kernels are pinned, supporting the paper's
#          recommendation.
#
# Evaluation is in-domain plus all external sets, no TTA (not needed here).
#
# Usage:  bash polypseg/run_repeat.sh 2>&1 | tee polypseg/outputs/run_repeat.log
set -euo pipefail

MODELS=("unet" "unetpp" "segformer" "segformer_b2" "unet_pvt")
REPEATS=(1 2 3)
DET_MODELS=("unet" "segformer_b2")
DET_REPEATS=(1 2)
FOLD=0
SEED=42
SIZE=352
BATCH=16
EPOCHS=60
WORKERS=8
RUNS_DIR="polypseg/runs_repeat"
OUT_DIR="polypseg/outputs/repeat"
SPLITS="polypseg/outputs/folds/fold${FOLD}.csv"

mkdir -p "$RUNS_DIR" "$OUT_DIR"

EXTERNAL=()
for d in data/external/*/; do
  [ -d "${d}images" ] && [ -d "${d}masks" ] || continue
  EXTERNAL+=("$(basename "$d")=$d")
done
echo "External test sets: ${EXTERNAL[*]:-none}"

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
    echo "== eval ${run}"
    python -m polypseg.src.evaluate \
      --checkpoint "$ckpt" --model "$model" --size "$SIZE" \
      --splits "$SPLITS" --split test \
      --out "${OUT_DIR}/perimage_${run}.csv" \
      ${EXTERNAL[@]+--external "${EXTERNAL[@]}"}
  fi
}

echo "### Phase A: repeated runs, default (non-deterministic) kernels"
for model in "${MODELS[@]}"; do
  for r in "${REPEATS[@]}"; do
    run_one "$model" "fold${FOLD}rep${r}"
  done
done

echo "### Phase B: repeated runs with deterministic kernels"
for model in "${DET_MODELS[@]}"; do
  for r in "${DET_REPEATS[@]}"; do
    run_one "$model" "fold${FOLD}det${r}" "--deterministic"
  done
done

echo "== repeat study complete"
