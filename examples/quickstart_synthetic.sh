#!/usr/bin/env bash
# quickstart_synthetic.sh — full MOSAIC run on the synthetic double-well
# benchmark with the paper-canonical hyperparameters (single seed).
# Generates data, trains Phase 1 + Phase 2, and evaluates MCC / Z@top3 / X_Z@top3.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEED="${SEED:-42}"
GPU="${GPU:-0}"
EPOCHS_P1="${EPOCHS_P1:-80}"
EPOCHS_P2="${EPOCHS_P2:-80}"
RUN_DIR="experiments/quickstart/synthetic/seed_${SEED}"

echo "[quickstart] 1. Generating synthetic double-well data..."
python scripts/data_prep/synthetic/generate_data.py
python scripts/data_prep/synthetic/prepare.py

echo "[quickstart] 2. Phase 1 (dense decoder, ${EPOCHS_P1} epochs)..."
python scripts/train/train_phase1.py \
    --data_dir data/synthetic_well_v2/source \
    --log_dir "${RUN_DIR}/phase1" \
    --z_dim 8 --hidden_dim 128 --hidden_per_z 32 \
    --decoder_type dense \
    --lambda_sparse 0 --lambda_diversity 0 --lambda_balance 0 \
    --lambda_col 0 --lambda_row 0 \
    --beta 0.002 --gamma_kld 0.02 \
    --warmup_epochs 0 --rampup_epochs 0 --freeze_epochs 0 \
    --epochs "${EPOCHS_P1}" --seed "${SEED}" --gpu "${GPU}"

echo "[quickstart] 3. Phase 2 (additive decoder + entropy sparsity, ${EPOCHS_P2} epochs)..."
python scripts/train/train_phase2.py \
    --data_dir data/synthetic_well_v2/source \
    --phase1_dir "${RUN_DIR}/phase1" \
    --log_dir "${RUN_DIR}/phase2" \
    --z_dim 8 --hidden_dim 128 --hidden_per_z 4 \
    --lambda_sparse 50 --beta 0.002 --gamma_kld 0.02 \
    --epochs "${EPOCHS_P2}" --seed "${SEED}" --gpu "${GPU}"

echo "[quickstart] 4. Evaluation..."
python scripts/eval/eval_unified.py \
    --benchmark synthetic \
    --ckpt_glob "${RUN_DIR}/phase2/lightning_logs/version_*/checkpoints/best-*.ckpt" \
    --data_dir data/synthetic_well_v2/source \
    --out "${RUN_DIR}/eval"

echo "[quickstart] Done. Results in ${RUN_DIR}/eval."
