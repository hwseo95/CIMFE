#!/usr/bin/env bash
# Pretraining hyperparameter sweep, run on GPU 0.
# A second grid (sweep_gpu1.sh) covers a complementary set of settings so the
# two can run in parallel across two GPUs.
set -euo pipefail
cd "$(dirname "$0")"

gpu=0
data_root=${DATA_ROOT:-../data}
sl=60; p_ep=100; norm=layernorm; pl=5; pretask=psi; p_lr=5e-3

for s in 2025 2026 2027; do
    els=(3 4)
    nhs=(16 32)
    dms=(128)
    alphas=(0.1 0.5 1)

    for el in "${els[@]}"; do
        for nh in "${nhs[@]}"; do
            for dm in "${dms[@]}"; do
                df=$((dm * 4))

                for alp in "${alphas[@]}"; do
                    python train.py --gpu "$gpu" --data_root "$data_root" \
                        --pretrain_epochs "$p_ep" --pretrain_learning_rate "$p_lr" \
                        --seq_len "$sl" --e_layers "$el" --d_ff "$df" --d_model "$dm" --n_heads "$nh" \
                        --patch_len "$pl" --stride "$pl" --seed "$s" \
                        --pretrain_task "$pretask" --alpha "$alp" --norm "$norm"
                done
            done
        done
    done
done
