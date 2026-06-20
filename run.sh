#!/bin/bash
#SBATCH --job-name=simclr_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=97G
#SBATCH --time=02:00:00
#SBATCH --output=logs_train/run-%j.log
#SBATCH --qos=gpu_access

module add anaconda
conda activate simclr

PYTHON=/mnt/whitsett/yilinliu/miniconda3/envs/simclr/bin/python
export LD_LIBRARY_PATH=/mnt/whitsett/yilinliu/miniconda3/envs/simclr/lib:$LD_LIBRARY_PATH
EPOCHS=200

# Run one experiment: train then plot.
# Usage: run_exp <folder_name> [extra run.py args...]
run_exp() {
    local name=$1
    shift

    echo ""
    echo "========================================================"
    echo "  Experiment: $name"
    echo "========================================================"

    $PYTHON run.py --epochs $EPOCHS "$@"

    # The run directory is named by timestamp; pick the newest one.
    local run_dir
    run_dir=$(ls -1dt runs/*/ | head -1)
    run_dir="${run_dir%/}"   # strip trailing slash

    $PYTHON plot_results.py \
        --run_dir  "$run_dir" \
        --out_dir  "plots/$name" \
        --epochs   $EPOCHS

    echo "  Run  → $run_dir"
    echo "  Plots → plots/$name"
}

# ── Experiments (vary one hyperparameter at a time vs baseline) ─────────────
# Baseline: temperature=0.07, lr=3e-4, batch_size=32

run_exp "temp0.05_lr3e-4_bs32"   --temperature 0.05
run_exp "temp0.10_lr3e-4_bs32"   --temperature 0.10
run_exp "temp0.20_lr3e-4_bs32"   --temperature 0.20
run_exp "temp0.07_lr1e-3_bs32"   --lr 1e-3
run_exp "temp0.07_lr1e-4_bs32"   --lr 1e-4

echo ""
echo "All experiments complete."
echo "Plot directories:"
ls -d plots/temp*/
