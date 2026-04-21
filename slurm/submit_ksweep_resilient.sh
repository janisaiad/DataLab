#!/bin/bash
#SBATCH --job-name=ksweep_res
#SBATCH --account=m25146
#SBATCH --partition=mesonet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_ksweep_res_K%a.out
#SBATCH --error=slurm_ksweep_res_K%a.err
#SBATCH --array=2,3,6,8

export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
cd /home/ayoudeba1/IASD/DataLab
source .venv/bin/activate

echo "=== K=${SLURM_ARRAY_TASK_ID}, variant=resilient_mcl, batch=512 ==="
echo "Node: $(hostname)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
python run_variant.py --variant resilient_mcl --K ${SLURM_ARRAY_TASK_ID}
