#!/bin/bash
#SBATCH --job-name=datalab_ksweep
#SBATCH --account=m25146
#SBATCH --partition=mesonet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_ksweep_K%a.out
#SBATCH --error=slurm_ksweep_K%a.err
#SBATCH --array=2,8

export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"

cd /home/ayoudeba1/IASD/DataLab
source .venv/bin/activate

echo "=== K-sweep Job Info ==="
echo "Node: $(hostname)"
echo "K = ${SLURM_ARRAY_TASK_ID}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "========================"

python run_variant.py --variant annealed_wta --K ${SLURM_ARRAY_TASK_ID}
