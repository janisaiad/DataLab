#!/bin/bash
#SBATCH --job-name=batchsweep
#SBATCH --account=m25146
#SBATCH --partition=mesonet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_batch_B%a.out
#SBATCH --error=slurm_batch_B%a.err
#SBATCH --array=128,256,1024

export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
cd /home/ayoudeba1/IASD/DataLab
source .venv/bin/activate

echo "=== K=4, variant=annealed_wta, batch=${SLURM_ARRAY_TASK_ID} ==="
echo "Node: $(hostname)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
python run_variant.py --variant annealed_wta --K 4 --batch_size ${SLURM_ARRAY_TASK_ID}
