#!/bin/bash
#SBATCH --job-name=datalab_K2_B100
#SBATCH --account=m25146
#SBATCH --partition=mesonet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=slurm_K2_B100.out
#SBATCH --error=slurm_K2_B100.err

export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"

cd /home/ayoudeba1/IASD/DataLab
source .venv/bin/activate

echo "=== K=2, batch_size=100, annealed_wta ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

python run_variant.py --variant annealed_wta --K 2 --batch_size 100
