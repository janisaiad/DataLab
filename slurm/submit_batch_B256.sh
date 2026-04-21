#!/bin/bash
#SBATCH --job-name=batch_B256
#SBATCH --account=m25146
#SBATCH --partition=mesonet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_batch_B256.out
#SBATCH --error=slurm_batch_B256.err

export TMPDIR=$HOME/tmp
mkdir -p \"$TMPDIR\"
cd /home/ayoudeba1/IASD/DataLab
source .venv/bin/activate

echo \"=== K=4, variant=annealed_wta, batch=256 ===\"
python run_variant.py --variant annealed_wta --K 4 --batch_size 256
