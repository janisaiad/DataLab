#!/bin/bash
#SBATCH --job-name=datalab
#SBATCH --account=m25146
#SBATCH --partition=mesonet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"

cd /home/ayoudeba1/IASD/DataLab
source .venv/bin/activate

echo "=== Job Info ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "================"

python run_all.py
