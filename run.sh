#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=01:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode29
#SBATCH --gres=gpu:rtx_a4500:1
#SBATCH --output=llm_output.out   # Standard output
#SBATCH --error=llm_error.err    # Standard error

source ../llm/bin/activate

# Test if the virtual environment is loaded properly
echo "Python executable: $(which python)"
export PYTHONUNBUFFERED=1
python ch5/pretraining.py