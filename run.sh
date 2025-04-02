#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=30G
#SBATCH --time=00:20:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode20
#SBATCH --gres=gpu:rtx_2070:1
#SBATCH --output=llm_output.out   # Standard output
#SBATCH --error=llm_error.err    # Standard error

source ../llm/bin/activate

# Test if the virtual environment is loaded properly
echo "Python executable: $(which python)"
export PYTHONUNBUFFERED=1
python instruction_finetuning/evaluate.py --eval-model openai