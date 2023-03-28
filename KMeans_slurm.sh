#!/bin/bash
#SBATCH --job-name=fuzzing
#SBATCH --output=out_slur
#SBATCH --gres=gpu:1
#SBATCH -o %A.out
#SBATCH -e %A.err

#export PYTHONPATH="/home/prober6/thesis/fuzzing_venv/bin/python"
#export PYTHONHOME="/home/prober6/thesis/fuzzing_venv/bin/python"

cd /home/prober6/scratch/fuzzing_thesis
source /home/prober6/scratch/new_fuzzing_venv/bin/activate

python k_means_clustered.py
