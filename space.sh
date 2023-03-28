#!/bin/bash
#SBATCH --job-name=fuzzing
#SBATCH --output=out_slur
#SBATCH --gres=gpu:1
#SBATCH -o %A.out
#SBATCH -e %A.err

cd /home/prober6/scratch/fuzzing_thesis
source /home/prober6/scratch/new_fuzzing_venv/bin/activate

python space.py
