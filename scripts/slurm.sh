#!/bin/bash
#SBATCH --job-name=fuzzing
#SBATCH --output=out_slur
#SBATCH --gres=gpu:1
#SBATCH -o %A.out
#SBATCH -e %A.err

cd /home/prober6/scratch/fuzzing_thesis
source /home/prober6/scratch/new_fuzzing_venv/bin/activate

python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=./saved_models/tokenizer_bert \
    --model_name_or_path=./saved_models/model_bert \
    --cache_dir=./saved_models/ \
    --do_train \
    --train_data_file=train.json \
    --eval_data_file=validate.json \
    --test_data_file=test.json \
    --epoch 100 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 \
    --warmup_steps 1000 
