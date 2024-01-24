#!/bin/bash
#SBATCH -p Nvidia_A800
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem 500G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -J infer
#SBATCH -e logs/%j.err.log
#SBATCH -o logs/%j.log
#SBATCH -x gpu[1-8]

echo "START TIME: $(date)"

# python text_generation.py --stream

echo "END TIME: $(date)"
