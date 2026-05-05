#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=def-papyan
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=300G
#SBATCH --time=00:10:00
#SBATCH --job-name=speedrun_lora
#SBATCH --output=runs/slurm-%j_%x.out
#SBATCH --error=runs/slurm-%j_%x.err

cd /home/yucz/nanochat

export NPROC_PER_NODE=4

bash runs/lora_sft.sh "$@"
