#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --account=def-papyan
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=300G
#SBATCH --time=00:10:00
#SBATCH --job-name=speedrun
#SBATCH --output=runs/slurm-%j_%x.out
#SBATCH --error=runs/slurm-%j_%x.err

cd /home/yucz/links/projects/def-papyan/yucz/nanochat

export NPROC_PER_NODE=4

bash runs/speedrun.sh "$@"
