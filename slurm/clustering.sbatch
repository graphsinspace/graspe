#!/bin/bash
#SBATCH --job-name=clustering_eval
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --nodelist=n15
#SBATCH --time=120:00:00 #the maximum walltime in format h:m:s
#SBATCH --output slurm/slurm.%J.out # STDOUT
#SBATCH --error slurm/slurm.%J.err # STDERR
#SBATCH --export=ALL

srun /home/knezevicd/anaconda3/envs/graspe/bin/python src/graspe/evaluation/clustering_eval.py
