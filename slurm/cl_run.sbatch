#!/bin/bash
#SBATCH --job-name=CL_RUN
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --output slurm.%J.out
#SBATCH --error slurm.%J.err
#SBATCH --export=ALL

srun /home/knezevicd/anaconda3/envs/graspe/bin/python src/graspe/evaluation/clustering_eval.py /home/shared/grasp/n2v_best_embeddings/