#!/bin/bash
#SBATCH --job-name=CULP_RUN
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
###SBATCH --nodelist=n16
#SBATCH --cpus-per-task=10
#SBATCH --output slurm.%J.out
#SBATCH --error slurm.%J.err
#SBATCH --export=ALL

srun env PYTHONUNBUFFERED=1 PYTHONPATH=src/graspe /home/knezevicd/anaconda3/envs/graspe/bin/python src/graspe/evaluation/unsupervised_link_prediction.py /home/shared/grasp/lidn2v_best_embeddings_2/