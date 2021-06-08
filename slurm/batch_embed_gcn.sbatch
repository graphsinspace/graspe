#!/bin/bash
#SBATCH --job-name=gcn_be
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --nodelist=n01
#SBATCH --time=120:00:00 #the maximum walltime in format h:m:s
#SBATCH --output slurm/slurm.%J.out # STDOUT
#SBATCH --error slurm/slurm.%J.err # STDERR
#SBATCH --export=ALL

srun /home/milosevicn/miniconda3/envs/graspe/bin/python src/graspe batch_embed -g all -d 10 -p GCN -o /home/shared/grasp/embeddings/gcn/
