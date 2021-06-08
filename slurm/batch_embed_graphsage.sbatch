#!/bin/bash
#SBATCH --job-name=grsage_be
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --nodelist=n17
#SBATCH --time=120:00:00 #the maximum walltime in format h:m:s
#SBATCH --output slurm/slurm.%J.out # STDOUT
#SBATCH --error slurm/slurm.%J.err # STDERR
#SBATCH --export=ALL

srun /home/milosevicn/miniconda3/envs/graspe/bin/python src/graspe batch_embed -g all -d 10 -p GraphSAGE -o /home/shared/grasp/embeddings/GraphSAGE/
