#!/bin/bash
#SBATCH --job-name=graspe2
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --nodelist=n01

#SBATCH --output slurm.%J.out # STDOUT
#SBATCH --error slurm.%J.err # STDERR
#SBATCH --export=ALL

srun env PYTHONUNBUFFERED=1 PYTHONPATH=src/graspe /home/tomcica/miniconda3/envs/graspe/bin/python src/graspe/scripts/customrws/aleksandars_rw.py amazon_electronics_computers