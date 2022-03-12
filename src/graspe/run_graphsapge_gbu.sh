#!/bin/bash
#SBATCH --job-name=GR_GBU
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=n16
#SBATCH --time=24:00:00
#SBATCH --output /home/stamenkovicd/slurm.%J.out
#SBATCH --error /home/stamenkovicd/slurm.%J.err
#SBATCH --export=ALL
srun /home/stamenkovicd/miniconda3/envs/graspe/bin/python /home/stamenkovicd/graspe/src/graspe/graphsage_gbu.py

