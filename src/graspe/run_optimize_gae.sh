#!/bin/bash
#SBATCH --job-name=GAEOPT
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=n16
#SBATCH --time=23-20:00:00
#SBATCH --output /home/stamendu/graspe/src/graspe/slurm.%J.out
#SBATCH --error /home/stamendu/graspe/src/graspe/slurm.%J.err
#SBATCH --export=ALL
srun /home/stamendu/miniconda3/envs/graspe/bin/python /home/stamendu/graspe/src/graspe/optimize_gae.py
