#!/bin/bash
#SBATCH --job-name=NCLIDs
#SBATCH --partition=all
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4 
#SBATCH --nodelist=n01
#SBATCH --time=2-20:00:00 
#SBATCH --output /home/stamenkovicd/graspe/src/graspe/slurm.%J.out 
#SBATCH --error /home/stamenkovicd/graspe/src/graspe/slurm.%J.err 
#SBATCH --export=ALL 
srun /home/stamenkovicd/miniconda3/envs/graspe/bin/python /home/stamenkovicd/graspe/src/graspe/produce_nclids.py
  

