#!/bin/bash
#SBATCH --job-name=graspe
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --nodelist=n17
#SBATCH --time=24:00:00 #the maximum walltime in format h:m:s
#SBATCH --output slurm.%J.out # STDOUT
#SBATCH --error slurm.%J.err # STDERR
#SBATCH --export=ALL

srun /home/milosevicn/miniconda3/envs/graspe/bin/python src/graspe
