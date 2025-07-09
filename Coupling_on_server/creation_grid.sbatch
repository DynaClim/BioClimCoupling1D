#!/bin/bash
#SBATCH --job-name=creation_grille
#SBATCH --partition=shared-cpu
#SBATCH --time=12:00:00
#SBATCH --output=creation_grid.out
#SBATCH --error=creation_grid.err
#SBATCH --array=0-9

python Creation_grid_initial_concentration.py $SLURM_ARRAY_TASK_ID
