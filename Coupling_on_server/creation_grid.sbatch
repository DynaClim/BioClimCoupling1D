#!/bin/bash
#SBATCH --job-name=creation_grille
#SBATCH --partition=shared-cpu
#SBATCH --time=12:00:00
#SBATCH --output=creation_grid_v2.out
#SBATCH --error=creation_grid_v2.err
#SBATCH --array=0-9
# Charger le module Python si nécessaire (exemple : module load python/3.10)
# module load python

# Lancer le script Python
python Creation_grille_concentrations_val_ini.py $SLURM_ARRAY_TASK_ID
